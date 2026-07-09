from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

OPS_DIR = Path(__file__).resolve().parent
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch


ShapeSpec = Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class ForwardCase:
    name: str
    batch: int
    num_query: int
    num_heads: int
    channels: int
    num_points: int
    spatial_shapes: ShapeSpec
    im2col_step: int
    dtype: torch.dtype = torch.float32

    @property
    def num_levels(self):
        return len(self.spatial_shapes)


def get_benchmark_arg_parser():
    parser = argparse.ArgumentParser("Benchmark MSDeformAttn forward.")
    parser.add_argument("--num-iters", type=int, default=100, help="number of benchmark iterations")
    parser.add_argument("--warm-iters", type=int, default=10, help="number of warmup iterations")
    parser.add_argument("--seed", type=int, default=3, help="random seed")
    parser.add_argument("--skip-check", action="store_true", help="skip forward correctness checks")
    parser.add_argument("--skip-bench", action="store_true", help="skip forward benchmarks")
    parser.add_argument(
        "--bench-pytorch-ref",
        action="store_true",
        help="also benchmark the PyTorch reference implementation",
    )
    return parser


def _format_dtype(dtype):
    if dtype == torch.float64:
        return "FP64"
    if dtype == torch.float32:
        return "FP32"
    return str(dtype).replace("torch.", "").upper()


def _format_case(case):
    return (
        "{dtype} B={batch:<2d} Q={num_query:<4d} H={num_heads:<2d} "
        "C={channels:<2d} L={num_levels:<1d} P={num_points:<1d} "
        "step={im2col_step:<2d} ({name})"
    ).format(
        dtype=_format_dtype(case.dtype),
        batch=case.batch,
        num_query=case.num_query,
        num_heads=case.num_heads,
        channels=case.channels,
        num_levels=case.num_levels,
        num_points=case.num_points,
        im2col_step=case.im2col_step,
        name=case.name,
    )


def _make_level_start_index(spatial_shapes):
    return torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]),
        dim=0,
    ).contiguous()


def _normalize_attention_weights(attention_weights):
    denom = attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    return (attention_weights / denom).contiguous()


def _build_inputs(case, device):
    spatial_shapes = torch.as_tensor(case.spatial_shapes, dtype=torch.long, device=device).contiguous()
    level_start_index = _make_level_start_index(spatial_shapes)
    spatial_size = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())

    value = (torch.rand(case.batch, spatial_size, case.num_heads, case.channels, device=device, dtype=case.dtype) * 0.01).contiguous()
    sampling_locations = torch.rand(
        case.batch,
        case.num_query,
        case.num_heads,
        case.num_levels,
        case.num_points,
        2,
        device=device,
        dtype=case.dtype,
    ).contiguous()
    attention_weights = _normalize_attention_weights(
        torch.rand(
            case.batch,
            case.num_query,
            case.num_heads,
            case.num_levels,
            case.num_points,
            device=device,
            dtype=case.dtype,
        ) + 1e-5
    )
    return value, spatial_shapes, level_start_index, sampling_locations, attention_weights


def _max_relative_error(output_cuda, output_ref):
    diff = (output_cuda - output_ref).abs()
    denom = output_ref.abs().clamp_min(1e-12)
    return (diff / denom).max().item()


@torch.no_grad()
def run_forward_check(case):
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = _build_inputs(case, "cuda")

    output_ref = ms_deform_attn_core_pytorch(
        value, spatial_shapes, sampling_locations, attention_weights
    ).detach()
    output_cuda = MSDeformAttnFunction.apply(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        case.im2col_step,
    ).detach()

    if case.dtype == torch.float64:
        rtol, atol = 1e-5, 1e-8
    else:
        rtol, atol = 1e-2, 1e-3

    passed = torch.allclose(output_cuda, output_ref, rtol=rtol, atol=atol)
    max_abs_err = (output_cuda - output_ref).abs().max().item()
    max_rel_err = _max_relative_error(output_cuda, output_ref)
    tag = "PASS" if passed else "FAIL"
    print(
        "[{tag}] {case:<72s} [MAX] abs_err {max_abs_err:.3e} rel_err {max_rel_err:.3e}".format(
            tag=tag,
            case=_format_case(case),
            max_abs_err=max_abs_err,
            max_rel_err=max_rel_err,
        )
    )
    return passed


@torch.no_grad()
def _benchmark_impl(fn, warm_iters, num_iters):
    for _ in range(warm_iters):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_iters


def _estimate_io_gbps(tensors, avg_seconds):
    total_bytes = sum(t.numel() * t.element_size() for t in tensors)
    return total_bytes / avg_seconds / 1e9


@torch.no_grad()
def run_forward_benchmark(case, warm_iters, num_iters, benchmark_pytorch_ref=False):
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = _build_inputs(case, "cuda")

    def run_cuda():
        return MSDeformAttnFunction.apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            case.im2col_step,
        )

    output = run_cuda()
    avg_seconds = _benchmark_impl(run_cuda, warm_iters, num_iters)
    gbps = _estimate_io_gbps(
        (value, sampling_locations, attention_weights, output),
        avg_seconds,
    )
    print(
        "[BENCH] {case:<72s} {latency_us:>10.2f} us {gbps:>8.2f} GB/s".format(
            case=_format_case(case),
            latency_us=avg_seconds * 1e6,
            gbps=gbps,
        )
    )

    if benchmark_pytorch_ref:
        def run_ref():
            return ms_deform_attn_core_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        ref_output = run_ref()
        ref_avg_seconds = _benchmark_impl(run_ref, warm_iters, num_iters)
        ref_gbps = _estimate_io_gbps(
            (value, sampling_locations, attention_weights, ref_output),
            ref_avg_seconds,
        )
        speedup = ref_avg_seconds / avg_seconds
        print(
            "[REF  ] {case:<72s} {latency_us:>10.2f} us {gbps:>8.2f} GB/s speedup={speedup:.2f}x".format(
                case=_format_case(case),
                latency_us=ref_avg_seconds * 1e6,
                gbps=ref_gbps,
                speedup=speedup,
            )
        )


def _default_check_cases():
    return (
        ForwardCase(
            name="sanity",
            batch=1,
            num_query=2,
            num_heads=2,
            channels=2,
            num_points=2,
            spatial_shapes=((6, 4), (3, 2)),
            im2col_step=1,
            dtype=torch.float64,
        ),
        ForwardCase(
            name="multi-level",
            batch=2,
            num_query=32,
            num_heads=4,
            channels=8,
            num_points=4,
            spatial_shapes=((12, 8), (6, 4), (3, 2)),
            im2col_step=2,
            dtype=torch.float32,
        ),
        ForwardCase(
            name="detector-like",
            batch=2,
            num_query=128,
            num_heads=8,
            channels=16,
            num_points=4,
            spatial_shapes=((24, 16), (12, 8), (6, 4), (3, 2)),
            im2col_step=2,
            dtype=torch.float32,
        ),
    )


def _default_benchmark_cases():
    return (
        ForwardCase(
            name="deformable-detr",
            batch=1,
            num_query=300,
            num_heads=8,
            channels=32,
            num_points=4,
            spatial_shapes=((64, 64), (32, 32), (16, 16), (8, 8)),
            im2col_step=1,
            dtype=torch.float32,
        ),
        ForwardCase(
            name="batched-2",
            batch=2,
            num_query=300,
            num_heads=8,
            channels=32,
            num_points=4,
            spatial_shapes=((64, 64), (32, 32), (16, 16), (8, 8)),
            im2col_step=2,
            dtype=torch.float32,
        ),
        ForwardCase(
            name="batched-4",
            batch=4,
            num_query=300,
            num_heads=8,
            channels=32,
            num_points=4,
            spatial_shapes=((64, 64), (32, 32), (16, 16), (8, 8)),
            im2col_step=4,
            dtype=torch.float32,
        ),
        ForwardCase(
                    name="batched-16",
                    batch=16,
                    num_query=300,
                    num_heads=8,
                    channels=32,
                    num_points=4,
                    spatial_shapes=((64, 64), (32, 32), (16, 16), (8, 8)),
                    im2col_step=16,
                    dtype=torch.float32,
        ),
        ForwardCase(
            name="batched-64",
            batch=64,
            num_query=300,
            num_heads=8,
            channels=32,
            num_points=4,
            spatial_shapes=((64, 64), (32, 32), (16, 16), (8, 8)),
            im2col_step=64,
            dtype=torch.float32,
        ),
        ForwardCase(
            name="batched-256",
            batch=256,
            num_query=300,
            num_heads=8,
            channels=32,
            num_points=4,
            spatial_shapes=((64, 64), (32, 32), (16, 16), (8, 8)),
            im2col_step=256,
            dtype=torch.float32,
        ),
        ForwardCase(
            name="batched-1024",
            batch=1024,
            num_query=300,
            num_heads=8,
            channels=32,
            num_points=4,
            spatial_shapes=((64, 64), (32, 32), (16, 16), (8, 8)),
            im2col_step=1024,
            dtype=torch.float32,
        ),
    )


def _validate_args(args):
    if args.num_iters <= 0:
        raise ValueError("--num-iters must be positive")
    if args.warm_iters < 0:
        raise ValueError("--warm-iters must be non-negative")
    if args.warm_iters >= args.num_iters:
        raise ValueError("--warm-iters must be smaller than --num-iters")


def _check_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because MSDeformAttnFunction forward only supports CUDA tensors.")


def benchmark():
    args = get_benchmark_arg_parser().parse_args()
    _validate_args(args)
    _check_runtime()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    all_pass = True

    if not args.skip_check:
        print("=== MSDeformAttn forward tests ===\n")
        for case in _default_check_cases():
            all_pass &= run_forward_check(case)
        print(
            "\n{summary}\n".format(
                summary="All forward tests PASSED." if all_pass else "Some forward tests FAILED."
            )
        )

    if not args.skip_bench:
        print(
            "=== MSDeformAttn forward benchmarks (warmup={warm_iters}, iters={num_iters}) ===".format(
                warm_iters=args.warm_iters,
                num_iters=args.num_iters,
            )
        )
        for case in _default_benchmark_cases():
            run_forward_benchmark(case, args.warm_iters, args.num_iters, args.bench_pytorch_ref)

    return all_pass


if __name__ == "__main__":
    raise SystemExit(0 if benchmark() else 1)
