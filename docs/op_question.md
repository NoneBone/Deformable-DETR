# 最简测试
## 指令
```sh
cd ./models/ops
sh ./make.sh
python test.py
# quick nsys
nsys profile \
  --trace=cuda,cublas,cudnn,nvtx \
  --sample=none \
  --output=../../docs/nsys/naive \
  --force-overwrite true \
  --gpu-metrics-devices=all \
python test.py
python benchmark.py

```
可见最耗时算子为：
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
56.7%	116.272 ms	15415	7.542 μs	9.343 μs	4.064 μs	9.664 μs	1.871 μs	void ms_deformable_im2col_gpu_kernel<double>(int, const T1 *, const long *, const long *, const T1 *, const T1 *, int, int, int, int, int, int, int, T1 *)

## 计算逻辑概述
内核行为：
- 随机采样点导致线程不规则的访存，从 GMEM 读取可能存在未合并问题；
- 单线程完成双线性插值及最后的 attention 计算，很容易计算密集；
- 内核计算复杂度：层数 * 采样点数

## ncu 问题

full 采集无法进行，能否只采集关键指标？待尝试。
直接采用权限为 1 的机器验证。
```sh
cat /proc/sys/kernel/perf_event_paranoid
sudo sh -c ‘echo 1 >/proc/sys/kernel/perf_event_paranoid’
sudo sh -c ‘echo kernel.perf_event_paranoid=1 > /etc/sysctl.d/local.conf’
sudo sysctl kernel.perf_event_paranoid=1
cat /proc/sys/kernel/perf_event_paranoid
```

```sh
# ncu
skip=12
launch=12
sudo /usr/local/cuda-12.8/bin/ncu --set full -o  \
../../docs/nsys/detr3.ncu-rep \
-k regex:"attn|defor" \
--launch-skip "${skip}" --launch-count "${launch}" \
-f \
/home/l8w/miniconda3/envs/cp312/bin/python test.py
```
## 简单分析

- 预测加速比为 2x。目前报告问题包括：

1. 理论 occupancy 受限于寄存器使用，导致仅 66%，而实际 occupancy 仅为 13%，过低。
2. 只采用了 1 个块的 1024 个线程，计算量 really 很小？目前的计算与访存吞吐都极低。
3. DRAM视角为 mb， L1/L2视角为 cb，smem 未被使用。而常量mem，存在频繁的数据读取，数据加载拖慢整体速度。

# benchmark 测试

## naive
```sh
=== MSDeformAttn forward tests ===

[PASS] FP64 B=1  Q=2    H=2  C=2  L=2 P=2 step=1  (sanity)                      [MAX] abs_err=8.674e-19 rel_err=1.831e-16
[PASS] FP32 B=2  Q=32   H=4  C=8  L=3 P=4 step=2  (multi-level)                 [MAX] abs_err=1.397e-09 rel_err=2.593e-07
[PASS] FP32 B=2  Q=128  H=8  C=16 L=4 P=4 step=2  (detector-like)               [MAX] abs_err=1.397e-09 rel_err=3.321e-07

All forward tests PASSED.

=== MSDeformAttn forward benchmarks (warmup=10, iters=100) ===
[BENCH] FP32 B=1  Q=300  H=8  C=32 L=4 P=4 step=1  (deformable-detr)                  21.55 us   294.07 GB/s
[BENCH] FP32 B=2  Q=300  H=8  C=32 L=4 P=4 step=2  (batched-2)                        34.71 us   365.25 GB/s
[BENCH] FP32 B=4  Q=300  H=8  C=32 L=4 P=4 step=4  (batched-4)                        61.31 us   413.54 GB/s
[BENCH] FP32 B=16 Q=300  H=8  C=32 L=4 P=4 step=16 (batched-16)                      379.86 us   266.98 GB/s
[BENCH] FP32 B=64 Q=300  H=8  C=32 L=4 P=4 step=64 (batched-64)                     1525.58 us   265.91 GB/s
[BENCH] FP32 B=256 Q=300  H=8  C=32 L=4 P=4 step=256 (batched-256)                  6042.61 us   268.54 GB/s
[BENCH] FP32 B=1024 Q=300  H=8  C=32 L=4 P=4 step=1024 (batched-1024)              24199.16 us   268.22 GB/s
```

## opt1.smem 

一个 block 对应一个 (b, q, m)。block 内线程沿 channel 维展开。
然后所有 channel 线程复用这份 shared metadata 去读 data_value 并累加。

- 计算：
  - 计算任务重新组织：从 "1线程=1输出" 到 "1 Block=1采样组"：一个 block 内所有线程协作，每个采样点只计算一次双线性系数
  - 预计算双线性权重，分离"地址计算"与"数据读取"：先把当前 (b,q,m) 的 num_levels * num_point 个采样点的 4 邻域指针偏移和 4 个双线性系数预计算到 shared memory。
  - 双线性入口的边界检查
- 访存：
  - 全局内存读写模式优化
  - __restrict__+ 只读缓存：让编译器推断采用常量或纹理 mem


问题：
- SMEM 的store 存在19200/24000 的 BC，
- 额外引入了一定的 WD；

```sh
=== MSDeformAttn forward tests ===

[PASS] FP64 B=1  Q=2    H=2  C=2  L=2 P=2 step=1  (sanity)                      [MAX] abs_err=8.674e-19 rel_err=2.396e-16
[PASS] FP32 B=2  Q=32   H=4  C=8  L=3 P=4 step=2  (multi-level)                 [MAX] abs_err=1.863e-09 rel_err=4.372e-07
[PASS] FP32 B=2  Q=128  H=8  C=16 L=4 P=4 step=2  (detector-like)               [MAX] abs_err=2.794e-09 rel_err=5.488e-07

All forward tests PASSED.

=== MSDeformAttn forward benchmarks (warmup=10, iters=100) ===
[BENCH] FP32 B=1  Q=300  H=8  C=32 L=4 P=4 step=1  (deformable-detr)                  16.41 us   386.37 GB/s
[BENCH] FP32 B=2  Q=300  H=8  C=32 L=4 P=4 step=2  (batched-2)                        28.81 us   440.01 GB/s
[BENCH] FP32 B=4  Q=300  H=8  C=32 L=4 P=4 step=4  (batched-4)                        52.29 us   484.92 GB/s
[BENCH] FP32 B=16 Q=300  H=8  C=32 L=4 P=4 step=16 (batched-16)                      344.39 us   294.48 GB/s
[BENCH] FP32 B=64 Q=300  H=8  C=32 L=4 P=4 step=64 (batched-64)                     1401.75 us   289.40 GB/s
[BENCH] FP32 B=256 Q=300  H=8  C=32 L=4 P=4 step=256 (batched-256)                  5556.07 us   292.05 GB/s
[BENCH] FP32 B=1024 Q=300  H=8  C=32 L=4 P=4 step=1024 (batched-1024)              22221.82 us   292.09 GB/s
```

## opt2.smem double buffer

基于 opt1，无效的进一步优化:
- float4 向量化加载，__ldg 加载，反而变慢了，这符合计算密集型 op 的特性。
- doublebuffer 向量化加载。

```sh
# float4
=== MSDeformAttn forward tests ===

[PASS] FP64 B=1  Q=2    H=2  C=2  L=2 P=2 step=1  (sanity)                      [MAX] abs_err 8.674e-19 rel_err 2.396e-16
[PASS] FP32 B=2  Q=32   H=4  C=8  L=3 P=4 step=2  (multi-level)                 [MAX] abs_err 1.863e-09 rel_err 4.372e-07
[PASS] FP32 B=2  Q=128  H=8  C=16 L=4 P=4 step=2  (detector-like)               [MAX] abs_err 2.794e-09 rel_err 5.488e-07

All forward tests PASSED.

=== MSDeformAttn forward benchmarks (warmup=10, iters=100) ===
[BENCH] FP32 B=1  Q=300  H=8  C=32 L=4 P=4 step=1  (deformable-detr)                  16.80 us   377.31 GB/s
[BENCH] FP32 B=2  Q=300  H=8  C=32 L=4 P=4 step=2  (batched-2)                        29.89 us   424.13 GB/s
[BENCH] FP32 B=4  Q=300  H=8  C=32 L=4 P=4 step=4  (batched-4)                        54.83 us   462.40 GB/s
[BENCH] FP32 B=16 Q=300  H=8  C=32 L=4 P=4 step=16 (batched-16)                      349.42 us   290.25 GB/s
[BENCH] FP32 B=64 Q=300  H=8  C=32 L=4 P=4 step=64 (batched-64)                     1424.07 us   284.86 GB/s
[BENCH] FP32 B=256 Q=300  H=8  C=32 L=4 P=4 step=256 (batched-256)                  5633.47 us   288.04 GB/s
[BENCH] FP32 B=1024 Q=300  H=8  C=32 L=4 P=4 step=1024 (batched-1024)              22544.51 us   287.91 GB/s

# doubleBuffer
=== MSDeformAttn forward tests ===

[PASS] FP64 B=1  Q=2    H=2  C=2  L=2 P=2 step=1  (sanity)                      [MAX] abs_err 8.674e-19 rel_err 2.396e-16
[PASS] FP32 B=2  Q=32   H=4  C=8  L=3 P=4 step=2  (multi-level)                 [MAX] abs_err 1.863e-09 rel_err 4.372e-07
[PASS] FP32 B=2  Q=128  H=8  C=16 L=4 P=4 step=2  (detector-like)               [MAX] abs_err 2.794e-09 rel_err 5.488e-07

All forward tests PASSED.

=== MSDeformAttn forward benchmarks (warmup=10, iters=100) ===
[BENCH] FP32 B=1  Q=300  H=8  C=32 L=4 P=4 step=1  (deformable-detr)                  17.04 us   371.93 GB/s
[BENCH] FP32 B=2  Q=300  H=8  C=32 L=4 P=4 step=2  (batched-2)                        29.96 us   423.11 GB/s
[BENCH] FP32 B=4  Q=300  H=8  C=32 L=4 P=4 step=4  (batched-4)                        55.00 us   460.95 GB/s
[BENCH] FP32 B=16 Q=300  H=8  C=32 L=4 P=4 step=16 (batched-16)                      351.13 us   288.83 GB/s
[BENCH] FP32 B=64 Q=300  H=8  C=32 L=4 P=4 step=64 (batched-64)                     1428.01 us   284.08 GB/s
[BENCH] FP32 B=256 Q=300  H=8  C=32 L=4 P=4 step=256 (batched-256)                  5664.16 us   286.48 GB/s
[BENCH] FP32 B=1024 Q=300  H=8  C=32 L=4 P=4 step=1024 (batched-1024)              22695.42 us   285.99 GB/s
```

## opt3.naive+float4
保持 opt0 的线程组织与循环结构，仅在 `float32 + channels % 4 == 0` 时改为
“1 线程 = 1 float4”，用于隔离验证向量化本身的收益。

```sh
=== MSDeformAttn forward tests ===

[PASS] FP64 B=1  Q=2    H=2  C=2  L=2 P=2 step=1  (sanity)                      [MAX] abs_err 8.674e-19 rel_err 1.978e-16
[PASS] FP32 B=2  Q=32   H=4  C=8  L=3 P=4 step=2  (multi-level)                 [MAX] abs_err 4.657e-10 rel_err 1.130e-07
[PASS] FP32 B=2  Q=128  H=8  C=16 L=4 P=4 step=2  (detector-like)               [MAX] abs_err 4.657e-10 rel_err 1.130e-07

All forward tests PASSED.

=== MSDeformAttn forward benchmarks (warmup=10, iters=100) ===
[BENCH] FP32 B=1  Q=300  H=8  C=32 L=4 P=4 step=1  (deformable-detr)                  15.24 us   415.90 GB/s
[BENCH] FP32 B=2  Q=300  H=8  C=32 L=4 P=4 step=2  (batched-2)                        31.36 us   404.31 GB/s
[BENCH] FP32 B=4  Q=300  H=8  C=32 L=4 P=4 step=4  (batched-4)                        51.84 us   489.12 GB/s
[BENCH] FP32 B=16 Q=300  H=8  C=32 L=4 P=4 step=16 (batched-16)                      313.81 us   323.18 GB/s
[BENCH] FP32 B=64 Q=300  H=8  C=32 L=4 P=4 step=64 (batched-64)                     1265.29 us   320.61 GB/s
[BENCH] FP32 B=256 Q=300  H=8  C=32 L=4 P=4 step=256 (batched-256)                  5002.58 us   324.37 GB/s
[BENCH] FP32 B=1024 Q=300  H=8  C=32 L=4 P=4 step=1024 (batched-1024)              19954.29 us   325.28 GB/s
```

## opt4.v3+reg_prefetch
在 opt3 的基础上，只做 one-step 寄存器级预取：
当前 sample 计算时，提前把下一个 sample 的 `loc/weight/shape/level_start`
读到寄存器，验证是否存在可被软件流水隐藏的 latency。

```sh
=== MSDeformAttn forward tests ===

[PASS] FP64 B=1  Q=2    H=2  C=2  L=2 P=2 step=1  (sanity)                      [MAX] abs_err 8.674e-19 rel_err 1.831e-16
[PASS] FP32 B=2  Q=32   H=4  C=8  L=3 P=4 step=2  (multi-level)                 [MAX] abs_err 1.397e-09 rel_err 2.593e-07
[PASS] FP32 B=2  Q=128  H=8  C=16 L=4 P=4 step=2  (detector-like)               [MAX] abs_err 1.397e-09 rel_err 3.321e-07

All forward tests PASSED.

=== MSDeformAttn forward benchmarks (warmup=10, iters=100) ===
[BENCH] FP32 B=1  Q=300  H=8  C=32 L=4 P=4 step=1  (deformable-detr)                  15.33 us   413.54 GB/s
[BENCH] FP32 B=2  Q=300  H=8  C=32 L=4 P=4 step=2  (batched-2)                        31.54 us   401.99 GB/s
[BENCH] FP32 B=4  Q=300  H=8  C=32 L=4 P=4 step=4  (batched-4)                        51.47 us   492.57 GB/s
[BENCH] FP32 B=16 Q=300  H=8  C=32 L=4 P=4 step=16 (batched-16)                      313.58 us   323.41 GB/s
[BENCH] FP32 B=64 Q=300  H=8  C=32 L=4 P=4 step=64 (batched-64)                     1263.61 us   321.04 GB/s
[BENCH] FP32 B=256 Q=300  H=8  C=32 L=4 P=4 step=256 (batched-256)                  4993.21 us   324.98 GB/s
[BENCH] FP32 B=1024 Q=300  H=8  C=32 L=4 P=4 step=1024 (batched-1024)              19906.06 us   326.07 GB/s
```

## opt5.bf16实现，TC 支持


```sh

```
# 集成测试

## 算法架构

- transformer：编解码，（linear+layerNorm）*2

```
# opt0
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
10.7%	546.406 ms	600	910.677 μs	673.095 μs	25.632 μs	2.798 ms	927.264 μs	void ms_deformable_im2col_gpu_kernel<float>(int, const T1 *, const long *, const long *, const T1 *, const T1 *, int, int, int, int, int, int, int, T1 *)
# opt1 1.37x
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
8.0%	397.700 ms	600	662.833 μs	490.228 μs	20.992 μs	2.038 ms	671.912 μs	void ms_deformable_im2col_gpu_kernel<float>(int, const T1 *, const long *, const long *, const T1 *, const T1 *, int, int, int, int, int, int, int, T1 *)
# opt3 1.90x
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
5.9%	287.526 ms	600	479.209 μs	349.420 μs	19.391 μs	1.486 ms	481.262 μs	void ms_deformable_im2col_gpu_kernel<float>(int, const T1 *, const long *, const long *, const T1 *, const T1 *, int, int, int, int, int, int, int, T1 *)
# 

```