
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

```
可见最耗时算子为：
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
56.7%	116.272 ms	15415	7.542 μs	9.343 μs	4.064 μs	9.664 μs	1.871 μs	void ms_deformable_im2col_gpu_kernel<double>(int, const T1 *, const long *, const long *, const T1 *, const T1 *, int, int, int, int, int, int, int, T1 *)
具体计算逻辑为：

# ncu 问题

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

- 预测加速比为 2x。目前报告问题包括：

1. 理论 occupancy 受限于寄存器使用，导致仅 66%，而实际 occupancy 仅为 13%，过低。
2. 只采用了 1 个块的 1024 个线程，计算量 really 很小？目前的计算与访存吞吐都极低。
3. DRAM视角为 mb， L1/L2视角为 cb，smem 未被使用。而常量mem，存在频繁的数据读取，数据加载拖慢整体速度。