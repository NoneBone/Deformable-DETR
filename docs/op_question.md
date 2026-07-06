
```sh
cd ./models/ops
sh ./make.sh
python test.py
# quick nsys
nsys profile \
  --trace=cuda,cublas,cudnn,nvtx \
  --sample=none \
  --output=../../docs/nsys/naive.nsys-rep \
  --force-overwrite true \
  python test.py

```
可见最耗时算子为：
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
56.7%	116.272 ms	15415	7.542 μs	9.343 μs	4.064 μs	9.664 μs	1.871 μs	void ms_deformable_im2col_gpu_kernel<double>(int, const T1 *, const long *, const long *, const T1 *, const T1 *, int, int, int, int, int, int, int, T1 *)
具体计算逻辑为：

# ncu 问题
full 采集无法进行，能否只采集关键指标？待尝试。

```sh
# ncu
ncu --set full -o  \
../../docs/nsys/naive.ncu-rep \
-f \
python test.py

# -k regex:"attn|forward" \
```