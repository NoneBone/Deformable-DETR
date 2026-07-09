#### Quick NSYS
```sh
# para.
op=demo_e2e
# cmd
NSYS_DIR=./A-LLM-PERF/$op/NSYS
mkdir -p "${NSYS_DIR}" || { echo "X Failed to create the directory"; exit 1; }
/usr/local/cuda-12.8/bin/nsys profile --stats=true --force-overwrite=true -o $NSYS_DIR/test_$op python3 demo_e2e.py

# e2e
/usr/local/cuda-12.8/bin/nsys profile \
  -w true \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s cpu \
  --cudabacktrace=true \
  -x true \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --gpuctxsw=true \
  --gpu-metrics-devices=none \
  -o $NSYS_DIR/test_$op \
  python3 demo_e2e.py

```

#### Quick NCU

```sh
# para.
op=demo_e2e
skip=10
launch=10
# cmd
NCU_DIR=./A-LLM-PERF/$op/NCU
mkdir -p "${NCU_DIR}" || { echo "X Failed to create the directory"; exit 1; }

export TORCH_CUDA_ARCH_LIST="8.9"

sudo /usr/local/cuda-12.8/bin/ncu  --set full --launch-skip "${skip}" --launch-count "${launch}" -k regex:"attention" --target-processes all -f -o $NCU_DIR/attn_$op.ncu-rep /home/l8w/miniconda3/envs/cp310/bin/python demo_e2e.py

# |rms|Kernel|gemv2T
num=9
sudo /usr/local/cuda-12.8/bin/ncu \
  --import $NCU_DIR/test_$op-$num.ncu-rep \
  --csv \
  > $NCU_DIR/test_$op-$num.csv
```