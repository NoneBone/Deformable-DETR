# Deformable DETR

作者：[Xizhou Zhu](https://scholar.google.com/citations?user=02RXI00AAAAJ)、[Weijie Su](https://www.weijiesu.com/)、[Lewei Lu](https://www.linkedin.com/in/lewei-lu-94015977/)、[Bin Li](http://staff.ustc.edu.cn/~binli/)、[Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/)、[Jifeng Dai](https://jifengdai.org/)。

本仓库是论文 [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)的官方实现。

## 简介

**太长不看。** Deformable DETR 是一种高效且收敛快速的端到端目标检测器。它通过一种新颖的基于采样的注意力机制，缓解了 DETR 计算复杂度高和收敛慢的问题。

![deformable_detr](./figs/illustration.png)

![deformable_detr](./figs/convergence.png)


**摘要。** DETR 近期被提出，旨在消除目标检测中许多手工设计的组件，同时展现出良好的性能。然而，由于 Transformer 注意力模块在处理图像特征图时的局限性，它存在收敛缓慢和特征空间分辨率有限的问题。为缓解这些问题，我们提出了 Deformable DETR，其注意力模块仅关注参考点周围的一小部分关键采样点。Deformable DETR 仅需 DETR 十分之一的训练轮数，即可取得更优的性能（尤其在小目标上）。在 COCO 基准上的大量实验证明了我们方法的有效性。

## 许可证

本项目基于 [Apache 2.0 许可证](LICENSE)发布。

## 更新日志

详细的主要变更记录请见 [changelog.md](docs/changelog.md)。

## 引用 Deformable DETR

如果您的研究中使用了 Deformable DETR，请考虑引用：

```
@article{zhu2020deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```

## 主要结果

| 方法                              | 训练轮数 |  AP  | APS  | APM  | APL  | 参数量 (M) | 计算量 (G) | 总训练 时间 (GPU 小时) | 训练速度 (GPU 小时 /轮) | 推理速度 (FPS) | 批量推理 速度 (FPS) | 链接                                                         |
| --------------------------------- | :------: | :--: | :--: | :--: | :--: | :--------: | :--------: | :--------------------: | :---------------------: | :------------: | ------------------- | ------------------------------------------------------------ |
| Faster R-CNN + FPN                |   109    | 42.0 | 26.6 | 45.4 | 53.4 |     42     |    180     |          380           |           3.5           |      25.6      | 28.0                | -                                                            |
| DETR                              |   500    | 42.0 | 20.5 | 45.8 | 61.1 |     41     |     86     |          2000          |           4.0           |      27.0      | 38.3                | -                                                            |
| DETR-DC5                          |   500    | 43.3 | 22.5 | 47.3 | 61.1 |     41     |    187     |          7000          |          14.0           |      11.4      | 12.4                | -                                                            |
| DETR-DC5                          |    50    | 35.3 | 15.2 | 37.5 | 53.6 |     41     |    187     |          700           |          14.0           |      11.4      | 12.4                | -                                                            |
| DETR-DC5+                         |    50    | 36.2 | 16.3 | 39.2 | 53.9 |     41     |    187     |          700           |          14.0           |      11.4      | 12.4                | -                                                            |
| **Deformable DETR (单尺度)**      |    50    | 39.4 | 20.6 | 43.0 | 55.5 |     34     |     78     |          160           |           3.2           |      27.0      | 42.4                | [配置文件](configs/r50_deformable_detr_single_scale.sh) [日志](https://drive.google.com/file/d/1n3ZnZ-UAqmTUR4AZoM4qQntIDn6qCZx4/view?usp=sharing) [模型](https://drive.google.com/file/d/1WEjQ9_FgfI5sw5OZZ4ix-OKk-IJ_-SDU/view?usp=sharing) |
| **Deformable DETR (单尺度, DC5)** |    50    | 41.5 | 24.1 | 45.3 | 56.0 |     34     |    128     |          215           |           4.3           |      22.1      | 29.4                | [配置文件](configs/r50_deformable_detr_single_scale_dc5.sh) [日志](https://drive.google.com/file/d/1-UfTp2q4GIkJjsaMRIkQxa5k5vn8_n-B/view?usp=sharing) [模型](https://drive.google.com/file/d/1m_TgMjzH7D44fbA-c_jiBZ-xf-odxGdk/view?usp=sharing) |
| **Deformable DETR**               |    50    | 44.5 | 27.1 | 47.6 | 59.6 |     40     |    173     |          325           |           6.5           |      15.0      | 19.4                | [配置文件](configs/r50_deformable_detr.sh) [日志](https://drive.google.com/file/d/18YSLshFjc_erOLfFC-hHu4MX4iyz1Dqr/view?usp=sharing) [模型](https://drive.google.com/file/d/1nDWZWHuRwtwGden77NLM9JoWe-YisJnA/view?usp=sharing) |
| **+ 迭代边界框优化**              |    50    | 46.2 | 28.3 | 49.2 | 61.5 |     41     |    173     |          325           |           6.5           |      15.0      | 19.4                | [配置文件](configs/r50_deformable_detr_plus_iterative_bbox_refinement.sh) [日志](https://drive.google.com/file/d/1DFNloITi1SFBWjYzvVEAI75ndwmGM1Uj/view?usp=sharing) [模型](https://drive.google.com/file/d/1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI/view?usp=sharing) |
| **++ 两阶段 Deformable DETR**     |    50    | 46.9 | 29.6 | 50.1 | 61.6 |     41     |    173     |          340           |           6.8           |      14.5      | 18.8                | [配置文件](configs/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh) [日志](https://drive.google.com/file/d/1ozi0wbv5-Sc5TbWt1jAuXco72vEfEtbY/view?usp=sharing) [模型](https://drive.google.com/file/d/15I03A7hNTpwuLNdfuEmW9_taZMNVssEp/view?usp=sharing) |

*注：*

1. Deformable DETR 的所有模型均在总批次大小为 32 的条件下训练。

2. 训练和推理速度均在 NVIDIA Tesla V100 GPU 上测得。

3. “Deformable DETR (单尺度)”指仅使用 res5 特征图（步幅为 32）作为 Deformable Transformer 编码器的输入特征图。

4. “DC5”指移除 ResNet 中 C5 阶段的步幅，并添加空洞率为 2 的空洞卷积。

5. “DETR-DC5+”表示经过一些修改的 DETR-DC5，包括使用 Focal Loss 进行边界框分类，并将目标查询数量增加到 300。

6. “批量推理速度”指批次大小为 4 时的推理速度，以最大化 GPU 利用率。

7. 原始实现基于我们的内部代码库。由于平台切换中的诸多细节差异，最终精度和运行时间可能存在细微差别。

## 安装

### 环境要求

- Linux，CUDA>=9.2，GCC>=5.4

- Python>=3.7

  建议使用 Anaconda 创建 conda 环境：

  ```
  conda create -n deformable_detr python=3.7 pip
  ```

  然后激活环境：

  ```
  conda activate deformable_detr
  ```

- PyTorch>=1.5.1，torchvision>=0.6.1（安装指南见[此处](https://pytorch.org/)）

  例如，若您的 CUDA 版本为 9.2，可按如下方式安装 pytorch 和 torchvision：

  ```
  conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
  ```

- 其他依赖

  ```
  pip install -r requirements.txt
  ```

### 编译 CUDA 算子

```
cd ./models/ops
sh ./make.sh
# 单元测试（应看到所有检查结果为 True）
python test.py
```
测试结果如下：

```sh
* True check_forward_equal_with_pytorch_double: max_abs_err 8.67e-19 max_rel_err 1.98e-16
* True check_forward_equal_with_pytorch_float: max_abs_err 4.66e-10 max_rel_err 1.13e-07
* True check_gradient_numerical(D=30)
* True check_gradient_numerical(D=32)
* True check_gradient_numerical(D=64)
* True check_gradient_numerical(D=71)
...
```
## 使用方法

### 数据集准备

请下载 [COCO 2017 数据集](https://cocodataset.org/)，并按如下结构组织：

```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

### 训练

#### 单节点训练

例如，在 2 张 GPU 上训练 Deformable DETR 的命令如下：

```
GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/r50_deformable_detr.sh
```

#### 多节点训练

例如，在 2 个节点（每个节点 8 张 GPU）上训练 Deformable DETR 的命令如下：

在节点 1 上：

```
MASTER_ADDR=<节点 1 的 IP 地址> NODE_RANK=0 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./configs/r50_deformable_detr.sh
```

在节点 2 上：

```
MASTER_ADDR=<节点 1 的 IP 地址> NODE_RANK=1 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./configs/r50_deformable_detr.sh
```

#### 在 Slurm 集群上训练

若您使用 Slurm 集群，可运行以下命令在 1 个节点（8 张 GPU）上训练：

```
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <分区名> deformable_detr 8 configs/r50_deformable_detr.sh
```

或在 2 个节点（每个节点 8 张 GPU）上训练：

```
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <分区名> deformable_detr 16 configs/r50_deformable_detr.sh
```

#### 加速训练的小技巧

- 如果您的文件系统读取图像较慢，可考虑启用 `--cache_mode`选项，在训练开始时将整个数据集加载到内存中。

- 根据您的 GPU 显存，可增加批次大小以最大化 GPU 利用率，例如设置 `--batch_size 3`或 `--batch_size 4`。

### 评估

您可以获取 Deformable DETR 的配置文件和预训练模型（链接位于“主要结果”章节），然后运行以下命令在 COCO 2017 验证集上进行评估：

```
<配置文件路径> --resume <预训练模型路径> --eval
```

您也可以使用 `./tools/run_dist_launch.sh`或 `./tools/run_dist_slurm.sh`运行分布式评估。