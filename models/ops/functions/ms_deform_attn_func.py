# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)# 从 HW 图像变为 查询的采样点 Lq * p
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    # list of (N_*M_, D_, Lq_, P_) -> (N_*M_, D_, Lq_, L_, P_) -> 
    # (N_*M_, D_, Lq_, L_*P_) * (N_*M_, 1, Lq_, L_*P_) -> (N_*M_, D_, Lq_, L_*P_) ->
    # (N_*M_, D_, Lq_) -> (N_, M_*D_, Lq_) -> (N_, Lq_, M_*D_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()

'''
算法名称
Multi-Scale Deformable Attention(多尺度可变形注意力)
核心创新点
- 稀疏采样：每个查询只关注少量关键位置，而非全局注意力
- 多尺度融合：从不同分辨率的特征图中采样信息
- 动态偏移：采样位置由网络预测，可适应不同形状和尺度目标

论文: Deformable DETR: Deformable Transformers for End-to-End Object Detection

N, M, D = 1, 2, 2      # batch_size=1, 注意力头数=2, 每个头的特征维度=2
Lq, L, P = 2, 2, 2     # 查询点数=2, 特征层数=2, 每层采样点数=2

shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
value: 1,30,2,2
sampling_locations: 1,2,2,2,2,2
attention_weights: 1,2,2,2,2 -> 2,1,2,4
output: 1,2,4

输入: 多尺度特征图 [Level1(H1xW1), Level2(H2xW2)]
        ↓
为每个查询生成:
  ├─ 采样偏移 (Lq x M x L x P x 2)
  └─ 注意力权重 (Lq x M x L x P)
        ↓
双线性插值采样 → 获得采样值 (NxMxDxLqxLxP)
        ↓
注意力权重加权 → 加权求和
        ↓
输出: (N, Lq, MxD)
'''