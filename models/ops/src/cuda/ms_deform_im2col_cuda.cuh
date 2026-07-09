/*!
**************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************
* Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
* Copyright (c) 2018 Microsoft
**************************************************************************
*/

#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>
#define OPT 3 // 优化方法控制
#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads)
{
  return (N + num_threads - 1) / num_threads;
}

/*
采样: https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.grid_sample.html
双线性加权公式: 
output[n,c,h,w] =
  I(y0,x0) * (1-dx)*(1-dy)
+ I(y0,x1) *   dx  *(1-dy)
+ I(y1,x0) * (1-dx)*  dy
+ I(y1,x1) *   dx  *  dy
单个线程读取 4 个采样点, 双线性插值的工业级 cuda kernel 实现如何？
*/
template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(const scalar_t* &bottom_data, 
                                                   const int &height, const int &width, const int &nheads, const int &channels,
                                                   const scalar_t &h, const scalar_t &w, const int &m, const int &c)
{
  const int h_low = floor(h);// 向下取整
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
  {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t ldg_read(const scalar_t* ptr)
{
#if __CUDA_ARCH__ >= 350
  return __ldg(ptr);
#else
  return *ptr;
#endif
}

__device__ __forceinline__ float4 ldg_read_float4(const float* ptr)
{
#if __CUDA_ARCH__ >= 350
  return __ldg(reinterpret_cast<const float4*>(ptr));
#else
  return *reinterpret_cast<const float4*>(ptr);
#endif
}

__device__ __forceinline__ float4 load_float4(const float* ptr)
{
  return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ float4 ms_deform_attn_im2col_bilinear_float4(
    const float* bottom_data,
    const int height,
    const int width,
    const int nheads,
    const int channels,
    const float h,
    const float w,
    const int m,
    const int c_base)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h - h_low;
  const float lw = w - w_low;
  const float hh = 1.f - lh;
  const float hw = 1.f - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c_base;

  float4 v1 = make_float4(0.f, 0.f, 0.f, 0.f);
  if (h_low >= 0 && w_low >= 0)
  {
    v1 = load_float4(bottom_data + h_low_ptr_offset + w_low_ptr_offset + base_ptr);
  }
  float4 v2 = make_float4(0.f, 0.f, 0.f, 0.f);
  if (h_low >= 0 && w_high <= width - 1)
  {
    v2 = load_float4(bottom_data + h_low_ptr_offset + w_high_ptr_offset + base_ptr);
  }
  float4 v3 = make_float4(0.f, 0.f, 0.f, 0.f);
  if (h_high <= height - 1 && w_low >= 0)
  {
    v3 = load_float4(bottom_data + h_high_ptr_offset + w_low_ptr_offset + base_ptr);
  }
  float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    v4 = load_float4(bottom_data + h_high_ptr_offset + w_high_ptr_offset + base_ptr);
  }

  const float w1 = hh * hw;
  const float w2 = hh * lw;
  const float w3 = lh * hw;
  const float w4 = lh * lw;

  float4 val;
  val.x = w1 * v1.x + w2 * v2.x + w3 * v3.x + w4 * v4.x;
  val.y = w1 * v1.y + w2 * v2.y + w3 * v3.y + w4 * v4.y;
  val.z = w1 * v1.z + w2 * v2.z + w3 * v3.z + w4 * v4.z;
  val.w = w1 * v1.w + w2 * v2.w + w3 * v3.w + w4 * v4.w;
  return val;
}


template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear(const scalar_t* &bottom_data, 
                                                   const int &height, const int &width, const int &nheads, const int &channels,
                                                   const scalar_t &h, const scalar_t &w, const int &m, const int &c,
                                                   const scalar_t &top_grad,
                                                   const scalar_t &attn_weight,
                                                   scalar_t* &grad_value, 
                                                   scalar_t* grad_sampling_loc,
                                                   scalar_t* grad_attn_weight)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value+ptr1, w1*top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
  {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value+ptr2, w2*top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value+ptr3, w3*top_grad_value); 
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value+ptr4, w4*top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
}


template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear_gm(const scalar_t* &bottom_data, 
                                                   const int &height, const int &width, const int &nheads, const int &channels,
                                                   const scalar_t &h, const scalar_t &w, const int &m, const int &c,
                                                   const scalar_t &top_grad,
                                                   const scalar_t &attn_weight,
                                                   scalar_t* &grad_value, 
                                                   scalar_t* grad_sampling_loc,
                                                   scalar_t* grad_attn_weight)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value+ptr1, w1*top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
  {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value+ptr2, w2*top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value+ptr3, w3*top_grad_value); 
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value+ptr4, w4*top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_attn_weight, top_grad * val); 
  atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
}

#if OPT == 0
template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(const int n,
                                                const scalar_t *data_value, 
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *data_col)
{
  CUDA_KERNEL_LOOP(index, n) // idx = 0 ～ 7;
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;
    
    for (int l_col=0; l_col < num_levels; ++l_col)// 0; 1
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col=0; p_col < num_point; ++p_col)// 0; 1
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          col += ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col) * weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}
#elif OPT == 1
template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n,
    const scalar_t* __restrict__ data_value,
    const int64_t* __restrict__ data_spatial_shapes,
    const int64_t* __restrict__ data_level_start_index,
    const scalar_t* __restrict__ data_sampling_loc,
    const scalar_t* __restrict__ data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* __restrict__ data_col)
{
    extern __shared__ unsigned char smem_raw[];
    const int sampling_index = blockIdx.x;
    const int total_sampling = n / channels;
    if (sampling_index >= total_sampling) return;

    const int num_level_points = num_levels * num_point;
    int* smem_ptr = reinterpret_cast<int*>(smem_raw);
    scalar_t* smem_weight = reinterpret_cast<scalar_t*>(smem_ptr + num_level_points * 4);

    const int m_col = sampling_index % num_heads;
    const int b_col = (sampling_index / num_heads) / num_query;
    const int qid_stride = num_heads * channels;
    const int w_stride = qid_stride;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    const scalar_t* data_value_ptr = data_value + data_value_ptr_init_offset;
    const int sampling_offset = sampling_index * num_level_points;

    for (int sample_offset = threadIdx.x; sample_offset < num_level_points; sample_offset += blockDim.x)
    {
      const int l_col = sample_offset / num_point;
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = static_cast<int>(data_spatial_shapes[spatial_h_ptr]);
      const int spatial_w = static_cast<int>(data_spatial_shapes[spatial_h_ptr + 1]);
      const int level_start_id = static_cast<int>(data_level_start_index[l_col]);
      const int data_weight_ptr = sampling_offset + sample_offset;
      const int data_loc_w_ptr = data_weight_ptr << 1;
      const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
      const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
      const scalar_t attn_weight = data_attn_weight[data_weight_ptr];

      int* ptr = smem_ptr + sample_offset * 4;
      scalar_t* weight = smem_weight + sample_offset * 4;
      ptr[0] = -1; ptr[1] = -1; ptr[2] = -1; ptr[3] = -1;
      weight[0] = 0; weight[1] = 0; weight[2] = 0; weight[3] = 0;

      const scalar_t h_im = loc_h * spatial_h - static_cast<scalar_t>(0.5);
      const scalar_t w_im = loc_w * spatial_w - static_cast<scalar_t>(0.5);

      if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
      {
        const int h_low = floor(h_im);
        const int w_low = floor(w_im);
        const int h_high = h_low + 1;
        const int w_high = w_low + 1;

        const scalar_t lh = h_im - h_low;
        const scalar_t lw = w_im - w_low;
        const scalar_t hh = 1 - lh;
        const scalar_t hw = 1 - lw;

        const int h_stride = spatial_w * w_stride;
        const int base_ptr = level_start_id * qid_stride + m_col * channels;
        const int h_low_ptr_offset = h_low * h_stride;
        const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
        const int w_low_ptr_offset = w_low * w_stride;
        const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

        if (h_low >= 0 && w_low >= 0)
        {
          ptr[0] = base_ptr + h_low_ptr_offset + w_low_ptr_offset;
          weight[0] = attn_weight * hh * hw;
        }
        if (h_low >= 0 && w_high <= spatial_w - 1)
        {
          ptr[1] = base_ptr + h_low_ptr_offset + w_high_ptr_offset;
          weight[1] = attn_weight * hh * lw;
        }
        if (h_high <= spatial_h - 1 && w_low >= 0)
        {
          ptr[2] = base_ptr + h_high_ptr_offset + w_low_ptr_offset;
          weight[2] = attn_weight * lh * hw;
        }
        if (h_high <= spatial_h - 1 && w_high <= spatial_w - 1)
        {
          ptr[3] = base_ptr + h_high_ptr_offset + w_high_ptr_offset;
          weight[3] = attn_weight * lh * lw;
        }
      }
    }
    __syncthreads();

    scalar_t* data_col_ptr = data_col + sampling_index * channels;
    for (int c_col = threadIdx.x; c_col < channels; c_col += blockDim.x)
    {
      scalar_t col = 0;
      #pragma unroll
      for (int sample_offset = 0; sample_offset < num_level_points; ++sample_offset)
      {
        const int* ptr = smem_ptr + sample_offset * 4;
        const scalar_t* weight = smem_weight + sample_offset * 4;
        if (ptr[0] >= 0) col += data_value_ptr[ptr[0] + c_col] * weight[0];
        if (ptr[1] >= 0) col += data_value_ptr[ptr[1] + c_col] * weight[1];
        if (ptr[2] >= 0) col += data_value_ptr[ptr[2] + c_col] * weight[2];
        if (ptr[3] >= 0) col += data_value_ptr[ptr[3] + c_col] * weight[3];
      }
      data_col_ptr[c_col] = col;
    }
}
#elif OPT == 2 // float4
template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n,
    const scalar_t* __restrict__ data_value,
    const int64_t* __restrict__ data_spatial_shapes,
    const int64_t* __restrict__ data_level_start_index,
    const scalar_t* __restrict__ data_sampling_loc,
    const scalar_t* __restrict__ data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* __restrict__ data_col)
{
    extern __shared__ unsigned char smem_raw[];
    const int sampling_index = blockIdx.x;
    const int total_sampling = n / channels;
    if (sampling_index >= total_sampling) return;

    const int num_level_points = num_levels * num_point;
    int* smem_ptr = reinterpret_cast<int*>(smem_raw);
    scalar_t* smem_weight = reinterpret_cast<scalar_t*>(smem_ptr + num_level_points * 4);

    const int m_col = sampling_index % num_heads;
    const int b_col = (sampling_index / num_heads) / num_query;
    const int qid_stride = num_heads * channels;
    const int w_stride = qid_stride;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    const scalar_t* data_value_ptr = data_value + data_value_ptr_init_offset;
    const int sampling_offset = sampling_index * num_level_points;

    for (int sample_offset = threadIdx.x; sample_offset < num_level_points; sample_offset += blockDim.x)
    {
      const int l_col = sample_offset / num_point;
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = static_cast<int>(ldg_read(data_spatial_shapes + spatial_h_ptr));
      const int spatial_w = static_cast<int>(ldg_read(data_spatial_shapes + spatial_h_ptr + 1));
      const int level_start_id = static_cast<int>(ldg_read(data_level_start_index + l_col));
      const int data_weight_ptr = sampling_offset + sample_offset;
      const int data_loc_w_ptr = data_weight_ptr << 1;
      const scalar_t loc_w = ldg_read(data_sampling_loc + data_loc_w_ptr);
      const scalar_t loc_h = ldg_read(data_sampling_loc + data_loc_w_ptr + 1);
      const scalar_t attn_weight = ldg_read(data_attn_weight + data_weight_ptr);

      int* ptr = smem_ptr + sample_offset * 4;
      scalar_t* weight = smem_weight + sample_offset * 4;
      ptr[0] = -1; ptr[1] = -1; ptr[2] = -1; ptr[3] = -1;
      weight[0] = 0; weight[1] = 0; weight[2] = 0; weight[3] = 0;

      const scalar_t h_im = loc_h * spatial_h - static_cast<scalar_t>(0.5);
      const scalar_t w_im = loc_w * spatial_w - static_cast<scalar_t>(0.5);

      if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
      {
        const int h_low = floor(h_im);
        const int w_low = floor(w_im);
        const int h_high = h_low + 1;
        const int w_high = w_low + 1;

        const scalar_t lh = h_im - h_low;
        const scalar_t lw = w_im - w_low;
        const scalar_t hh = 1 - lh;
        const scalar_t hw = 1 - lw;

        const int h_stride = spatial_w * w_stride;
        const int base_ptr = level_start_id * qid_stride + m_col * channels;
        const int h_low_ptr_offset = h_low * h_stride;
        const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
        const int w_low_ptr_offset = w_low * w_stride;
        const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

        if (h_low >= 0 && w_low >= 0)
        {
          ptr[0] = base_ptr + h_low_ptr_offset + w_low_ptr_offset;
          weight[0] = attn_weight * hh * hw;
        }
        if (h_low >= 0 && w_high <= spatial_w - 1)
        {
          ptr[1] = base_ptr + h_low_ptr_offset + w_high_ptr_offset;
          weight[1] = attn_weight * hh * lw;
        }
        if (h_high <= spatial_h - 1 && w_low >= 0)
        {
          ptr[2] = base_ptr + h_high_ptr_offset + w_low_ptr_offset;
          weight[2] = attn_weight * lh * hw;
        }
        if (h_high <= spatial_h - 1 && w_high <= spatial_w - 1)
        {
          ptr[3] = base_ptr + h_high_ptr_offset + w_high_ptr_offset;
          weight[3] = attn_weight * lh * lw;
        }
      }
    }
    __syncthreads();

    scalar_t* data_col_ptr = data_col + sampling_index * channels;
    if constexpr (std::is_same<scalar_t, float>::value)
    {
      const bool use_float4 =
          ((channels & 3) == 0) &&
          ((reinterpret_cast<uintptr_t>(data_value_ptr) & 0xF) == 0) &&
          ((reinterpret_cast<uintptr_t>(data_col_ptr) & 0xF) == 0);
      if (use_float4)
      {
        const int vec_channels = channels >> 2;
        float* out_ptr = reinterpret_cast<float*>(data_col_ptr);
        const float* in_ptr = reinterpret_cast<const float*>(data_value_ptr);
        for (int c_vec = threadIdx.x; c_vec < vec_channels; c_vec += blockDim.x)
        {
          const int c_base = c_vec << 2;
          float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
          #pragma unroll
          for (int sample_offset = 0; sample_offset < num_level_points; ++sample_offset)
          {
            const int* ptr = smem_ptr + sample_offset * 4;
            const scalar_t* weight = smem_weight + sample_offset * 4;
            if (ptr[0] >= 0)
            {
              const float4 value = ldg_read_float4(in_ptr + ptr[0] + c_base);
              const float w = static_cast<float>(weight[0]);
              acc.x += value.x * w; acc.y += value.y * w; acc.z += value.z * w; acc.w += value.w * w;
            }
            if (ptr[1] >= 0)
            {
              const float4 value = ldg_read_float4(in_ptr + ptr[1] + c_base);
              const float w = static_cast<float>(weight[1]);
              acc.x += value.x * w; acc.y += value.y * w; acc.z += value.z * w; acc.w += value.w * w;
            }
            if (ptr[2] >= 0)
            {
              const float4 value = ldg_read_float4(in_ptr + ptr[2] + c_base);
              const float w = static_cast<float>(weight[2]);
              acc.x += value.x * w; acc.y += value.y * w; acc.z += value.z * w; acc.w += value.w * w;
            }
            if (ptr[3] >= 0)
            {
              const float4 value = ldg_read_float4(in_ptr + ptr[3] + c_base);
              const float w = static_cast<float>(weight[3]);
              acc.x += value.x * w; acc.y += value.y * w; acc.z += value.z * w; acc.w += value.w * w;
            }
          }
          reinterpret_cast<float4*>(out_ptr)[c_vec] = acc;
        }
        return;
      }
    }

    for (int c_col = threadIdx.x; c_col < channels; c_col += blockDim.x)
    {
      scalar_t col = 0;
      #pragma unroll
      for (int sample_offset = 0; sample_offset < num_level_points; ++sample_offset)
      {
        const int* ptr = smem_ptr + sample_offset * 4;
        const scalar_t* weight = smem_weight + sample_offset * 4;
        if (ptr[0] >= 0) col += ldg_read(data_value_ptr + ptr[0] + c_col) * weight[0];
        if (ptr[1] >= 0) col += ldg_read(data_value_ptr + ptr[1] + c_col) * weight[1];
        if (ptr[2] >= 0) col += ldg_read(data_value_ptr + ptr[2] + c_col) * weight[2];
        if (ptr[3] >= 0) col += ldg_read(data_value_ptr + ptr[3] + c_col) * weight[3];
      }
      data_col_ptr[c_col] = col;
    }
}
#elif OPT == 3 // naive+float4
template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(const int n,
                                                const scalar_t *data_value, 
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *data_col)
{
  if constexpr (std::is_same<scalar_t, float>::value)
  {
    if ((channels & 3) == 0)
    {
      const int vec_channels = channels >> 2;
      CUDA_KERNEL_LOOP(index, n)
      {
        int _temp = index;
        const int c_vec = _temp % vec_channels;
        _temp /= vec_channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        const int q_col = _temp % num_query;
        _temp /= num_query;
        const int b_col = _temp;

        float4 *data_col_ptr = reinterpret_cast<float4*>(data_col) + index;
        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int qid_stride = num_heads * channels;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        const float *data_value_ptr_init =
            reinterpret_cast<const float*>(data_value) + data_value_ptr_init_offset;
        const int c_base = c_vec << 2;
        float4 col = make_float4(0.f, 0.f, 0.f, 0.f);

        for (int l_col = 0; l_col < num_levels; ++l_col)
        {
          const int level_start_id = data_level_start_index[l_col];
          const int spatial_h_ptr = l_col << 1;
          const int spatial_h = data_spatial_shapes[spatial_h_ptr];
          const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
          const float *data_value_ptr =
              data_value_ptr_init + level_start_id * qid_stride;
          for (int p_col = 0; p_col < num_point; ++p_col)
          {
            const float loc_w = data_sampling_loc[data_loc_w_ptr];
            const float loc_h = data_sampling_loc[data_loc_w_ptr + 1];
            const float weight = data_attn_weight[data_weight_ptr];

            const float h_im = loc_h * spatial_h - 0.5f;
            const float w_im = loc_w * spatial_w - 0.5f;

            if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
            {
              const float4 val = ms_deform_attn_im2col_bilinear_float4(
                  data_value_ptr,
                  spatial_h,
                  spatial_w,
                  num_heads,
                  channels,
                  h_im,
                  w_im,
                  m_col,
                  c_base);
              col.x += val.x * weight;
              col.y += val.y * weight;
              col.z += val.z * weight;
              col.w += val.w * weight;
            }

            data_weight_ptr += 1;
            data_loc_w_ptr += 2;
          }
        }
        *data_col_ptr = col;
      }
      return;
    }
  }

  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col = 0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t *data_value_ptr =
          data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          col += ms_deform_attn_im2col_bilinear(
              data_value_ptr,
              spatial_h,
              spatial_w,
              num_heads,
              channels,
              h_im,
              w_im,
              m_col,
              c_col) * weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}
#elif OPT == 4
#endif
template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_w=cache_grad_sampling_loc[0], _grad_h=cache_grad_sampling_loc[1], _grad_a=cache_grad_attn_weight[0];
          int sid=2;
          for (unsigned int tid = 1; tid < blockSize; ++tid)
          {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 2;
          }
          
          
          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();

        for (unsigned int s=blockSize/2; s>0; s>>=1)
        {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
          }
          __syncthreads();
        }

        if (tid == 0)
        { 
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v1(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_w=cache_grad_sampling_loc[0], _grad_h=cache_grad_sampling_loc[1], _grad_a=cache_grad_attn_weight[0];
          int sid=2;
          for (unsigned int tid = 1; tid < blockDim.x; ++tid)
          {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 2;
          }
          
          
          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();

        for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s>>=1, spre>>=1)
        {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre)
            {
              cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            } 
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc+(threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc+((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            cache_grad_sampling_loc+(threadIdx.x << 1), cache_grad_attn_weight+threadIdx.x);
        }
        
        __syncthreads();

        for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s>>=1, spre>>=1)
        {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre)
            {
              cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] += cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          atomicAdd(grad_sampling_loc, cache_grad_sampling_loc[0]);
          atomicAdd(grad_sampling_loc + 1, cache_grad_sampling_loc[1]);
          atomicAdd(grad_attn_weight, cache_grad_attn_weight[0]);
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_gm(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          ms_deform_attn_col2im_bilinear_gm(
            data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col,
            top_grad, weight, grad_value_ptr, 
            grad_sampling_loc, grad_attn_weight);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
void ms_deformable_im2col_cuda(cudaStream_t stream,
                              const scalar_t* data_value,
                              const int64_t* data_spatial_shapes, 
                              const int64_t* data_level_start_index, 
                              const scalar_t* data_sampling_loc,
                              const scalar_t* data_attn_weight,
                              const int batch_size,
                              const int spatial_size, 
                              const int num_heads, 
                              const int channels, 
                              const int num_levels, 
                              const int num_query,
                              const int num_point,
                              scalar_t* data_col)
{
  int num_kernels = batch_size * num_query * num_heads * channels;       // 1,2,2,2=8
  const int rounded_channels = ((channels + 31) / 32) * 32;
  const int num_threads = std::min(CUDA_NUM_THREADS, std::max(32, rounded_channels));
  const size_t shared_memory_bytes =
      static_cast<size_t>(num_levels) * num_point * (4 * sizeof(int) + 4 * sizeof(scalar_t));
#if OPT == 3
  if constexpr (std::is_same<scalar_t, float>::value)
  {
    if ((channels & 3) == 0)
    {
      num_kernels /= 4;
    }
  }
#endif
  int num_actual_kernels = batch_size * num_query * num_heads;
#if OPT == 0 || OPT == 3
  num_actual_kernels = GET_BLOCKS(num_kernels, num_threads);
#endif
  ms_deformable_im2col_gpu_kernel<scalar_t>
      <<<num_actual_kernels, num_threads,
          shared_memory_bytes, stream>>>(
      num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight, 
      batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void ms_deformable_col2im_cuda(cudaStream_t stream,
                              const scalar_t* grad_col,
                              const scalar_t* data_value,
                              const int64_t * data_spatial_shapes,
                              const int64_t * data_level_start_index,
                              const scalar_t * data_sampling_loc,
                              const scalar_t * data_attn_weight,
                              const int batch_size, 
                              const int spatial_size, 
                              const int num_heads,
                              const int channels, 
                              const int num_levels,
                              const int num_query,
                              const int num_point, 
                              scalar_t* grad_value,
                              scalar_t* grad_sampling_loc,
                              scalar_t* grad_attn_weight)
{
  const int num_threads = (channels > CUDA_NUM_THREADS)?CUDA_NUM_THREADS:channels;
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  if (channels > 1024)
  {
    if ((channels & 1023) == 0)
    {
      ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*3*sizeof(scalar_t), stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
    }
    else
    {
      ms_deformable_col2im_gpu_kernel_gm<scalar_t>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
    }
  }
  else{
    switch(channels)
    {
      case 1:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 2:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 4:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 8:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 16:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 32:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 32>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 64:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 128:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 256:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 512:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 1024:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels, 
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index, 
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size, 
                      spatial_size, 
                      num_heads,
                      channels, 
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      default:
        if (channels < 64)
        {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*3*sizeof(scalar_t), stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        }
        else
        {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*3*sizeof(scalar_t), stream>>>(
                        num_kernels, 
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index, 
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size, 
                        spatial_size, 
                        num_heads,
                        channels, 
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        }
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}
