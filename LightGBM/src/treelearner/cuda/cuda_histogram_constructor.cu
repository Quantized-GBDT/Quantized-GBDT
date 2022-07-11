/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

#include <LightGBM/cuda/cuda_algorithms.hpp>

#include <algorithm>

namespace LightGBM {

template <typename BIN_TYPE, typename HIST_TYPE, int SHARED_HIST_SIZE>
__global__ void CUDAConstructHistogramDenseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const BIN_TYPE* data_ptr = data + partition_column_start * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) {
    HIST_TYPE* shared_hist_ptr = shared_hist + (column_hist_offsets[column_index] << 1);
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[data_index * num_columns_in_partition + threadIdx.x]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist_ptr + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

template <typename BIN_TYPE, int SHARED_HIST_SIZE, bool USE_16BIT_HIST>
__global__ void CUDAConstructDiscretizedHistogramDenseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ int16_t shared_hist[SHARED_HIST_SIZE];
  int32_t* shared_hist_packed = reinterpret_cast<int32_t*>(shared_hist);
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const BIN_TYPE* data_ptr = data + partition_column_start * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start);
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist_packed[i] = 0;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) {
    int32_t* shared_hist_ptr = shared_hist_packed + (column_hist_offsets[column_index]);
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[data_index * num_columns_in_partition + threadIdx.x]);
      int32_t* pos_ptr = shared_hist_ptr + bin;
      atomicAdd_block(pos_ptr, grad_and_hess);
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();
  if (USE_16BIT_HIST) {
    int32_t* feature_histogram_ptr = reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      atomicAdd_system(feature_histogram_ptr + i, packed_grad_hess);
    }
  } else {
    unsigned long long* feature_histogram_ptr = reinterpret_cast<unsigned long long*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd_system(feature_histogram_ptr + i, (unsigned long long)(packed_grad_hess_int64));
    }
  }
}

template <typename BIN_TYPE, typename DATA_PTR_TYPE, typename HIST_TYPE, int SHARED_HIST_SIZE>
__global__ void CUDAConstructHistogramSparseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + blockIdx.x * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

template <typename BIN_TYPE, typename DATA_PTR_TYPE, int SHARED_HIST_SIZE, bool USE_16BIT_HIST>
__global__ void CUDAConstructDiscretizedHistogramSparseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ int16_t shared_hist[SHARED_HIST_SIZE];
  int32_t* shared_hist_packed = reinterpret_cast<int32_t*>(shared_hist);
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + blockIdx.x * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start);
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist_packed[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      int32_t* pos_ptr = shared_hist_packed + bin;
      atomicAdd_block(pos_ptr, grad_and_hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  if (USE_16BIT_HIST) {
    int32_t* feature_histogram_ptr = reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      atomicAdd_system(feature_histogram_ptr + i, packed_grad_hess);
    }
  } else {
    unsigned long long* feature_histogram_ptr = reinterpret_cast<unsigned long long*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd_system(feature_histogram_ptr + i, (unsigned long long)(packed_grad_hess_int64));
    }
  }
}

// TODO: global memory buffer should also has double precision option
template <typename BIN_TYPE>
__global__ void CUDAConstructHistogramDenseKernel_GlobalMemory(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const data_size_t num_data,
  float* global_hist_buffer) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const BIN_TYPE* data_ptr = data + partition_column_start * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  const int num_total_bin = column_hist_offsets_full[gridDim.x];
  float* shared_hist = global_hist_buffer + (blockIdx.y * num_total_bin + partition_hist_start) * 2;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) {
    float* shared_hist_ptr = shared_hist + (column_hist_offsets[column_index] << 1);
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[data_index * num_columns_in_partition + threadIdx.x]);
      const uint32_t pos = bin << 1;
      float* pos_ptr = shared_hist_ptr + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

// TODO: global memory buffer should also has double precision option
template <typename BIN_TYPE, typename DATA_PTR_TYPE>
__global__ void CUDAConstructHistogramSparseKernel_GlobalMemory(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data,
  float* global_hist_buffer) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + blockIdx.x * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  const int num_total_bin = column_hist_offsets_full[gridDim.x];
  float* shared_hist = global_hist_buffer + (blockIdx.y * num_total_bin + partition_hist_start) * 2;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      const uint32_t pos = bin << 1;
      float* pos_ptr = shared_hist + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

void CUDAHistogramConstructor::LaunchConstructHistogramKernel(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const uint8_t num_bits_in_histogram_bins) {
  if (use_discretized_grad_) {
    CHECK(cuda_row_data_->shared_hist_size() == 6144 * 2 || cuda_row_data_->shared_hist_size() == 6144 * 4 || cuda_row_data_->shared_hist_size() == 3072);
    if (cuda_row_data_->shared_hist_size() == 6144 * 2) {
      LaunchConstructDiscretizedHistogramKernel<6144 * 2>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
    } else if (cuda_row_data_->shared_hist_size() == 6144 * 4) {
      LaunchConstructDiscretizedHistogramKernel<6144 * 4>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
    } else {
      LaunchConstructDiscretizedHistogramKernel<3072>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
    }
  } else if (cuda_row_data_->use_dp()) {
    CHECK_EQ(cuda_row_data_->shared_hist_size(), 6144);
    LaunchConstructHistogramKernelInner<double, 6144>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
  } else {
    CHECK(cuda_row_data_->shared_hist_size() == 6144 * 2 || cuda_row_data_->shared_hist_size() == 3072);
    if (cuda_row_data_->shared_hist_size() == 6144 * 2) {
      LaunchConstructHistogramKernelInner<float, 6144 * 2>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
    } else {
      LaunchConstructHistogramKernelInner<float, 3072>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
    }
  }
}

template <int SHARED_HIST_SIZE>
void CUDAHistogramConstructor::LaunchConstructDiscretizedHistogramKernel(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const uint8_t num_bits_in_histogram_bins) {
  int grid_dim_x = 0;
  int grid_dim_y = 0;
  int block_dim_x = 0;
  int block_dim_y = 0;
  CalcConstructHistogramKernelDim(&grid_dim_x, &grid_dim_y, &block_dim_x, &block_dim_y, num_data_in_smaller_leaf);
  dim3 grid_dim(grid_dim_x, grid_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  if (num_bits_in_histogram_bins <= 16) {
    if (!cuda_row_data_->is_sparse()) {
      if (cuda_row_data_->bit_type() == 8) {
        CUDAConstructDiscretizedHistogramDenseKernel<uint8_t, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          reinterpret_cast<const int32_t*>(cuda_gradients_),
          cuda_row_data_->cuda_data_uint8(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_);
      } else if (cuda_row_data_->bit_type() == 16) {
        CUDAConstructDiscretizedHistogramDenseKernel<uint16_t, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          reinterpret_cast<const int32_t*>(cuda_gradients_),
          cuda_row_data_->cuda_data_uint16(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_);
      } else if (cuda_row_data_->bit_type() == 32) {
        CUDAConstructDiscretizedHistogramDenseKernel<uint32_t, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          reinterpret_cast<const int32_t*>(cuda_gradients_),
          cuda_row_data_->cuda_data_uint32(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_);
      }
    } else {
      if (cuda_row_data_->bit_type() == 8) {
        if (cuda_row_data_->row_ptr_bit_type() == 16) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint8_t, uint16_t, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint8(),
            cuda_row_data_->cuda_row_ptr_uint16(),
            cuda_row_data_->cuda_partition_ptr_uint16(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint8_t, uint32_t, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint8(),
            cuda_row_data_->cuda_row_ptr_uint32(),
            cuda_row_data_->cuda_partition_ptr_uint32(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint8_t, uint64_t, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint8(),
            cuda_row_data_->cuda_row_ptr_uint64(),
            cuda_row_data_->cuda_partition_ptr_uint64(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
      } else if (cuda_row_data_->bit_type() == 16) {
        if (cuda_row_data_->row_ptr_bit_type() == 16) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint16_t, uint16_t, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint16(),
            cuda_row_data_->cuda_row_ptr_uint16(),
            cuda_row_data_->cuda_partition_ptr_uint16(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint16_t, uint32_t, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint16(),
            cuda_row_data_->cuda_row_ptr_uint32(),
            cuda_row_data_->cuda_partition_ptr_uint32(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint16_t, uint64_t, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint16(),
            cuda_row_data_->cuda_row_ptr_uint64(),
            cuda_row_data_->cuda_partition_ptr_uint64(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
      } else if (cuda_row_data_->bit_type() == 32) {
        if (cuda_row_data_->row_ptr_bit_type() == 16) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint32_t, uint16_t, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint32(),
            cuda_row_data_->cuda_row_ptr_uint16(),
            cuda_row_data_->cuda_partition_ptr_uint16(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint32_t, uint32_t, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint32(),
            cuda_row_data_->cuda_row_ptr_uint32(),
            cuda_row_data_->cuda_partition_ptr_uint32(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint32_t, uint64_t, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint32(),
            cuda_row_data_->cuda_row_ptr_uint64(),
            cuda_row_data_->cuda_partition_ptr_uint64(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
      }
    }
  } else {
    if (!cuda_row_data_->is_sparse()) {
      if (cuda_row_data_->bit_type() == 8) {
        CUDAConstructDiscretizedHistogramDenseKernel<uint8_t, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          reinterpret_cast<const int32_t*>(cuda_gradients_),
          cuda_row_data_->cuda_data_uint8(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_);
      } else if (cuda_row_data_->bit_type() == 16) {
        CUDAConstructDiscretizedHistogramDenseKernel<uint16_t, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          reinterpret_cast<const int32_t*>(cuda_gradients_),
          cuda_row_data_->cuda_data_uint16(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_);
      } else if (cuda_row_data_->bit_type() == 32) {
        CUDAConstructDiscretizedHistogramDenseKernel<uint32_t, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          reinterpret_cast<const int32_t*>(cuda_gradients_),
          cuda_row_data_->cuda_data_uint32(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_);
      }
    } else {
      if (cuda_row_data_->bit_type() == 8) {
        if (cuda_row_data_->row_ptr_bit_type() == 16) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint8_t, uint16_t, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint8(),
            cuda_row_data_->cuda_row_ptr_uint16(),
            cuda_row_data_->cuda_partition_ptr_uint16(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint8_t, uint32_t, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint8(),
            cuda_row_data_->cuda_row_ptr_uint32(),
            cuda_row_data_->cuda_partition_ptr_uint32(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint8_t, uint64_t, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint8(),
            cuda_row_data_->cuda_row_ptr_uint64(),
            cuda_row_data_->cuda_partition_ptr_uint64(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
      } else if (cuda_row_data_->bit_type() == 16) {
        if (cuda_row_data_->row_ptr_bit_type() == 16) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint16_t, uint16_t, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint16(),
            cuda_row_data_->cuda_row_ptr_uint16(),
            cuda_row_data_->cuda_partition_ptr_uint16(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint16_t, uint32_t, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint16(),
            cuda_row_data_->cuda_row_ptr_uint32(),
            cuda_row_data_->cuda_partition_ptr_uint32(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint16_t, uint64_t, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint16(),
            cuda_row_data_->cuda_row_ptr_uint64(),
            cuda_row_data_->cuda_partition_ptr_uint64(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
      } else if (cuda_row_data_->bit_type() == 32) {
        if (cuda_row_data_->row_ptr_bit_type() == 16) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint32_t, uint16_t, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint32(),
            cuda_row_data_->cuda_row_ptr_uint16(),
            cuda_row_data_->cuda_partition_ptr_uint16(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint32_t, uint32_t, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint32(),
            cuda_row_data_->cuda_row_ptr_uint32(),
            cuda_row_data_->cuda_partition_ptr_uint32(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
          CUDAConstructDiscretizedHistogramSparseKernel<uint32_t, uint64_t, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->cuda_data_uint32(),
            cuda_row_data_->cuda_row_ptr_uint64(),
            cuda_row_data_->cuda_partition_ptr_uint64(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
      }
    }
  }
}

template <typename HIST_TYPE, int SHARED_HIST_SIZE>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf) {
  int grid_dim_x = 0;
  int grid_dim_y = 0;
  int block_dim_x = 0;
  int block_dim_y = 0;
  CalcConstructHistogramKernelDim(&grid_dim_x, &grid_dim_y, &block_dim_x, &block_dim_y, num_data_in_smaller_leaf);
  dim3 grid_dim(grid_dim_x, grid_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  if (cuda_row_data_->NumLargeBinPartition() == 0) {
    if (cuda_row_data_->is_sparse()) {
      if (cuda_row_data_->bit_type() == 8) {
        if (cuda_row_data_->row_ptr_bit_type() == 16) {
          CUDAConstructHistogramSparseKernel<uint8_t, uint16_t, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint8(),
            cuda_row_data_->cuda_row_ptr_uint16(),
            cuda_row_data_->cuda_partition_ptr_uint16(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
          CUDAConstructHistogramSparseKernel<uint8_t, uint32_t, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint8(),
            cuda_row_data_->cuda_row_ptr_uint32(),
            cuda_row_data_->cuda_partition_ptr_uint32(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
          CUDAConstructHistogramSparseKernel<uint8_t, uint64_t, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint8(),
            cuda_row_data_->cuda_row_ptr_uint64(),
            cuda_row_data_->cuda_partition_ptr_uint64(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
      } else if (cuda_row_data_->bit_type() == 16) {
        if (cuda_row_data_->row_ptr_bit_type() == 16) {
          CUDAConstructHistogramSparseKernel<uint16_t, uint16_t, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint16(),
            cuda_row_data_->cuda_row_ptr_uint16(),
            cuda_row_data_->cuda_partition_ptr_uint16(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
          CUDAConstructHistogramSparseKernel<uint16_t, uint32_t, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint16(),
            cuda_row_data_->cuda_row_ptr_uint32(),
            cuda_row_data_->cuda_partition_ptr_uint32(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
          CUDAConstructHistogramSparseKernel<uint16_t, uint64_t, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint16(),
            cuda_row_data_->cuda_row_ptr_uint64(),
            cuda_row_data_->cuda_partition_ptr_uint64(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
      } else if (cuda_row_data_->bit_type() == 32) {
        if (cuda_row_data_->row_ptr_bit_type() == 16) {
          CUDAConstructHistogramSparseKernel<uint32_t, uint16_t, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint32(),
            cuda_row_data_->cuda_row_ptr_uint16(),
            cuda_row_data_->cuda_partition_ptr_uint16(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
          CUDAConstructHistogramSparseKernel<uint32_t, uint32_t, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint32(),
            cuda_row_data_->cuda_row_ptr_uint32(),
            cuda_row_data_->cuda_partition_ptr_uint32(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
          CUDAConstructHistogramSparseKernel<uint32_t, uint64_t, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint32(),
            cuda_row_data_->cuda_row_ptr_uint64(),
            cuda_row_data_->cuda_partition_ptr_uint64(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
      }
    } else {
      if (cuda_row_data_->bit_type() == 8) {
        CUDAConstructHistogramDenseKernel<uint8_t, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          cuda_row_data_->cuda_data_uint8(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_);
      } else if (cuda_row_data_->bit_type() == 16) {
        CUDAConstructHistogramDenseKernel<uint16_t, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          cuda_row_data_->cuda_data_uint16(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_);
      } else if (cuda_row_data_->bit_type() == 32) {
        CUDAConstructHistogramDenseKernel<uint32_t, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          cuda_row_data_->cuda_data_uint32(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_);
      }
    }
  } else {
    if (cuda_row_data_->is_sparse()) {
      if (cuda_row_data_->bit_type() == 8) {
        if (cuda_row_data_->row_ptr_bit_type() == 16) {
          CUDAConstructHistogramSparseKernel_GlobalMemory<uint8_t, uint16_t><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint8(),
            cuda_row_data_->cuda_row_ptr_uint16(),
            cuda_row_data_->cuda_partition_ptr_uint16(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            cuda_hist_buffer_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
          CUDAConstructHistogramSparseKernel_GlobalMemory<uint8_t, uint32_t><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint8(),
            cuda_row_data_->cuda_row_ptr_uint32(),
            cuda_row_data_->cuda_partition_ptr_uint32(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            cuda_hist_buffer_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
          CUDAConstructHistogramSparseKernel_GlobalMemory<uint8_t, uint64_t><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint8(),
            cuda_row_data_->cuda_row_ptr_uint64(),
            cuda_row_data_->cuda_partition_ptr_uint64(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            cuda_hist_buffer_);
        }
      } else if (cuda_row_data_->bit_type() == 16) {
        if (cuda_row_data_->row_ptr_bit_type() == 16) {
          CUDAConstructHistogramSparseKernel_GlobalMemory<uint16_t, uint16_t><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint16(),
            cuda_row_data_->cuda_row_ptr_uint16(),
            cuda_row_data_->cuda_partition_ptr_uint16(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            cuda_hist_buffer_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
          CUDAConstructHistogramSparseKernel_GlobalMemory<uint16_t, uint32_t><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint16(),
            cuda_row_data_->cuda_row_ptr_uint32(),
            cuda_row_data_->cuda_partition_ptr_uint32(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            cuda_hist_buffer_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
          CUDAConstructHistogramSparseKernel_GlobalMemory<uint16_t, uint64_t><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint16(),
            cuda_row_data_->cuda_row_ptr_uint64(),
            cuda_row_data_->cuda_partition_ptr_uint64(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            cuda_hist_buffer_);
        }
      } else if (cuda_row_data_->bit_type() == 32) {
        if (cuda_row_data_->row_ptr_bit_type() == 16) {
          CUDAConstructHistogramSparseKernel_GlobalMemory<uint32_t, uint16_t><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint32(),
            cuda_row_data_->cuda_row_ptr_uint16(),
            cuda_row_data_->cuda_partition_ptr_uint16(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            cuda_hist_buffer_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
          CUDAConstructHistogramSparseKernel_GlobalMemory<uint32_t, uint32_t><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint32(),
            cuda_row_data_->cuda_row_ptr_uint32(),
            cuda_row_data_->cuda_partition_ptr_uint32(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            cuda_hist_buffer_);
        } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
          CUDAConstructHistogramSparseKernel_GlobalMemory<uint32_t, uint64_t><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->cuda_data_uint32(),
            cuda_row_data_->cuda_row_ptr_uint64(),
            cuda_row_data_->cuda_partition_ptr_uint64(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            cuda_hist_buffer_);
        }
      }
    } else {
      if (cuda_row_data_->bit_type() == 8) {
        CUDAConstructHistogramDenseKernel_GlobalMemory<uint8_t><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          cuda_row_data_->cuda_data_uint8(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_,
          cuda_hist_buffer_);
      } else if (cuda_row_data_->bit_type() == 16) {
        CUDAConstructHistogramDenseKernel_GlobalMemory<uint16_t><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          cuda_row_data_->cuda_data_uint16(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_,
          cuda_hist_buffer_);
      } else if (cuda_row_data_->bit_type() == 32) {
        CUDAConstructHistogramDenseKernel_GlobalMemory<uint32_t><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          cuda_row_data_->cuda_data_uint32(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_,
          cuda_hist_buffer_);
      }
    }
  }
}

__global__ void SubtractHistogramKernel(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits) {
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  const int cuda_larger_leaf_index_ref = cuda_larger_leaf_splits->leaf_index;
  if (cuda_larger_leaf_index_ref >= 0) {
    const hist_t* smaller_leaf_hist = cuda_smaller_leaf_splits->hist_in_leaf;
    hist_t* larger_leaf_hist = cuda_larger_leaf_splits->hist_in_leaf;
    if (global_thread_index < 2 * num_total_bin) {
      larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
    }
  }
}

__global__ void FixHistogramKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits) {
  __shared__ hist_t shared_mem_buffer[32];
  const unsigned int blockIdx_x = blockIdx.x;
  const int feature_index = cuda_need_fix_histogram_features[blockIdx_x];
  const uint32_t num_bin_aligned = cuda_need_fix_histogram_features_num_bin_aligned[blockIdx_x];
  const uint32_t feature_hist_offset = cuda_feature_hist_offsets[feature_index];
  const uint32_t most_freq_bin = cuda_feature_most_freq_bins[feature_index];
  const double leaf_sum_gradients = cuda_smaller_leaf_splits->sum_of_gradients;
  const double leaf_sum_hessians = cuda_smaller_leaf_splits->sum_of_hessians;
  hist_t* feature_hist = cuda_smaller_leaf_splits->hist_in_leaf + feature_hist_offset * 2;
  const unsigned int threadIdx_x = threadIdx.x;
  const uint32_t num_bin = cuda_feature_num_bins[feature_index];
  const uint32_t hist_pos = threadIdx_x << 1;
  const hist_t bin_gradient = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[hist_pos] : 0.0f;
  const hist_t bin_hessian = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[hist_pos + 1] : 0.0f;
  const hist_t sum_gradient = ShuffleReduceSum<hist_t>(bin_gradient, shared_mem_buffer, num_bin_aligned);
  const hist_t sum_hessian = ShuffleReduceSum<hist_t>(bin_hessian, shared_mem_buffer, num_bin_aligned);
  if (threadIdx_x == 0) {
    feature_hist[most_freq_bin << 1] = leaf_sum_gradients - sum_gradient;
    feature_hist[(most_freq_bin << 1) + 1] = leaf_sum_hessians - sum_hessian;
  }
}

template <bool SMALLER_USE_16BIT_HIST, bool LARGER_USE_16BIT_HIST, bool PARENT_USE_16BIT_HIST>
__global__ void SubtractHistogramDiscretizedKernel(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  hist_t* num_bit_change_buffer) {
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  const int cuda_larger_leaf_index_ref = cuda_larger_leaf_splits->leaf_index;
  if (cuda_larger_leaf_index_ref >= 0) {
    if (PARENT_USE_16BIT_HIST) {
      const int32_t* smaller_leaf_hist = reinterpret_cast<const int32_t*>(cuda_smaller_leaf_splits->hist_in_leaf);
      int32_t* larger_leaf_hist = reinterpret_cast<int32_t*>(cuda_larger_leaf_splits->hist_in_leaf);
      if (global_thread_index < num_total_bin) {
        larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
      }
    } else if (LARGER_USE_16BIT_HIST) {
      int32_t* buffer = reinterpret_cast<int32_t*>(num_bit_change_buffer);
      const int32_t* smaller_leaf_hist = reinterpret_cast<const int32_t*>(cuda_smaller_leaf_splits->hist_in_leaf);
      int64_t* larger_leaf_hist = reinterpret_cast<int64_t*>(cuda_larger_leaf_splits->hist_in_leaf);
      if (global_thread_index < num_total_bin) {
        const int64_t parent_hist_item = larger_leaf_hist[global_thread_index];
        const int32_t smaller_hist_item = smaller_leaf_hist[global_thread_index];
        const int64_t smaller_hist_item_int64 = (static_cast<int64_t>(static_cast<int16_t>(smaller_hist_item >> 16)) << 32) |
          static_cast<int64_t>(smaller_hist_item & 0x0000ffff);
        const int64_t larger_hist_item = parent_hist_item - smaller_hist_item_int64;
        buffer[global_thread_index] = static_cast<int32_t>(static_cast<int16_t>(larger_hist_item >> 32) << 16) |
          static_cast<int32_t>(larger_hist_item & 0x000000000000ffff);
      }
    } else if (SMALLER_USE_16BIT_HIST) {
        const int32_t* smaller_leaf_hist = reinterpret_cast<const int32_t*>(cuda_smaller_leaf_splits->hist_in_leaf);
        int64_t* larger_leaf_hist = reinterpret_cast<int64_t*>(cuda_larger_leaf_splits->hist_in_leaf);
        if (global_thread_index < num_total_bin) {
          const int64_t parent_hist_item = larger_leaf_hist[global_thread_index];
          const int32_t smaller_hist_item = smaller_leaf_hist[global_thread_index];
          const int64_t smaller_hist_item_int64 = (static_cast<int64_t>(static_cast<int16_t>(smaller_hist_item >> 16)) << 32) |
            static_cast<int64_t>(smaller_hist_item & 0x0000ffff);
          const int64_t larger_hist_item = parent_hist_item - smaller_hist_item_int64;
          larger_leaf_hist[global_thread_index] = larger_hist_item;
        }
    } else {
      const int64_t* smaller_leaf_hist = reinterpret_cast<const int64_t*>(cuda_smaller_leaf_splits->hist_in_leaf);
      int64_t* larger_leaf_hist = reinterpret_cast<int64_t*>(cuda_larger_leaf_splits->hist_in_leaf);
      if (global_thread_index < num_total_bin) {
        larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
      }
    }
  }
}

__global__ void CopyChangedNumBitHistogram(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  hist_t* num_bit_change_buffer) {
  int32_t* hist_dst = reinterpret_cast<int32_t*>(cuda_larger_leaf_splits->hist_in_leaf);
  const int32_t* hist_src = reinterpret_cast<const int32_t*>(num_bit_change_buffer);
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (global_thread_index < static_cast<unsigned int>(num_total_bin)) {
    hist_dst[global_thread_index] = hist_src[global_thread_index];
  }
}

template <bool USE_16BIT_HIST>
__global__ void FixHistogramDiscretizedKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits) {
  __shared__ int64_t shared_mem_buffer[32];
  const unsigned int blockIdx_x = blockIdx.x;
  const int feature_index = cuda_need_fix_histogram_features[blockIdx_x];
  const uint32_t num_bin_aligned = cuda_need_fix_histogram_features_num_bin_aligned[blockIdx_x];
  const uint32_t feature_hist_offset = cuda_feature_hist_offsets[feature_index];
  const uint32_t most_freq_bin = cuda_feature_most_freq_bins[feature_index];
  if (USE_16BIT_HIST) {
    const int64_t leaf_sum_gradients_hessians_int64 = cuda_smaller_leaf_splits->sum_of_gradients_hessians;
    const int32_t leaf_sum_gradients_hessians =
      (static_cast<int32_t>(leaf_sum_gradients_hessians_int64 >> 32) << 16) | static_cast<int32_t>(leaf_sum_gradients_hessians_int64 & 0x000000000000ffff);
    int32_t* feature_hist = reinterpret_cast<int32_t*>(cuda_smaller_leaf_splits->hist_in_leaf) + feature_hist_offset;
    const unsigned int threadIdx_x = threadIdx.x;
    const uint32_t num_bin = cuda_feature_num_bins[feature_index];
    const int32_t bin_gradient_hessian = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[threadIdx_x] : 0;
    const int32_t sum_gradient_hessian = ShuffleReduceSum<int32_t>(
      bin_gradient_hessian,
      reinterpret_cast<int32_t*>(shared_mem_buffer),
      num_bin_aligned);
    if (threadIdx_x == 0) {
      feature_hist[most_freq_bin] = leaf_sum_gradients_hessians - sum_gradient_hessian;
    }
  } else {
    const int64_t leaf_sum_gradients_hessians = cuda_smaller_leaf_splits->sum_of_gradients_hessians;
    int64_t* feature_hist = reinterpret_cast<int64_t*>(cuda_smaller_leaf_splits->hist_in_leaf) + feature_hist_offset;
    const unsigned int threadIdx_x = threadIdx.x;
    const uint32_t num_bin = cuda_feature_num_bins[feature_index];
    const int64_t bin_gradient_hessian = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[threadIdx_x] : 0;
    const int64_t sum_gradient_hessian = ShuffleReduceSum<int64_t>(bin_gradient_hessian, shared_mem_buffer, num_bin_aligned);
    if (threadIdx_x == 0) {
      feature_hist[most_freq_bin] = leaf_sum_gradients_hessians - sum_gradient_hessian;
    }
  }
}

void CUDAHistogramConstructor::LaunchSubtractHistogramKernel(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  const bool use_discretized_grad,
  const uint8_t parent_num_bits_in_histogram_bins,
  const uint8_t smaller_num_bits_in_histogram_bins,
  const uint8_t larger_num_bits_in_histogram_bins) {
  if (!use_discretized_grad) {
    const int num_subtract_threads = 2 * num_total_bin_;
    const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
    global_timer.Start("CUDAHistogramConstructor::FixHistogramKernel", nccl_thread_index_);
    if (need_fix_histogram_features_.size() > 0) {
      FixHistogramKernel<<<need_fix_histogram_features_.size(), FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
        cuda_feature_num_bins_,
        cuda_feature_hist_offsets_,
        cuda_feature_most_freq_bins_,
        cuda_need_fix_histogram_features_,
        cuda_need_fix_histogram_features_num_bin_aligned_,
        cuda_smaller_leaf_splits);
    }
    global_timer.Stop("CUDAHistogramConstructor::FixHistogramKernel", nccl_thread_index_);
    global_timer.Start("CUDAHistogramConstructor::SubtractHistogramKernel", nccl_thread_index_);
    SubtractHistogramKernel<<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
      num_total_bin_,
      cuda_smaller_leaf_splits,
      cuda_larger_leaf_splits);
    global_timer.Stop("CUDAHistogramConstructor::SubtractHistogramKernel", nccl_thread_index_);
  } else {
    const int num_subtract_threads = num_total_bin_;
    const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
    global_timer.Start("CUDAHistogramConstructor::FixHistogramDiscretizedKernel", nccl_thread_index_);
    if (need_fix_histogram_features_.size() > 0) {
      if (smaller_num_bits_in_histogram_bins <= 16) {
        FixHistogramDiscretizedKernel<true><<<need_fix_histogram_features_.size(), FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
          cuda_feature_num_bins_,
          cuda_feature_hist_offsets_,
          cuda_feature_most_freq_bins_,
          cuda_need_fix_histogram_features_,
          cuda_need_fix_histogram_features_num_bin_aligned_,
          cuda_smaller_leaf_splits);
      } else {
        FixHistogramDiscretizedKernel<false><<<need_fix_histogram_features_.size(), FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
          cuda_feature_num_bins_,
          cuda_feature_hist_offsets_,
          cuda_feature_most_freq_bins_,
          cuda_need_fix_histogram_features_,
          cuda_need_fix_histogram_features_num_bin_aligned_,
          cuda_smaller_leaf_splits);
      }
    }
    global_timer.Stop("CUDAHistogramConstructor::FixHistogramDiscretizedKernel", nccl_thread_index_);
    global_timer.Start("CUDAHistogramConstructor::SubtractHistogramDiscretizedKernel", nccl_thread_index_);
    if (parent_num_bits_in_histogram_bins <= 16) {
      CHECK_LE(smaller_num_bits_in_histogram_bins, 16);
      CHECK_LE(larger_num_bits_in_histogram_bins, 16);
      SubtractHistogramDiscretizedKernel<true, true, true><<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
        num_total_bin_,
        cuda_smaller_leaf_splits,
        cuda_larger_leaf_splits,
        hist_buffer_for_num_bit_change_.RawData());
    } else if (larger_num_bits_in_histogram_bins <= 16) {
      CHECK_LE(smaller_num_bits_in_histogram_bins, 16);
      SubtractHistogramDiscretizedKernel<true, true, false><<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
        num_total_bin_,
        cuda_smaller_leaf_splits,
        cuda_larger_leaf_splits,
        hist_buffer_for_num_bit_change_.RawData());
      CopyChangedNumBitHistogram<<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
        num_total_bin_,
        cuda_larger_leaf_splits,
        hist_buffer_for_num_bit_change_.RawData());
    } else if (smaller_num_bits_in_histogram_bins <= 16) {
      SubtractHistogramDiscretizedKernel<true, false, false><<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
        num_total_bin_,
        cuda_smaller_leaf_splits,
        cuda_larger_leaf_splits,
        hist_buffer_for_num_bit_change_.RawData());
    } else {
      SubtractHistogramDiscretizedKernel<false, false, false><<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
        num_total_bin_,
        cuda_smaller_leaf_splits,
        cuda_larger_leaf_splits,
        hist_buffer_for_num_bit_change_.RawData());
    }
    global_timer.Stop("CUDAHistogramConstructor::SubtractHistogramDiscretizedKernel", nccl_thread_index_);
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
