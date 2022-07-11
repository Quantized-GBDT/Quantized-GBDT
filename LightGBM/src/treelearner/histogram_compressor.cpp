/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "histogram_compressor.hpp"
#include <random>
#include <chrono>
#include "../external_libs/FastPFor/headers/codecfactory.h"

namespace LightGBM {

using namespace FastPForLib;

HistogramCompressor::HistogramCompressor(const int num_threads) {
  num_threads_ = num_threads > 0 ? num_threads : OMP_NUM_THREADS();
  thread_first_bits_.resize(num_threads_, 0);
  thread_first_.resize(num_threads_, 0);
  thread_total_half_bytes_offset_.resize(num_threads_ + 1, 0);
}

template <typename S_HIST_T, typename U_HIST_T>
void HistogramCompressor::Compress(const S_HIST_T* in_buffer, uint8_t* out_buffer, data_size_t num_bin) {
  const data_size_t block_size = (num_bin + num_threads_ - 1) / num_threads_;
  const uint32_t total_size_out_bits_buffer = (num_bin * 2 + 3) / 4;
  uint32_t* out_len = reinterpret_cast<uint32_t*>(out_buffer);
  uint32_t* out_num_threads = reinterpret_cast<uint32_t*>(out_buffer + 4);
  uint32_t* out_thread_info = reinterpret_cast<uint32_t*>(out_buffer + 8);
  out_buffer += 8 + 4 * (num_threads_ + 1);
  uint8_t* out_bits_buffer = (out_buffer) + 2 * num_bin * 8;
  global_timer.Start("HistogramCompressor::Compress::ComputeThreadHalfBytes");
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (uint32_t i = 0; i < total_size_out_bits_buffer; ++i) {
    out_bits_buffer[i] = 0;
  }
  thread_total_half_bytes_offset_[0] = 0;
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t start = thread_index * block_size;
    const data_size_t end = std::min(start + block_size, num_bin);
    const uint32_t thread_total_bytes = ComputeThreadHalfBytes<S_HIST_T, U_HIST_T>(
      in_buffer,
      out_bits_buffer,
      &thread_first_bits_[thread_index],
      start, end);
    thread_total_half_bytes_offset_[thread_index + 1] = thread_total_bytes;
  }
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t start = thread_index * block_size;
    if (start % 2 != 0) {
      out_bits_buffer[start / 2] |= thread_first_bits_[thread_index];
    }
    thread_total_half_bytes_offset_[thread_index + 1] += thread_total_half_bytes_offset_[thread_index];
  }
  global_timer.Stop("HistogramCompressor::Compress::ComputeThreadHalfBytes");
  global_timer.Start("HistogramCompressor::Compress::WriteThreadCompressedData");
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t start = thread_index * block_size;
    const data_size_t end = std::min(start + block_size, num_bin);
    WriteThreadCompressedData<S_HIST_T, U_HIST_T>(in_buffer,
      out_bits_buffer, out_buffer,
      &thread_first_[thread_index],
      start, end, thread_total_half_bytes_offset_[thread_index]);
  }
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const uint32_t cur_half_bytes = thread_total_half_bytes_offset_[thread_index];
    const uint32_t pos = cur_half_bytes / 2;
    const uint8_t offset = cur_half_bytes % 2;
    if (offset == 1) {
      out_buffer[pos] &= 0x0f;
      out_buffer[pos] |= thread_first_[thread_index];
    }
  }
  CHECK_LE(thread_total_half_bytes_offset_.back(), 2 * static_cast<uint32_t>(num_bin) * 16);
  const uint32_t bit_start_pos = (thread_total_half_bytes_offset_.back() + 1) / 2;
  uint8_t* bits_write_ptr = out_buffer + bit_start_pos;
  const int num_bytes_for_bits = (num_bin + 1) / 2;
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < num_bytes_for_bits; ++i) {
    bits_write_ptr[i] = out_bits_buffer[i];
  }
  for (int thread_index = 0; thread_index < num_threads_ + 1; ++thread_index) {
    out_thread_info[thread_index] = thread_total_half_bytes_offset_[thread_index];
  }
  *out_len = (thread_total_half_bytes_offset_.back() + 1) / 2 + static_cast<uint32_t>(num_bytes_for_bits);
  *out_num_threads = static_cast<uint32_t>(num_threads_);
  global_timer.Stop("HistogramCompressor::Compress::WriteThreadCompressedData");
}

template <typename S_HIST_T, typename U_HIST_T>
uint32_t HistogramCompressor::ComputeThreadHalfBytes(
  const S_HIST_T* in_buffer,
  uint8_t* out_bits_buffer,
  uint8_t* thread_first_bits_buffer,
  data_size_t start_bin,
  data_size_t end_bin) {
  int32_t prev_hess =  static_cast<int32_t>(start_bin == 0 ? 0 : static_cast<U_HIST_T>(in_buffer[(start_bin - 1) << 1]));
  int32_t prev_grad =  static_cast<int32_t>(start_bin == 0 ? 0 : in_buffer[((start_bin - 1) << 1) + 1]);
  uint32_t total_half_bytes = 0;
  data_size_t bin = start_bin;
  *thread_first_bits_buffer = 0;
  for (; bin < (start_bin + 1) / 2 * 2; ++bin) {
    const data_size_t bin_offset = (bin << 1);
    const int32_t hess = static_cast<int32_t>(static_cast<U_HIST_T>(in_buffer[bin_offset]));
    const int32_t hess_diff = hess - prev_hess;
    const uint8_t hess_offset = (bin_offset % 4) << 1;
    const int32_t grad = static_cast<int32_t>(in_buffer[bin_offset + 1]);
    const int32_t grad_diff = grad - prev_grad;
    const uint8_t grad_offset = ((bin_offset + 1) % 4) << 1;
    if (hess_diff >= -8 && hess_diff < 8) {
      total_half_bytes += 1;
    } else if (hess_diff >= -128 && hess_diff < 128) {
      total_half_bytes += 2;
      (*thread_first_bits_buffer) |= (0x01 << hess_offset);
    } else if (hess_diff >= -32768 && hess_diff < 32768) {
      total_half_bytes += 4;
      (*thread_first_bits_buffer) |= (0x02 << hess_offset);
    } else {
      total_half_bytes += 8;
      (*thread_first_bits_buffer) |= (0x03 << hess_offset);
    }
    if (grad_diff >= -8 && grad_diff < 8) {
      total_half_bytes += 1;
    } else if (grad_diff >= -128 && grad_diff < 128) {
      total_half_bytes += 2;
      (*thread_first_bits_buffer) |= (0x01 << grad_offset);
    } else if (grad_diff >= -32768 && grad_diff < 32768) {
      total_half_bytes += 4;
      (*thread_first_bits_buffer) |= (0x02 << grad_offset);
    } else {
      total_half_bytes += 8;
      (*thread_first_bits_buffer) |= (0x03 << grad_offset);
    }
    prev_grad = grad;
    prev_hess = hess;
  }
  uint8_t one_byte = 0;
  uint32_t grad_pos = 0;
  uint8_t grad_offset = 0;
  for (; bin < end_bin; ++bin) {
    const data_size_t bin_offset = (bin << 1);
    const int32_t hess = static_cast<int32_t>(static_cast<U_HIST_T>(in_buffer[bin_offset]));
    const int32_t hess_diff = hess - prev_hess;
    const uint32_t hess_pos = (bin_offset / 4);
    const uint8_t hess_offset = (bin_offset % 4) << 1;
    const int32_t grad = static_cast<int32_t>(in_buffer[bin_offset + 1]);
    const int32_t grad_diff = grad - prev_grad;
    grad_pos = ((bin_offset + 1) / 4);
    grad_offset = ((bin_offset + 1) % 4) << 1;
    const int32_t hess_diff_pos = hess_diff >= 0 ? hess_diff : ~hess_diff;
    uint8_t new_mask = (static_cast<uint8_t>(static_cast<bool>(hess_diff_pos & 0xfffffff8)) +
      static_cast<uint8_t>(static_cast<bool>(hess_diff_pos & 0xffffff80)) +
      static_cast<uint8_t>(static_cast<bool>(hess_diff_pos & 0xffff8000)));
    one_byte |= (new_mask << hess_offset);
    total_half_bytes += (1 << new_mask);
    if (hess_offset == 6) {
      out_bits_buffer[hess_pos] = one_byte;
      one_byte = 0;
    }

    const int32_t grad_diff_pos = grad_diff >= 0 ? grad_diff : ~grad_diff;
    new_mask = (static_cast<uint8_t>(static_cast<bool>(grad_diff_pos & 0xfffffff8)) +
      static_cast<uint8_t>(static_cast<bool>(grad_diff_pos & 0xffffff80)) +
      static_cast<uint8_t>(static_cast<bool>(grad_diff_pos & 0xffff8000)));
    one_byte |= (new_mask << grad_offset);
    total_half_bytes += (1 << new_mask);
    if (grad_offset == 6) {
      out_bits_buffer[grad_pos] = one_byte;
      one_byte = 0;
    }
    prev_grad = grad;
    prev_hess = hess;
  }
  if (grad_offset != 6) {
    out_bits_buffer[grad_pos] = one_byte;
  }
  return total_half_bytes;
}

template <typename S_HIST_T, typename U_HIST_T>
void HistogramCompressor::WriteThreadCompressedData(const S_HIST_T* in_buffer, const uint8_t* bits_buffer,
  uint8_t* out_buffer,
  uint8_t* thread_first_buffer,
  data_size_t start_bin,
  data_size_t end_bin,
  uint32_t thread_start_half_bytes) {
  int32_t prev_hess = (start_bin == 0 ? 0 : static_cast<U_HIST_T>(in_buffer[((start_bin - 1) << 1)]));
  int32_t prev_grad = (start_bin == 0 ? 0 : in_buffer[((start_bin - 1) << 1) + 1]);

  uint32_t cur_half_bytes = thread_start_half_bytes;
  *thread_first_buffer = 0;
  data_size_t bin = start_bin;
  if (thread_start_half_bytes % 2 == 1) {
    const data_size_t bin_offset = (bin << 1);
    const uint32_t hess_bits_pos = bin_offset / 4;
    const uint8_t hess_bits_offset = (bin_offset % 4) << 1;
    const uint8_t hess_bits = (bits_buffer[hess_bits_pos] >> hess_bits_offset) & 0x03;
    const int32_t hess = static_cast<int32_t>(static_cast<U_HIST_T>(in_buffer[bin_offset]));
    const int32_t hess_diff = hess - prev_hess;
    const uint32_t grad_bits_pos = (bin_offset + 1) / 4;
    const uint8_t grad_bits_offset = ((bin_offset + 1) % 4) << 1;
    const uint8_t grad_bits = (bits_buffer[grad_bits_pos] >> grad_bits_offset) & 0x03;
    const int32_t grad = static_cast<int32_t>(in_buffer[bin_offset + 1]);
    const int32_t grad_diff = grad - prev_grad;
    prev_grad = grad;
    prev_hess = hess;
    const uint32_t hess_pos = cur_half_bytes / 2;
    if (hess_bits == 0) {
      (*thread_first_buffer) |= static_cast<uint8_t>(hess_diff << 4);
      ++cur_half_bytes;
    } else if (hess_bits == 1) {
      (*thread_first_buffer) |= static_cast<uint8_t>(hess_diff << 4);
      out_buffer[hess_pos + 1] = (static_cast<uint8_t>(hess_diff >> 4) & 0x0f);
      cur_half_bytes += 2;
    } else if (hess_bits == 2) {
      (*thread_first_buffer) |= static_cast<uint8_t>(hess_diff << 4);
      out_buffer[hess_pos + 1] = static_cast<uint8_t>(hess_diff >> 4);
      out_buffer[hess_pos + 2] = (static_cast<uint8_t>(hess_diff >> 12) & 0x0f);
      cur_half_bytes += 4;
    } else {
      (*thread_first_buffer) |= static_cast<uint8_t>(hess_diff << 4);
      out_buffer[hess_pos + 1] = static_cast<uint8_t>(hess_diff >> 4);
      out_buffer[hess_pos + 2] = static_cast<uint8_t>(hess_diff >> 12);
      out_buffer[hess_pos + 3] = static_cast<uint8_t>(hess_diff >> 20);
      out_buffer[hess_pos + 4] = (static_cast<uint8_t>(hess_diff >> 28) & 0x0f);
      cur_half_bytes += 8;
    }
    const uint32_t grad_pos = cur_half_bytes / 2;
    const uint8_t grad_offset = cur_half_bytes % 2;
    if (grad_offset == 0) {
      if (grad_bits == 0) {
        out_buffer[grad_pos] = (static_cast<uint8_t>(grad_diff) & 0x0f);
        ++cur_half_bytes;
      } else if (grad_bits == 1) {
        out_buffer[grad_pos] = static_cast<uint8_t>(grad_diff);
        cur_half_bytes += 2;
      } else if (grad_bits == 2) {
        out_buffer[grad_pos] = static_cast<uint8_t>(grad_diff);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 8);
        cur_half_bytes += 4;
      } else if (grad_bits == 3) {
        out_buffer[grad_pos] = static_cast<uint8_t>(grad_diff);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 8);
        out_buffer[grad_pos + 2] = static_cast<uint8_t>(grad_diff >> 16);
        out_buffer[grad_pos + 3] = static_cast<uint8_t>(grad_diff >> 24);
        cur_half_bytes += 8;
      }
    } else {
      if (grad_bits == 0) {
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        ++cur_half_bytes;
      } else if (grad_bits == 1) {
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        out_buffer[grad_pos + 1] = (static_cast<uint8_t>(grad_diff >> 4) & 0x0f);
        cur_half_bytes += 2;
      } else if (grad_bits == 2) {
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 4);
        out_buffer[grad_pos + 2] = (static_cast<uint8_t>(grad_diff >> 12) & 0x0f);
        cur_half_bytes += 4;
      } else if (grad_bits == 3) {
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 4);
        out_buffer[grad_pos + 2] = static_cast<uint8_t>(grad_diff >> 12);
        out_buffer[grad_pos + 3] = static_cast<uint8_t>(grad_diff >> 20);
        out_buffer[grad_pos + 4] = (static_cast<uint8_t>(grad_diff >> 28) & 0x0f);
        cur_half_bytes += 8;
      }
    }
    ++bin;
  }
  for (; bin < end_bin; ++bin) {
    const data_size_t bin_offset = (bin << 1);
    const uint32_t hess_bits_pos = bin_offset / 4;
    const uint8_t hess_bits_offset = (bin_offset % 4) << 1;
    const uint8_t hess_bits = (bits_buffer[hess_bits_pos] >> hess_bits_offset) & 0x03;
    const int32_t hess = static_cast<int32_t>(static_cast<U_HIST_T>(in_buffer[bin_offset]));
    const int32_t hess_diff = hess - prev_hess;
    const uint32_t grad_bits_pos = (bin_offset + 1) / 4;
    const uint8_t grad_bits_offset = ((bin_offset + 1) % 4) << 1;
    const uint8_t grad_bits = (bits_buffer[grad_bits_pos] >> grad_bits_offset) & 0x03;
    const int32_t grad = static_cast<int32_t>(in_buffer[bin_offset + 1]);
    const int32_t grad_diff = grad - prev_grad;
    prev_grad = grad;
    prev_hess = hess;
    const uint32_t hess_pos = cur_half_bytes / 2;
    const uint8_t hess_offset = cur_half_bytes % 2;
    if (hess_offset == 1) {
      if (hess_bits == 0) {
        out_buffer[hess_pos] |= static_cast<uint8_t>(hess_diff << 4);
        ++cur_half_bytes;
      } else if (hess_bits == 1) {
        out_buffer[hess_pos] |= static_cast<uint8_t>(hess_diff << 4);
        out_buffer[hess_pos + 1] = (static_cast<uint8_t>(hess_diff >> 4) & 0x0f);
        cur_half_bytes += 2;
      } else if (hess_bits == 2) {
        out_buffer[hess_pos] |= static_cast<uint8_t>(hess_diff << 4);
        out_buffer[hess_pos + 1] = static_cast<uint8_t>(hess_diff >> 4);
        out_buffer[hess_pos + 2] = (static_cast<uint8_t>(hess_diff >> 12) & 0x0f);
        cur_half_bytes += 4;
      } else {
        out_buffer[hess_pos] |= static_cast<uint8_t>(hess_diff << 4);
        out_buffer[hess_pos + 1] = static_cast<uint8_t>(hess_diff >> 4);
        out_buffer[hess_pos + 2] = static_cast<uint8_t>(hess_diff >> 12);
        out_buffer[hess_pos + 3] = static_cast<uint8_t>(hess_diff >> 20);
        out_buffer[hess_pos + 4] = (static_cast<uint8_t>(hess_diff >> 28) & 0x0f);
        cur_half_bytes += 8;
      }
    } else {
      if (hess_bits == 0) {
        out_buffer[hess_pos] = (static_cast<uint8_t>(hess_diff) & 0x0f);
        ++cur_half_bytes;
      } else if (hess_bits == 1) {
        out_buffer[hess_pos] = static_cast<uint8_t>(hess_diff);
        cur_half_bytes += 2;
      } else if (hess_bits == 2) {
        out_buffer[hess_pos] = static_cast<uint8_t>(hess_diff);
        out_buffer[hess_pos + 1] = static_cast<uint8_t>(hess_diff >> 8);
        cur_half_bytes += 4;
      } else if (hess_bits == 3) {
        out_buffer[hess_pos] = static_cast<uint8_t>(hess_diff);
        out_buffer[hess_pos + 1] = static_cast<uint8_t>(hess_diff >> 8);
        out_buffer[hess_pos + 2] = static_cast<uint8_t>(hess_diff >> 16);
        out_buffer[hess_pos + 3] = static_cast<uint8_t>(hess_diff >> 24);
        cur_half_bytes += 8;
      }
    }
    const uint32_t grad_pos = cur_half_bytes / 2;
    const uint8_t grad_offset = cur_half_bytes % 2;
    if (grad_offset == 0) {
      if (grad_bits == 0) {
        out_buffer[grad_pos] = (static_cast<uint8_t>(grad_diff) & 0x0f);
        ++cur_half_bytes;
      } else if (grad_bits == 1) {
        out_buffer[grad_pos] = static_cast<uint8_t>(grad_diff);
        cur_half_bytes += 2;
      } else if (grad_bits == 2) {
        out_buffer[grad_pos] = static_cast<uint8_t>(grad_diff);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 8);
        cur_half_bytes += 4;
      } else if (grad_bits == 3) {
        out_buffer[grad_pos] = static_cast<uint8_t>(grad_diff);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 8);
        out_buffer[grad_pos + 2] = static_cast<uint8_t>(grad_diff >> 16);
        out_buffer[grad_pos + 3] = static_cast<uint8_t>(grad_diff >> 24);
        cur_half_bytes += 8;
      }
    } else {
      if (grad_bits == 0) {
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        ++cur_half_bytes;
      } else if (grad_bits == 1) {
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        out_buffer[grad_pos + 1] = (static_cast<uint8_t>(grad_diff >> 4) & 0x0f);
        cur_half_bytes += 2;
      } else if (grad_bits == 2) {
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 4);
        out_buffer[grad_pos + 2] = (static_cast<uint8_t>(grad_diff >> 12) & 0x0f);
        cur_half_bytes += 4;
      } else if (grad_bits == 3) {
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 4);
        out_buffer[grad_pos + 2] = static_cast<uint8_t>(grad_diff >> 12);
        out_buffer[grad_pos + 3] = static_cast<uint8_t>(grad_diff >> 20);
        out_buffer[grad_pos + 4] = (static_cast<uint8_t>(grad_diff >> 28) & 0x0f);
        cur_half_bytes += 8;
      }
    }
  }
}

template <typename S_HIST_T, typename U_HIST_T>
void HistogramCompressor::Decompress(const uint8_t* in_buffer, data_size_t num_bin, S_HIST_T* out_buffer) {
  global_timer.Start("HistogramCompressor::Decompress");
  const int num_threads = static_cast<int>(*reinterpret_cast<const uint32_t*>(in_buffer + 4));
  num_threads_ = num_threads;
  const data_size_t block_size = (num_bin + num_threads_ - 1) / num_threads_;
  const uint32_t* thread_total_half_bytes_offset = reinterpret_cast<const uint32_t*>(in_buffer + 8);
  in_buffer += 8;
  in_buffer += (num_threads_ + 1) * 4;
  const uint8_t* in_bits_buffer = in_buffer + (thread_total_half_bytes_offset[num_threads_] + 1) / 2;
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t start = thread_index * block_size;
    const data_size_t end = std::min(start + block_size, num_bin);
    const uint32_t half_byte_start = thread_total_half_bytes_offset[thread_index];
    uint32_t cur_half_byte = half_byte_start;
    int32_t prev_grad = 0;
    int32_t prev_hess = 0;
    int32_t grad = 0;
    int32_t hess = 0;
    for (data_size_t bin = start; bin < end; ++bin) {
      const data_size_t bin_offset = (bin << 1);
      const uint32_t hess_bits_pos = bin_offset / 4;
      const uint8_t hess_bits_offset = (bin_offset % 4) << 1;
      const uint8_t hess_bits = (in_bits_buffer[hess_bits_pos] >> hess_bits_offset) & 0x03;
      const uint8_t hess_offset = cur_half_byte % 2;
      const uint32_t hess_pos = cur_half_byte / 2;
      if (hess_offset == 1) {
        if (hess_bits == 0) {
          hess = static_cast<int32_t>(static_cast<int8_t>(in_buffer[hess_pos] & 0xf0)) >> 4;
          ++cur_half_byte;
        } else if (hess_bits == 1) {
          hess = static_cast<int32_t>(static_cast<uint8_t>(in_buffer[hess_pos] >> 4) & 0x0f) |
            static_cast<int32_t>(static_cast<int8_t>(in_buffer[hess_pos + 1] << 4));
          cur_half_byte += 2;
        } else if (hess_bits == 2) {
          hess = static_cast<int32_t>(static_cast<uint8_t>(in_buffer[hess_pos] >> 4) & 0x0f) |
            (static_cast<int32_t>(static_cast<uint8_t>(in_buffer[hess_pos + 1])) << 4) |
            (static_cast<int32_t>(static_cast<int8_t>(in_buffer[hess_pos + 2] << 4)) << 8);
          cur_half_byte += 4;
        } else if (hess_bits == 3) {
          hess = static_cast<int32_t>(static_cast<uint8_t>(in_buffer[hess_pos] >> 4) & 0x0f) |
            (static_cast<int32_t>(static_cast<uint8_t>(in_buffer[hess_pos + 1])) << 4) |
            (static_cast<int32_t>(static_cast<uint8_t>(in_buffer[hess_pos + 2])) << 12) |
            (static_cast<int32_t>(static_cast<uint8_t>(in_buffer[hess_pos + 3])) << 20) |
            (static_cast<int32_t>(static_cast<int8_t>(in_buffer[hess_pos + 4] << 4)) << 24);
          cur_half_byte += 8;
        }
      } else {
        if (hess_bits == 0) {
          hess = static_cast<int32_t>(static_cast<int8_t>(in_buffer[hess_pos] << 4)) >> 4;
          cur_half_byte += 1;
        } else if (hess_bits == 1) {
          hess = static_cast<int32_t>(static_cast<int8_t>(in_buffer[hess_pos]));
          cur_half_byte += 2;
        } else if (hess_bits == 2) {
          hess = static_cast<int32_t>(static_cast<uint8_t>(in_buffer[hess_pos])) |
            (static_cast<int32_t>(static_cast<int8_t>(in_buffer[hess_pos + 1])) << 8);
          cur_half_byte += 4;
        } else if (hess_bits == 3) {
          hess = static_cast<int32_t>(static_cast<uint8_t>(in_buffer[hess_pos])) |
            (static_cast<int32_t>(static_cast<uint8_t>(in_buffer[hess_pos + 1])) << 8) |
            (static_cast<int32_t>(static_cast<uint8_t>(in_buffer[hess_pos + 2])) << 16) |
            (static_cast<int32_t>(static_cast<int8_t>(in_buffer[hess_pos + 3])) << 24);
          cur_half_byte += 8;
        }
      }
      hess += prev_hess;
      prev_hess = hess;
      out_buffer[bin_offset] = static_cast<U_HIST_T>(hess);
      const uint32_t grad_bits_pos = (bin_offset + 1) / 4;
      const uint8_t grad_bits_offset = ((bin_offset + 1) % 4) << 1;
      const uint8_t grad_bits = (in_bits_buffer[grad_bits_pos] >> grad_bits_offset) & 0x03;
      const uint8_t grad_offset = cur_half_byte % 2;
      const uint32_t grad_pos = cur_half_byte / 2;
      if (grad_offset == 1) {
        if (grad_bits == 0) {
          grad = static_cast<int32_t>(static_cast<int8_t>(in_buffer[grad_pos] & 0xf0)) >> 4;
          ++cur_half_byte;
        } else if (grad_bits == 1) {
          grad = static_cast<int32_t>(static_cast<uint8_t>(in_buffer[grad_pos] >> 4) & 0x0f) |
            static_cast<int32_t>(static_cast<int8_t>(in_buffer[grad_pos + 1] << 4));
          cur_half_byte += 2;
        } else if (grad_bits == 2) {
          grad = static_cast<int32_t>(static_cast<uint8_t>(in_buffer[grad_pos] >> 4) & 0x0f) |
            (static_cast<int32_t>(static_cast<uint8_t>(in_buffer[grad_pos + 1])) << 4) |
            (static_cast<int32_t>(static_cast<int8_t>(in_buffer[grad_pos + 2] << 4)) << 8);
          cur_half_byte += 4;
        } else if (grad_bits == 3) {
          grad = static_cast<int32_t>(static_cast<uint8_t>(in_buffer[grad_pos] >> 4) & 0x0f) |
            (static_cast<int32_t>(static_cast<uint8_t>(in_buffer[grad_pos + 1])) << 4) |
            (static_cast<int32_t>(static_cast<uint8_t>(in_buffer[grad_pos + 2])) << 12) |
            (static_cast<int32_t>(static_cast<uint8_t>(in_buffer[grad_pos + 3])) << 20) |
            (static_cast<int32_t>(static_cast<int8_t>(in_buffer[grad_pos + 4] << 4)) << 24);
          cur_half_byte += 8;
        }
      } else {
        if (grad_bits == 0) {
          grad = static_cast<int32_t>(static_cast<int8_t>(in_buffer[grad_pos] << 4)) >> 4;
          cur_half_byte += 1;
        } else if (grad_bits == 1) {
          grad = static_cast<int32_t>(static_cast<int8_t>(in_buffer[grad_pos]));
          cur_half_byte += 2;
        } else if (grad_bits == 2) {
          grad = static_cast<int32_t>(static_cast<uint8_t>(in_buffer[grad_pos])) |
            (static_cast<int32_t>(static_cast<int8_t>(in_buffer[grad_pos + 1])) << 8);
          cur_half_byte += 4;
        } else if (grad_bits == 3) {
          grad = static_cast<int32_t>(static_cast<uint8_t>(in_buffer[grad_pos])) |
            (static_cast<int32_t>(static_cast<uint8_t>(in_buffer[grad_pos + 1])) << 8) |
            (static_cast<int32_t>(static_cast<uint8_t>(in_buffer[grad_pos + 2])) << 16) |
            (static_cast<int32_t>(static_cast<int8_t>(in_buffer[grad_pos + 3])) << 24);
          cur_half_byte += 8;
        }
      }
      grad += prev_grad;
      prev_grad = grad;
      out_buffer[bin_offset + 1] = static_cast<S_HIST_T>(grad);
    }
  }
  std::vector<S_HIST_T> thread_grad_offset(num_threads_ + 1, 0);
  std::vector<U_HIST_T> thread_hess_offset(num_threads_ + 1, 0);
  for (int thread_index = 1; thread_index < num_threads_ + 1; ++thread_index) {
    const data_size_t start = thread_index * block_size;
    thread_grad_offset[thread_index] = out_buffer[((start - 1) << 1) + 1] + thread_grad_offset[thread_index - 1];
    thread_hess_offset[thread_index] = static_cast<U_HIST_T>(out_buffer[(start - 1) << 1]) + thread_hess_offset[thread_index - 1];
  }
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t start = thread_index * block_size;
    const data_size_t end = std::min(start + block_size, num_bin);
    const S_HIST_T grad_offset = thread_grad_offset[thread_index];
    const U_HIST_T hess_offset = thread_hess_offset[thread_index];
    for (data_size_t bin = start; bin < end; ++bin) {
      const data_size_t bin_offset = (bin << 1);
      out_buffer[bin_offset + 1] += grad_offset;
      out_buffer[bin_offset] = static_cast<U_HIST_T>(static_cast<U_HIST_T>(out_buffer[bin_offset]) + static_cast<U_HIST_T>(hess_offset));
    }
  }
  global_timer.Stop("HistogramCompressor::Decompress");
}

template void HistogramCompressor::Compress<int32_t, uint32_t>(const int32_t* in_buffer, uint8_t* out_buffer, data_size_t num_bin);

template void HistogramCompressor::Compress<int16_t, uint16_t>(const int16_t* in_buffer, uint8_t* out_buffer, data_size_t num_bin);

template void HistogramCompressor::Decompress<int32_t, uint32_t>(const uint8_t* in_buffer, data_size_t num_bin, int32_t* out_buffer);

template void HistogramCompressor::Decompress<int16_t, uint16_t>(const uint8_t* in_buffer, data_size_t num_bin, int16_t* out_buffer);

void HistogramCompressor::Test() {
  const size_t test_len = 12424;
  std::vector<int16_t> int16_test_array(test_len * 2);
  std::mt19937 rand_eng(0);
  const int16_t num_range = 100;
  std::vector<double> start_prob(num_range, 1.0f / num_range);
  std::discrete_distribution<int16_t> dist_start(start_prob.begin(), start_prob.end());
  const int16_t diff_range = 100;
  std::vector<double> diff_prob(diff_range, 1.0f / diff_range);
  std::discrete_distribution<int16_t> dist_diff(diff_prob.begin(), diff_prob.end());
  std::vector<uint8_t> out_buffer(test_len * 2 * 20, 0);
  std::vector<int16_t> result(test_len * 2, 0);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 10000; ++i) {
    int16_t grad = dist_start(rand_eng);
    int16_t hess = std::abs(dist_start(rand_eng));
    for (size_t i = 0; i < test_len; ++i) {
      int16_t grad_diff = dist_diff(rand_eng) - 50;
      int16_t hess_diff = dist_diff(rand_eng) - 50;
      int16_test_array[(i << 1) + 1] = grad;
      int16_test_array[(i << 1)] = hess;
      grad = grad + grad_diff;
      hess = std::abs(hess + hess_diff);
    }
    Compress<int16_t, uint16_t>(int16_test_array.data(), out_buffer.data(), test_len);
    global_timer.Start("Decompress");
    Decompress<int16_t, uint16_t>(out_buffer.data(), static_cast<data_size_t>(test_len), result.data());
    global_timer.Stop("Decompress");

    const uint8_t* bits = out_buffer.data() + (thread_total_half_bytes_offset_.back() + 1) / 2;
    auto end = std::chrono::steady_clock::now();
    std::vector<data_size_t> num_data_per_type(4);
    for (size_t i = 0; i < test_len * 2; ++i) {
      const size_t pos = (i / 4);
      const uint8_t offset = ((i % 4) << 1);
      ++num_data_per_type[(bits[pos] >> offset) & 0x03];
    }
    for (size_t i = 0; i < num_data_per_type.size(); ++i) {
      Log::Warning("num_data_per_type[%d] = %d", i, num_data_per_type[i]);
    }
    Log::Warning("compression and depression finished in %.10f seconds", static_cast<std::chrono::duration<double>>(end - start).count());
    Log::Warning("finish decompress, total half bytes = %ld", thread_total_half_bytes_offset_.back());
    size_t total_bytes = thread_total_half_bytes_offset_.back() / 2 + test_len / 2;
    Log::Warning("compressed bytes = %ld", total_bytes);
    Log::Warning("compress ratio = %f", static_cast<double>(total_bytes) / (2 * test_len * sizeof(int16_t)));
    for (size_t i = 0; i < test_len; ++i) {
      if (int16_test_array[(i << 1) + 1] != result[(i << 1) + 1] ||
        int16_test_array[(i << 1)] != result[(i << 1)]) {
        if (i > 1) {
          Log::Warning("i = %d, grad = %d, hess = %d, grad hat = %d, hess hat = %d",
          i - 2, int16_test_array[((i - 2) << 1) + 1], int16_test_array[((i - 2) << 1)],
            result[((i - 2) << 1) + 1], result[((i - 2) << 1)]);
        }
        if (i > 0) {
          Log::Warning("i = %d, grad = %d, hess = %d, grad hat = %d, hess hat = %d",
          i - 1, int16_test_array[((i - 1) << 1) + 1], int16_test_array[((i - 1) << 1)],
            result[((i - 1) << 1) + 1], result[((i - 1) << 1)]);
        }
        Log::Warning("i = %d, grad = %d, hess = %d, grad hat = %d, hess hat = %d",
          i, int16_test_array[(i << 1) + 1], int16_test_array[(i << 1)],
            result[(i << 1) + 1], result[(i << 1)]);
      }
      CHECK_EQ(int16_test_array[(i << 1) + 1], result[(i << 1) + 1]);
      CHECK_EQ(static_cast<uint16_t>(int16_test_array[(i << 1)]), static_cast<uint16_t>(result[(i << 1)]));
    }
  }
  global_timer.Print();
}

HistogramCompressorV2::HistogramCompressorV2(const int num_threads) {
  num_threads_ = num_threads;
}

template <typename S_HIST_T, typename U_HIST_T>
void HistogramCompressorV2::Compress(const S_HIST_T* in_buffer, uint32_t* int_buffer, uint8_t* out_buffer, data_size_t num_bin) {
  uint8_t* sign_bits = reinterpret_cast<uint8_t*>(int_buffer + 2 * num_bin);
  const U_HIST_T* in_buffer_unsigned = reinterpret_cast<const U_HIST_T*>(in_buffer);
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int bin = 0; bin < num_bin; ++bin) {
    const int bin_offset = (bin << 1);
    const int32_t grad_diff = bin == 0 ? static_cast<int32_t>(in_buffer[bin_offset + 1]) :
      static_cast<int32_t>(in_buffer[bin_offset + 1]) - static_cast<int32_t>(in_buffer[bin_offset - 1]);
    const int32_t hess_diff = bin == 0 ? static_cast<int32_t>(in_buffer_unsigned[bin_offset]) :
      static_cast<int32_t>(in_buffer[bin_offset]) - static_cast<int32_t>(in_buffer[bin_offset - 2]);
    if (grad_diff < 0) {
      int_buffer[bin_offset + 1] = static_cast<uint32_t>(~grad_diff);
      sign_bits[bin_offset + 1] = 1;
    } else {
      int_buffer[bin_offset + 1] = static_cast<uint32_t>(grad_diff);
      sign_bits[bin_offset + 1] = 0;
    }
    if (hess_diff < 0) {
      int_buffer[bin_offset] = static_cast<uint32_t>(~hess_diff);
      sign_bits[bin_offset] = 1;
    } else {
      int_buffer[bin_offset] = static_cast<uint32_t>(hess_diff);
      sign_bits[bin_offset] = 0;
    }
  }
  IntegerCODEC &codec = *CODECFactory::getFromName("simdfastpfor256");
  uint32_t* compressed_len_ptr = reinterpret_cast<uint32_t*>(out_buffer);
  uint32_t* out_buffer_true = compressed_len_ptr + 1;
  size_t compressed_len = static_cast<size_t>(2 * num_bin);
  codec.encodeArray(int_buffer, 2 * num_bin, out_buffer_true, compressed_len);
  *compressed_len_ptr = static_cast<uint32_t>(compressed_len);
  uint8_t* out_sign_bits = reinterpret_cast<uint8_t*>(out_buffer_true + compressed_len);
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int bin = 0; bin < 2 * num_bin; ++bin) {
    out_sign_bits[bin] = sign_bits[bin];
  }
}

template <typename S_HIST_T, typename U_HIST_T>
void HistogramCompressorV2::Decompress(const uint8_t* in_buffer, data_size_t num_bin, uint32_t* int_buffer, S_HIST_T* out_buffer) {
  const uint32_t* compressed_len_ptr = reinterpret_cast<const uint32_t*>(in_buffer);
  const uint32_t* compressed_data = compressed_len_ptr + 1;
  U_HIST_T* out_buffer_unsigned = reinterpret_cast<U_HIST_T*>(out_buffer);
  const uint8_t* signed_bits = reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint32_t*>(in_buffer) + (*compressed_len_ptr) + 1);
  IntegerCODEC &codec = *CODECFactory::getFromName("simdfastpfor256");
  size_t decompressed_len = 2 * static_cast<size_t>(num_bin);
  codec.decodeArray(compressed_data, *compressed_len_ptr, int_buffer, decompressed_len);
  const data_size_t block_size = (num_bin + num_threads_ - 1) / num_threads_;
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t block_start = thread_index * block_size;
    const data_size_t block_end = std::min(block_start + block_size, num_bin);
    const data_size_t bin_offset = (block_start << 1);
    const uint8_t grad_signed_bit = signed_bits[bin_offset + 1];
    const uint8_t hess_signed_bit = signed_bits[bin_offset];
    const uint32_t unsigned_grad_diff = int_buffer[bin_offset + 1];
    const uint32_t unsigned_hess_diff = int_buffer[bin_offset];
    int32_t grad = (grad_signed_bit == 0 ? static_cast<int32_t>(unsigned_grad_diff) : static_cast<int32_t>(~unsigned_grad_diff));
    int32_t hess = (hess_signed_bit == 0 ? static_cast<int32_t>(unsigned_hess_diff) : static_cast<int32_t>(~unsigned_hess_diff));
    out_buffer[bin_offset] = hess;
    out_buffer[bin_offset + 1] = grad;
    for (data_size_t bin = block_start + 1; bin < block_end; ++bin) {
      const data_size_t bin_offset = (bin << 1);
      const uint8_t grad_signed_bit = signed_bits[bin_offset + 1];
      const uint8_t hess_signed_bit = signed_bits[bin_offset];
      const uint32_t unsigned_grad_diff = int_buffer[bin_offset + 1];
      const uint32_t unsigned_hess_diff = int_buffer[bin_offset];
      grad += (grad_signed_bit == 0 ? static_cast<int32_t>(unsigned_grad_diff) : static_cast<int32_t>(~unsigned_grad_diff));
      hess += (hess_signed_bit == 0 ? static_cast<int32_t>(unsigned_hess_diff) : static_cast<int32_t>(~unsigned_hess_diff));
      out_buffer[bin_offset] = hess;
      out_buffer[bin_offset + 1] = grad;
    }
  }
  std::vector<int32_t> thread_grad_base(num_threads_ + 1, 0);
  std::vector<uint32_t> thread_hess_base(num_threads_ + 1, 0);
  for (int thread_index = 1; thread_index < num_threads_ + 1; ++thread_index) {
    const data_size_t start = (thread_index - 1) * block_size;
    const data_size_t end = std::min(start + block_size, num_bin) - 1;
    const data_size_t bin_offset = (end << 1);
    thread_grad_base[thread_index] += thread_grad_base[thread_index - 1] + out_buffer[bin_offset + 1];
    thread_hess_base[thread_index] += thread_hess_base[thread_index - 1] + out_buffer[bin_offset];
  }
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    if (thread_index > 0) {
      const data_size_t block_start = thread_index * block_size;
      const data_size_t block_end = std::min(block_start + block_size, num_bin);
      const S_HIST_T grad_base = thread_grad_base[thread_index];
      const S_HIST_T hess_base = thread_hess_base[thread_index];
      for (data_size_t bin = block_start; bin < block_end; ++bin) {
        const data_size_t bin_offset = (bin << 1);
        out_buffer_unsigned[bin_offset] = static_cast<U_HIST_T>(out_buffer[bin_offset] + hess_base);
        out_buffer[bin_offset + 1] = out_buffer[bin_offset + 1] + grad_base;
      }
    }
  }
  /*for (int bin = 0; bin < num_bin; ++bin) {
    const int bin_offset = (bin << 1);
    const uint32_t unsigned_grad_diff = int_buffer[bin_offset + 1];
    const uint8_t grad_signed_bit = signed_bits[bin_offset + 1];
    if (grad_signed_bit == 1) {
      out_buffer[bin_offset + 1] = static_cast<S_HIST_T>(~unsigned_grad_diff);
    } else {
      out_buffer[bin_offset + 1] = static_cast<S_HIST_T>(unsigned_grad_diff);
    }
    const uint32_t unsigned_hess_diff = int_buffer[bin_offset];
    const uint8_t hess_signed_bit = signed_bits[bin_offset];
    if (hess_signed_bit == 1) {
      out_buffer[bin_offset] = static_cast<S_HIST_T>(~unsigned_hess_diff);
    } else {
      out_buffer[bin_offset] = static_cast<S_HIST_T>(unsigned_hess_diff);
    }
  }*/
  if (decompressed_len != 2 * static_cast<size_t>(num_bin)) {
    Log::Warning("decompressed_len = %d, 2 * num_bin = %d", decompressed_len, 2 * num_bin);
  }
  CHECK_EQ(decompressed_len, 2 * static_cast<size_t>(num_bin));
}

template void HistogramCompressorV2::Compress<int32_t, uint32_t>(const int32_t* in_buffer, uint32_t* int_buffer, uint8_t* out_buffer, data_size_t num_bin);

template void HistogramCompressorV2::Compress<int16_t, uint16_t>(const int16_t* in_buffer, uint32_t* int_buffer, uint8_t* out_buffer, data_size_t num_bin);

template void HistogramCompressorV2::Decompress<int32_t, uint32_t>(const uint8_t* in_buffer, data_size_t num_bin, uint32_t* int_buffer, int32_t* out_buffer);

template void HistogramCompressorV2::Decompress<int16_t, uint16_t>(const uint8_t* in_buffer, data_size_t num_bin, uint32_t* int_buffer, int16_t* out_buffer);

void HistogramCompressorV2::Test() {
  const size_t test_len = 500000;
  std::vector<int16_t> int16_test_array(test_len * 2);
  std::mt19937 rand_eng(0);
  const int16_t num_range = 100;
  std::vector<double> start_prob(num_range, 1.0f / num_range);
  std::discrete_distribution<int16_t> dist_start(start_prob.begin(), start_prob.end());
  const int16_t diff_range = 100;
  std::vector<double> diff_prob(diff_range, 1.0f / diff_range);
  std::discrete_distribution<int16_t> dist_diff(diff_prob.begin(), diff_prob.end());
  std::vector<uint8_t> out_buffer(test_len * 2 * 20, 0);
  std::vector<uint32_t> int_buffer(test_len * 2 * 20, 0);
  std::vector<int16_t> result(test_len * 2, 0);
    int16_t grad = dist_start(rand_eng);
    int16_t hess = std::abs(dist_start(rand_eng));
    for (size_t i = 0; i < test_len; ++i) {
      int16_t grad_diff = dist_diff(rand_eng) - 50;
      int16_t hess_diff = dist_diff(rand_eng) - 50;
      int16_test_array[(i << 1) + 1] = grad;
      int16_test_array[(i << 1)] = hess;
      grad = grad + grad_diff;
      hess = std::abs(hess + hess_diff);
    }
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 6000; ++i) {
    Compress<int16_t, uint16_t>(int16_test_array.data(), int_buffer.data(), out_buffer.data(), test_len);
    global_timer.Start("Decompress");
    Decompress<int16_t, uint16_t>(out_buffer.data(), static_cast<data_size_t>(test_len), int_buffer.data(), result.data());
    global_timer.Stop("Decompress");
  }
    auto end = std::chrono::steady_clock::now();
    Log::Warning("compression and depression finished in %.10f seconds", static_cast<std::chrono::duration<double>>(end - start).count());
    const size_t total_bytes = static_cast<size_t>(*(reinterpret_cast<const uint32_t*>(out_buffer.data())));
    Log::Warning("compressed bytes = %ld", total_bytes);
    Log::Warning("compress ratio = %f", static_cast<double>(total_bytes) * sizeof(uint32_t) / (2 * test_len * sizeof(int16_t)));
    for (size_t i = 0; i < test_len; ++i) {
      if (int16_test_array[(i << 1) + 1] != result[(i << 1) + 1] ||
        int16_test_array[(i << 1)] != result[(i << 1)]) {
        if (i > 1) {
          Log::Warning("i = %d, grad = %d, hess = %d, grad hat = %d, hess hat = %d",
          i - 2, int16_test_array[((i - 2) << 1) + 1], int16_test_array[((i - 2) << 1)],
            result[((i - 2) << 1) + 1], result[((i - 2) << 1)]);
        }
        if (i > 0) {
          Log::Warning("i = %d, grad = %d, hess = %d, grad hat = %d, hess hat = %d",
          i - 1, int16_test_array[((i - 1) << 1) + 1], int16_test_array[((i - 1) << 1)],
            result[((i - 1) << 1) + 1], result[((i - 1) << 1)]);
        }
        Log::Warning("i = %d, grad = %d, hess = %d, grad hat = %d, hess hat = %d",
          i, int16_test_array[(i << 1) + 1], int16_test_array[(i << 1)],
            result[(i << 1) + 1], result[(i << 1)]);
      }
      CHECK_EQ(int16_test_array[(i << 1) + 1], result[(i << 1) + 1]);
      CHECK_EQ(static_cast<uint16_t>(int16_test_array[(i << 1)]), static_cast<uint16_t>(result[(i << 1)]));
    }
  global_timer.Print();
}

HistogramCompressorV3::HistogramCompressorV3(const int num_threads) {
  num_threads_ = num_threads;
  thread_end_bit_buffer_.resize(num_threads_, 0);
  thread_diff_encode_offset_.resize(num_threads + 1, 0);
}

template <typename S_HIST_T, typename U_HIST_T>
void HistogramCompressorV3::Compress(char* in_buffer, data_size_t num_bin) {
  global_timer.Start("HistogramCompressorV3::Compress");
  if (diff_buffer_.size() < static_cast<size_t>(2 * num_bin)) {
    diff_buffer_.resize(2 * num_bin, 0);
    encode_buffer_.resize(2 * num_bin, 0);
  }
  const size_t expected_bit_buffer_size = static_cast<size_t>((num_bin * 2 + 7) / 8 + num_threads_);
  if (bit_buffer_.size() < expected_bit_buffer_size) {
    bit_buffer_.resize(expected_bit_buffer_size, 0);
  }
  S_HIST_T* in_buffer_signed = reinterpret_cast<S_HIST_T*>(in_buffer);
  const U_HIST_T* in_buffer_unsigend = reinterpret_cast<const U_HIST_T*>(in_buffer);
  int block_size = (num_bin + num_threads_ - 1) / num_threads_;
  // block_size must be even to use simdfastpfor256
  block_size += (block_size % 2);
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t block_start = block_size * thread_index;
    const data_size_t block_end = std::min(block_start + block_size, num_bin);
    if (block_start >= block_end) {
      thread_diff_encode_offset_[thread_index + 1] = 0;
      continue;
    }
    int32_t prev_grad = 0;
    int32_t prev_hess = 0;
    const data_size_t remainder_bin = block_end % 4;
    for (data_size_t bin = block_start; bin < block_end - remainder_bin; ++bin) {
      const data_size_t bin_offset = (bin << 1);
      // assuming that grad and hess use most 31 bits of integer
      const int32_t grad = static_cast<int32_t>(in_buffer_signed[bin_offset + 1]);
      const int32_t hess = static_cast<int32_t>(in_buffer_unsigend[bin_offset]);
      const int32_t grad_diff = grad - prev_grad;
      const int32_t hess_diff = hess - prev_hess;
      prev_grad = grad;
      prev_hess = hess;
      const uint32_t grad_bit_pos = (bin_offset + 1) / 8;
      const uint8_t grad_bit_offset = (bin_offset + 1) % 8;
      const uint32_t hess_bit_pos = bin_offset / 8;
      const uint8_t hess_bit_offset = bin_offset % 8;
      if (hess_diff < 0) {
        if (hess_bit_offset == 0 || bin == block_start) {
          bit_buffer_[hess_bit_pos] = 1;
        } else {
          bit_buffer_[hess_bit_pos] |= (1 << hess_bit_offset);
        }
        diff_buffer_[bin_offset] = static_cast<uint32_t>(~hess_diff);
      } else {
        if (hess_bit_offset == 0 || bin == block_start) {
          bit_buffer_[hess_bit_pos] = 0;
        }
        diff_buffer_[bin_offset] = static_cast<uint32_t>(hess_diff);
      }
      if (grad_diff < 0) {
        bit_buffer_[grad_bit_pos] |= (1 << grad_bit_offset);
        diff_buffer_[bin_offset + 1] = static_cast<uint32_t>(~grad_diff);
      } else {
        diff_buffer_[bin_offset + 1] = static_cast<uint32_t>(grad_diff);
      }
    }
    uint8_t& thread_end_bit = thread_end_bit_buffer_[thread_index];
    for (data_size_t bin = block_end - remainder_bin; bin < block_end; ++bin) {
      const data_size_t bin_offset = (bin << 1);
      // assuming that grad and hess use most 31 bits of integer
      const int32_t grad = static_cast<int32_t>(in_buffer_signed[bin_offset + 1]);
      const int32_t hess = static_cast<int32_t>(in_buffer_unsigend[bin_offset]);
      const int32_t grad_diff = grad - prev_grad;
      const int32_t hess_diff = hess - prev_hess;
      prev_grad = grad;
      prev_hess = hess;
      const uint8_t grad_bit_offset = (bin_offset + 1) % 8;
      const uint8_t hess_bit_offset = bin_offset % 8;
      if (hess_diff < 0) {
        if (hess_bit_offset == 0) {
          thread_end_bit = 1;
        } else {
          thread_end_bit |= (1 << hess_bit_offset);
        }
        diff_buffer_[bin_offset] = static_cast<uint32_t>(~hess_diff);
      } else {
        if (hess_bit_offset == 0) {
          thread_end_bit = 0;
        }
        diff_buffer_[bin_offset] = static_cast<uint32_t>(hess_diff);
      }
      if (grad_diff < 0) {
        thread_end_bit |= (1 << grad_bit_offset);
        diff_buffer_[bin_offset + 1] = static_cast<uint32_t>(~grad_diff);
      } else {
        diff_buffer_[bin_offset + 1] = static_cast<uint32_t>(grad_diff);
      }
    }
    IntegerCODEC &codec = *CODECFactory::getFromName("simdfastpfor256", thread_index, 1);
    uint32_t* out_buffer = reinterpret_cast<uint32_t*>(encode_buffer_.data() + 2 * block_start);
    size_t out_buffer_len = static_cast<size_t>((block_end - block_start)) * sizeof(S_HIST_T) * 2 / sizeof(uint32_t);
    size_t nvalue = out_buffer_len;
    std::vector<uint32_t> decode_buffer(2 * (block_end - block_start));
    codec.encodeArray(
      diff_buffer_.data() + block_start * 2,
      static_cast<size_t>(2 * (block_end - block_start)),
      out_buffer,
      nvalue);
    CHECK_LE(nvalue, out_buffer_len);
    thread_diff_encode_offset_[thread_index + 1] = static_cast<uint32_t>(nvalue);
  }
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    thread_diff_encode_offset_[thread_index + 1] += thread_diff_encode_offset_[thread_index];

    // handle thread end bit
    const data_size_t block_start = block_size * thread_index;
    const data_size_t block_end = std::min(block_start + block_size, num_bin);
    const data_size_t remainder_bin = block_end % 4;
    if (remainder_bin > 0) {
      const data_size_t buffer_pos = ((block_end - remainder_bin) << 1) / 8;
      if (thread_index != num_threads_ - 1) {
        bit_buffer_[buffer_pos] |= thread_end_bit_buffer_[thread_index];
      } else {
        bit_buffer_[buffer_pos] = thread_end_bit_buffer_[thread_index];
      }
    }
  }
  uint32_t total_encoded_bytes =
    sizeof(uint32_t) + // length
    sizeof(uint32_t) + // num_threads_
    sizeof(uint32_t) * (num_threads_ + 1) + // thread_diff_encode_offset_
    thread_diff_encode_offset_.back() * sizeof(uint32_t) + // encoded diff length
    expected_bit_buffer_size * sizeof(uint8_t); // bits buffer
  CHECK_LE(total_encoded_bytes, 2 * num_bin * sizeof(S_HIST_T));
  // pack all encoded information togather in memory, copy to diff buffer first
  diff_buffer_[0] = static_cast<uint32_t>(num_threads_);
  for (int thread_index = 0; thread_index < num_threads_ + 1; ++thread_index) {
    diff_buffer_[thread_index + 1] = thread_diff_encode_offset_[thread_index];
  }

  uint32_t* packed_encode = reinterpret_cast<uint32_t*>(diff_buffer_.data() + num_threads_ + 2);
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t block_start = block_size * thread_index;
    const uint32_t* thread_encode = reinterpret_cast<const uint32_t*>(encode_buffer_.data() + 2 * block_start);
    uint32_t* thread_packed_encode = packed_encode + thread_diff_encode_offset_[thread_index];
    const uint32_t thread_encode_len = thread_diff_encode_offset_[thread_index + 1] - thread_diff_encode_offset_[thread_index];
    for (uint32_t i = 0; i < thread_encode_len; ++i) {
      thread_packed_encode[i] = thread_encode[i];
    }
  }

  uint32_t* in_buffer_as_out = reinterpret_cast<uint32_t*>(in_buffer);
  const uint32_t header_and_diff_encode_len = (num_threads_ + 2) + thread_diff_encode_offset_.back();
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (uint32_t i = 0; i < header_and_diff_encode_len; ++i) {
    in_buffer_as_out[i + 1] = diff_buffer_[i];
  }
  uint8_t* in_buffer_as_out_bits = reinterpret_cast<uint8_t*>(in_buffer_as_out + header_and_diff_encode_len + 1);
  const uint32_t bits_len = static_cast<uint32_t>((num_bin * 2 + 7) / 8);
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (uint32_t i = 0; i < bits_len; ++i) {
    in_buffer_as_out_bits[i] = bit_buffer_[i];
  }
  in_buffer_as_out[0] = bits_len * sizeof(uint8_t) + (header_and_diff_encode_len + 1) * sizeof(uint32_t);
  global_timer.Stop("HistogramCompressorV3::Compress");
}

template <typename S_HIST_T, typename U_HIST_T>
void HistogramCompressorV3::Decompress(char* in_buffer, data_size_t num_bin) {
  global_timer.Start("HistogramCompressorV3::Decompress");
  const uint32_t bits_len = static_cast<uint32_t>((num_bin * 2 + 7) / 8);
  if (diff_buffer_.size() < static_cast<size_t>(2 * num_bin)) {
    diff_buffer_.resize(2 * num_bin, 0);
    if (bit_buffer_.size() < bits_len) {
      bit_buffer_.resize(bits_len);
    }
  }
  // copy from encode
  const uint32_t* in_buffer_as_encode = reinterpret_cast<const uint32_t*>(in_buffer) + 1;
  num_threads_ = static_cast<int>(in_buffer_as_encode[0]);
  thread_diff_encode_offset_.resize(num_threads_ + 1, 0);
  for (int thread_index = 0; thread_index < num_threads_ + 1; ++thread_index) {
    thread_diff_encode_offset_[thread_index] = in_buffer_as_encode[thread_index + 1];
  }
  int block_size = (num_bin + num_threads_ - 1) / num_threads_;
  // block_size must be even to use simdfastpfor256
  block_size += (block_size % 2);
  in_buffer_as_encode += (2 + num_threads_);
  const uint8_t* in_buffer_as_encode_bits = reinterpret_cast<const uint8_t*>(in_buffer_as_encode + thread_diff_encode_offset_.back());
  if (thread_diff_encode_offset_.back() > encode_buffer_.size()) {
    encode_buffer_.resize(thread_diff_encode_offset_.back());
  }
  tmp_buffer_encode_.resize(num_threads_);
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const uint32_t thread_start = thread_diff_encode_offset_[thread_index];
    const uint32_t thread_end = thread_diff_encode_offset_[thread_index + 1];
    const uint32_t thread_size = thread_end - thread_start;
    if (thread_size > tmp_buffer_encode_[thread_index].size()) {
      tmp_buffer_encode_[thread_index].resize(thread_size);
    }
    for (uint32_t i = 0; i < thread_size; ++i) {
      tmp_buffer_encode_[thread_index][i] = in_buffer_as_encode[thread_start + i];
    }
  }
  // decompress diff
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t block_start = block_size * thread_index;
    const data_size_t block_end = std::min(block_start + block_size, num_bin);
    if (block_start >= block_end) {
      continue;
    }
    uint32_t* diff_buffer_ptr = diff_buffer_.data() + 2 * block_start;
    const uint32_t thread_encode_start = thread_diff_encode_offset_[thread_index];
    const uint32_t thread_encode_end = thread_diff_encode_offset_[thread_index + 1];
    size_t compressed_len = thread_encode_end - thread_encode_start;
    size_t decompressed_len = 2 * static_cast<size_t>(block_end - block_start);
    IntegerCODEC &codec = *CODECFactory::getFromName("simdfastpfor256", thread_index, 0);
    codec.decodeArray(tmp_buffer_encode_[thread_index].data(), compressed_len, diff_buffer_ptr, decompressed_len);
      CHECK_EQ(decompressed_len, 2 * static_cast<size_t>(block_end - block_start));
  }
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (uint32_t i = 0; i < bits_len; ++i) {
    bit_buffer_[i] = in_buffer_as_encode_bits[i];
  }

  //calculate original values
  S_HIST_T* in_buffer_signed = reinterpret_cast<S_HIST_T*>(in_buffer);
  U_HIST_T* in_buffer_unsigned = reinterpret_cast<U_HIST_T*>(in_buffer);
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t block_start = block_size * thread_index;
    const data_size_t block_end = std::min(block_start + block_size, num_bin);
    int32_t grad = 0;
    int32_t hess = 0;
    for (data_size_t bin = block_start; bin < block_end; ++bin) {
      const data_size_t bin_offset = (bin << 1);
      const uint32_t unsigned_grad_diff = diff_buffer_[bin_offset + 1];
      const uint32_t unsigned_hess_diff = diff_buffer_[bin_offset];
      const uint32_t grad_bit_pos = (bin_offset + 1) / 8;
      const uint32_t hess_bit_pos = (bin_offset) / 8;
      const uint8_t grad_bit_offset = (bin_offset + 1) % 8;
      const uint8_t hess_bit_offset = (bin_offset) % 8;
      const uint8_t grad_diff_signed_bit = ((bit_buffer_[grad_bit_pos] >> grad_bit_offset) & 0x01);
      const uint8_t hess_diff_signed_bit = ((bit_buffer_[hess_bit_pos] >> hess_bit_offset) & 0x01);
      if (grad_diff_signed_bit == 0) {
        grad += static_cast<int32_t>(unsigned_grad_diff);
      } else {
        grad += static_cast<int32_t>(~unsigned_grad_diff);
      }
      if (hess_diff_signed_bit == 0) {
        hess += static_cast<int32_t>(unsigned_hess_diff);
      } else {
        hess += static_cast<int32_t>(~unsigned_hess_diff);
      }
      in_buffer_unsigned[bin_offset] = static_cast<U_HIST_T>(hess);
      in_buffer_signed[bin_offset + 1] = static_cast<S_HIST_T>(grad);
    }
  }
  global_timer.Stop("HistogramCompressorV3::Decompress");
}

template void HistogramCompressorV3::Compress<int32_t, uint32_t>(char* in_buffer, data_size_t num_bin);

template void HistogramCompressorV3::Compress<int16_t, uint16_t>(char* in_buffer, data_size_t num_bin);

template void HistogramCompressorV3::Decompress<int32_t, uint32_t>(char* in_buffer, data_size_t num_bin);

template void HistogramCompressorV3::Decompress<int16_t, uint16_t>(char* in_buffer, data_size_t num_bin);

void HistogramCompressorV3::Test() {
  const size_t test_len = 500000;
  std::vector<int16_t> int16_test_array(test_len * 2);
  std::vector<int32_t> int16_test_array_copy_int32(test_len * 2);
  int16_t* int16_test_array_copy = reinterpret_cast<int16_t*>(int16_test_array_copy_int32.data());
  std::mt19937 rand_eng(0);
  const int16_t num_range = 100;
  std::vector<double> start_prob(num_range, 1.0f / num_range);
  std::discrete_distribution<int16_t> dist_start(start_prob.begin(), start_prob.end());
  const int16_t diff_range = 100;
  std::vector<double> diff_prob(diff_range, 1.0f / diff_range);
  std::discrete_distribution<int16_t> dist_diff(diff_prob.begin(), diff_prob.end());
  std::vector<int16_t> result(test_len * 2, 0);
  int16_t grad = dist_start(rand_eng);
  int16_t hess = std::abs(dist_start(rand_eng));
  for (size_t i = 0; i < test_len; ++i) {
    int16_t grad_diff = dist_diff(rand_eng) - 50;
    int16_t hess_diff = dist_diff(rand_eng) - 50;
    int16_test_array[(i << 1) + 1] = grad;
    int16_test_array[(i << 1)] = hess;
    int16_test_array_copy[(i << 1) + 1] = grad;
    int16_test_array_copy[(i << 1)] = hess;
    grad = grad + grad_diff;
    hess = std::abs(hess + hess_diff);
  }
  result = int16_test_array;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 6000; ++i) {
    Compress<int16_t, uint16_t>(reinterpret_cast<char*>(int16_test_array_copy), test_len);
    global_timer.Start("Decompress");
    Decompress<int16_t, uint16_t>(reinterpret_cast<char*>(int16_test_array_copy), test_len);
    global_timer.Stop("Decompress");
  }
    auto end = std::chrono::steady_clock::now();
    Log::Warning("compression and depression finished in %.10f seconds", static_cast<std::chrono::duration<double>>(end - start).count());
    for (size_t i = 0; i < test_len; ++i) {
      if (int16_test_array_copy[(i << 1) + 1] != result[(i << 1) + 1] ||
        int16_test_array_copy[(i << 1)] != result[(i << 1)]) {
        if (i > 1) {
          Log::Warning("i = %d, grad = %d, hess = %d, grad hat = %d, hess hat = %d",
          i - 2, int16_test_array_copy[((i - 2) << 1) + 1], int16_test_array_copy[((i - 2) << 1)],
            result[((i - 2) << 1) + 1], result[((i - 2) << 1)]);
        }
        if (i > 0) {
          Log::Warning("i = %d, grad = %d, hess = %d, grad hat = %d, hess hat = %d",
          i - 1, int16_test_array_copy[((i - 1) << 1) + 1], int16_test_array_copy[((i - 1) << 1)],
            result[((i - 1) << 1) + 1], result[((i - 1) << 1)]);
        }
        Log::Warning("i = %d, grad = %d, hess = %d, grad hat = %d, hess hat = %d",
          i, int16_test_array_copy[(i << 1) + 1], int16_test_array_copy[(i << 1)],
            result[(i << 1) + 1], result[(i << 1)]);
      }
      CHECK_EQ(int16_test_array_copy[(i << 1) + 1], result[(i << 1) + 1]);
      CHECK_EQ(static_cast<uint16_t>(int16_test_array_copy[(i << 1)]), static_cast<uint16_t>(result[(i << 1)]));
    }
  global_timer.Print();
}

}  // namespace LightGBM
