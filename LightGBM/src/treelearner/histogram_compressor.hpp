/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_HISTOGRAM_COMPRESSOR_HPP_
#define LIGHTGBM_TREELEARNER_HISTOGRAM_COMPRESSOR_HPP_

#include <LightGBM/meta.h>
#include <LightGBM/utils/threading.h>

namespace LightGBM {

class HistogramCompressor {
 public:
  HistogramCompressor(const int num_threads);

  template <typename S_HIST_T, typename U_HIST_T>
  void Compress(const S_HIST_T* in_buffer, uint8_t* out_buffer, data_size_t num_bin);

  template <typename S_HIST_T, typename U_HIST_T>
  void Decompress(const uint8_t* in_buffer, data_size_t num_bin, S_HIST_T* out_buffer);

  void Test();

 private:
  template <typename S_HIST_T, typename U_HIST_T>
  uint32_t ComputeThreadHalfBytes(const S_HIST_T* in_buffer, uint8_t* out_bits_buffer,
    uint8_t* thread_first_bits_buffer,
    data_size_t start_bin, data_size_t end_bin);

  template <typename S_HIST_T, typename U_HIST_T>
  void WriteThreadCompressedData(
    const S_HIST_T* in_buffer, const uint8_t* bits_buffer,
    uint8_t* out_buffer,
    uint8_t* thread_first_buffer,
    data_size_t start_bin,
    data_size_t end_bin,
    uint32_t thread_start_half_bytes);

  std::vector<uint8_t> thread_first_bits_;
  std::vector<uint8_t> thread_first_;
  std::vector<uint32_t> thread_total_half_bytes_offset_;
  int num_threads_;
};

class HistogramCompressorV2 {
 public:
  HistogramCompressorV2(const int num_threads);

  template <typename S_HIST_T, typename U_HIST_T>
  void Compress(const S_HIST_T* in_buffer, uint32_t* int_buffer, uint8_t* out_buffer, data_size_t num_bin);

  template <typename S_HIST_T, typename U_HIST_T>
  void Decompress(const uint8_t* in_buffer, data_size_t num_bin, uint32_t* int_buffer, S_HIST_T* out_buffer);

  void Test();

 private:
  int num_threads_;
};

class HistogramCompressorV3 {
 public:
  HistogramCompressorV3(const int num_threads);

  template <typename S_HIST_T, typename U_HIST_T>
  void Compress(char* in_buffer, data_size_t num_bin);

  template <typename S_HIST_T, typename U_HIST_T>
  void Decompress(char* in_buffer, data_size_t num_bin);

  void Test();

 private:
  int num_threads_;
  std::vector<uint32_t, Common::AlignmentAllocator<uint32_t, 32>> diff_buffer_;
  std::vector<uint8_t> bit_buffer_;
  std::vector<uint8_t> thread_end_bit_buffer_;
  std::vector<uint32_t> thread_diff_encode_offset_;
  std::vector<uint32_t> encode_buffer_;
  std::vector<std::vector<uint32_t>> tmp_buffer_encode_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_TREELEARNER_HISTOGRAM_COMPRESSOR_HPP_
