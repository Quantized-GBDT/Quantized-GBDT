/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "cuda_multiclass_objective.hpp"

#ifdef USE_CUDA

namespace LightGBM {

CUDAMulticlassSoftmax::CUDAMulticlassSoftmax(const Config& config): MulticlassSoftmax(config) {}

CUDAMulticlassSoftmax::CUDAMulticlassSoftmax(const std::vector<std::string>& strs): MulticlassSoftmax(strs) {}

CUDAMulticlassSoftmax::~CUDAMulticlassSoftmax() {}

void CUDAMulticlassSoftmax::Init(const Metadata& metadata, data_size_t num_data) {
  MulticlassSoftmax::Init(metadata, num_data);
  cuda_label_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  AllocateCUDAMemory<double>(&cuda_boost_from_score_, num_class_, __FILE__, __LINE__);
  AllocateCUDAMemory<double>(&cuda_softmax_buffer_, static_cast<size_t>(num_data) * static_cast<size_t>(num_class_), __FILE__, __LINE__);
  SetCUDAMemory<double>(cuda_boost_from_score_, 0, num_class_, __FILE__, __LINE__);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDAMulticlassSoftmax::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  LaunchGetGradientsKernel(score, gradients, hessians);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDAMulticlassSoftmax::ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
  LaunchConvertOutputCUDAKernel(num_data, input, output);
}

CUDAMulticlassOVA::CUDAMulticlassOVA(const Config& config) {
  num_class_ = config.num_class;
  for (int i = 0; i < num_class_; ++i) {
    cuda_binary_loss_.emplace_back(new CUDABinaryLogloss(config, i));
  }
  sigmoid_ = config.sigmoid;
}

CUDAMulticlassOVA::CUDAMulticlassOVA(const std::vector<std::string>& strs): MulticlassOVA(strs) {}

CUDAMulticlassOVA::~CUDAMulticlassOVA() {}

void CUDAMulticlassOVA::Init(const Metadata& metadata, data_size_t num_data) {
  num_data_ = num_data;
  for (int i = 0; i < num_class_; ++i) {
    cuda_binary_loss_[i]->Init(metadata, num_data);
  }
}

void CUDAMulticlassOVA::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  for (int i = 0; i < num_class_; ++i) {
    int64_t offset = static_cast<int64_t>(num_data_) * i;
    cuda_binary_loss_[i]->GetGradients(score + offset, gradients + offset, hessians + offset);
  }
}

void CUDAMulticlassOVA::ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
  for (int i = 0; i < num_class_; ++i) {
    cuda_binary_loss_[i]->ConvertOutputCUDA(num_data, input + i * num_data, output + i * num_data);
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA