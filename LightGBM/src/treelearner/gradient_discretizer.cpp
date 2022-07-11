/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "gradient_discretizer.hpp"
#include <LightGBM/network.h>

namespace LightGBM {

void GradientDiscretizer::Init(const data_size_t num_data) {
  discretized_gradients_and_hessians_vector_.resize(num_data * 2);
  gradient_random_values_.resize(num_data);
  hessian_random_values_.resize(num_data);
  random_values_use_start_eng_ = std::mt19937(random_seed_);
  random_values_use_start_dist_ = std::uniform_int_distribution<data_size_t>(0, num_data);

  const int num_threads = OMP_NUM_THREADS();
  int num_blocks = 0;
  data_size_t block_size = 0;
  Threading::BlockInfo<data_size_t>(num_data, 512, &num_blocks, &block_size);
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads)
  for (int thread_id = 0; thread_id < num_blocks; ++thread_id) {
    const data_size_t start = thread_id * block_size;
    const data_size_t end = std::min(start + block_size, num_data);
    std::mt19937 gradient_random_values_eng(random_seed_ + thread_id);
    std::uniform_real_distribution<double> gradient_random_values_dist(0.0f, 1.0f);
    std::mt19937 hessian_random_values_eng(random_seed_ + thread_id + num_threads);
    std::uniform_real_distribution<double> hessian_random_values_dist(0.0f, 1.0f);
    for (data_size_t i = start; i < end; ++i) {
      gradient_random_values_[i] = gradient_random_values_dist(gradient_random_values_eng);
      hessian_random_values_[i] = hessian_random_values_dist(hessian_random_values_eng);
    }
  }

  max_gradient_abs_ = 0.0f;
  max_hessian_abs_ = 0.0f;

  gradient_scale_ = 0.0f;
  hessian_scale_ = 0.0f;
  inverse_gradient_scale_ = 0.0f;
  inverse_hessian_scale_ = 0.0f;

  boundary_locked_ = false;
}

void GradientDiscretizer::DiscretizeGradients(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians) {
  if (!boundary_locked_) {
    double max_gradient = std::fabs(input_gradients[0]);
    double max_hessian = std::fabs(input_hessians[0]);
    int num_threads = OMP_NUM_THREADS();
    std::vector<double> thread_max_gradient(num_threads, max_gradient);
    std::vector<double> thread_max_hessian(num_threads, max_hessian);
    Threading::For<data_size_t>(0, num_data, 1024,
      [input_gradients, input_hessians, &thread_max_gradient, &thread_max_hessian]
      (int, data_size_t start, data_size_t end) {
        int thread_id = omp_get_thread_num();
        for (data_size_t i = start; i < end; ++i) {
          double fabs_grad = std::fabs(input_gradients[i]);
          double fabs_hess = std::fabs(input_hessians[i]);
          if (fabs_grad > thread_max_gradient[thread_id]) {
            thread_max_gradient[thread_id] = fabs_grad;
          }
          if (fabs_hess > thread_max_hessian[thread_id]) {
            thread_max_hessian[thread_id] = fabs_hess;
          }
        }});
    max_gradient = thread_max_gradient[0];
    max_hessian = thread_max_hessian[0];
    for (int thread_id = 1; thread_id < num_threads; ++thread_id) {
      if (max_gradient < thread_max_gradient[thread_id]) {
        max_gradient = thread_max_gradient[thread_id];
      }
      if (max_hessian < thread_max_hessian[thread_id]) {
        max_hessian = thread_max_hessian[thread_id];
      }
    }
    if (Network::num_machines() > 1) {
      max_gradient = Network::GlobalSyncUpByMax(max_gradient);
      max_hessian = Network::GlobalSyncUpByMax(max_hessian);
    }
    // TODO: fix this for more objectives
    if (max_gradient >= 0.99f && max_hessian >= 0.248f && can_lock_) {
      boundary_locked_ = true;
      max_gradient_abs_ = 1.0f;
      max_hessian_abs_ = 0.25f;
    } else {
      max_gradient_abs_ = max_gradient;
      max_hessian_abs_ = max_hessian;
    }
    gradient_scale_ = max_gradient_abs_ / static_cast<double>(grad_discretize_bins_ / 2);
    if (is_constant_hessian_) {
      hessian_scale_ = max_hessian_abs_;
    } else {
      hessian_scale_ = max_hessian_abs_ / static_cast<double>(2);
    }
    inverse_gradient_scale_ = 1.0f / gradient_scale_;
    inverse_hessian_scale_ = 1.0f / hessian_scale_;
  }

  const int num_threads = OMP_NUM_THREADS();
  const int random_values_use_start = random_values_use_start_dist_(random_values_use_start_eng_);
  int8_t* discretized_int8 = discretized_gradients_and_hessians_vector_.data();
  if (is_constant_hessian_) {
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (data_size_t i = 0; i < num_data; ++i) {
      const double gradient = input_gradients[i];
      const data_size_t random_value_pos = (i + random_values_use_start) % num_data;
      discretized_int8[2 * i + 1] = gradient >= 0.0f ?
        static_cast<int8_t>(gradient * inverse_gradient_scale_ + gradient_random_values_[random_value_pos]) :
        static_cast<int8_t>(gradient * inverse_gradient_scale_ - gradient_random_values_[random_value_pos]);
      discretized_int8[2 * i] = static_cast<int8_t>(1);
    }
  } else {
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (data_size_t i = 0; i < num_data; ++i) {
      const double gradient = input_gradients[i];
      const data_size_t random_value_pos = (i + random_values_use_start) % num_data;
      discretized_int8[2 * i + 1] = gradient >= 0.0f ?
        static_cast<int8_t>(gradient * inverse_gradient_scale_ + gradient_random_values_[random_value_pos]) :
        static_cast<int8_t>(gradient * inverse_gradient_scale_ - gradient_random_values_[random_value_pos]);
      discretized_int8[2 * i] = static_cast<int8_t>(input_hessians[i] * inverse_hessian_scale_ + hessian_random_values_[random_value_pos]);
    }
  }
}

}  // namespace LightGBM
