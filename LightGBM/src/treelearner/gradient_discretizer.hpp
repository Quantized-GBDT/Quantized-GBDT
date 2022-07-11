/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_TREE_LEARNER_GRADIENT_DISCRETIZER_HPP_
#define LIGHTGBM_TREE_LEARNER_GRADIENT_DISCRETIZER_HPP_

#include <LightGBM/bin.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/threading.h>
#include <random>

namespace LightGBM {

class GradientDiscretizer {
 public:
  GradientDiscretizer(int grad_discretize_bins, int num_trees, int random_seed, bool can_lock, bool is_constant_hessian) {
    grad_discretize_bins_ = grad_discretize_bins;
    iter_ = 0;
    num_trees_ = num_trees;
    random_seed_ = random_seed;
    can_lock_ = can_lock;
    is_constant_hessian_ = is_constant_hessian;
  }

  virtual void DiscretizeGradients(
    const data_size_t num_data,
    const score_t* input_gradients,
    const score_t* input_hessians);

  virtual const int32_t* discretized_gradients_and_hessians() const {
    return reinterpret_cast<const int32_t*>(discretized_gradients_and_hessians_vector_.data());
  }

  virtual const score_t* grad_scale() const {
    return &gradient_scale_;
  }

  virtual const score_t* hess_scale() const {
    return &hessian_scale_;
  }

  virtual void Init(const data_size_t num_data);

 protected:
  int grad_discretize_bins_;
  int iter_;
  int num_trees_;
  int random_seed_;

  std::vector<double> gradient_random_values_;
  std::vector<double> hessian_random_values_;
  std::mt19937 random_values_use_start_eng_;
  std::uniform_int_distribution<data_size_t> random_values_use_start_dist_;
  std::vector<int8_t> discretized_gradients_and_hessians_vector_;

  double max_gradient_abs_;
  double max_hessian_abs_;

  score_t gradient_scale_;
  score_t hessian_scale_;
  double inverse_gradient_scale_;
  double inverse_hessian_scale_;

  bool boundary_locked_;
  bool can_lock_;
  bool is_constant_hessian_;
};

}

#endif  // LIGHTGBM_TREE_LEARNER_GRADIENT_DISCRETIZER_HPP_
