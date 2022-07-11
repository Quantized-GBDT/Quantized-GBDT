/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_single_gpu_tree_learner.hpp"

#include <LightGBM/cuda/cuda_tree.hpp>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/feature_group.h>
#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>

#include <memory>

namespace LightGBM {

CUDASingleGPUTreeLearner::CUDASingleGPUTreeLearner(const Config* config): SerialTreeLearner(config) {
  nccl_comm_ = nullptr;
  gpu_rank_ = -1;
  gpu_device_id_ = -1;
  nccl_thread_index_ = -1;
}

void CUDASingleGPUTreeLearner::SetNCCL(ncclComm_t* comm, int gpu_rank, int gpu_device_id, data_size_t global_num_data) {
  nccl_comm_ = comm;
  gpu_rank_ = gpu_rank;
  gpu_device_id_ = gpu_device_id;
  global_num_data_ = global_num_data;
  nccl_thread_index_ = gpu_device_id;
}

CUDASingleGPUTreeLearner::~CUDASingleGPUTreeLearner() {
  if (nccl_comm_ != nullptr) {
    CUDASUCCESS_OR_FATAL(cudaStreamDestroy(nccl_stream_));
  }
}

void CUDASingleGPUTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  num_threads_ = OMP_NUM_THREADS();
  // use the first gpu by default
  if (nccl_comm_ == nullptr) {
    gpu_device_id_ = config_->gpu_device_id >= 0 ? config_->gpu_device_id : 0;
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_device_id_));
  }

  cuda_smaller_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  if (nccl_comm_ != nullptr) {
    cuda_smaller_leaf_splits_->SetNCCL(nccl_comm_);
  }
  cuda_smaller_leaf_splits_->Init(config_->use_discretized_grad);
  cuda_larger_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  if (nccl_comm_ != nullptr) {
    cuda_larger_leaf_splits_->SetNCCL(nccl_comm_);
  }
  cuda_larger_leaf_splits_->Init(config_->use_discretized_grad);

  cuda_histogram_constructor_.reset(new CUDAHistogramConstructor(train_data_, config_->num_leaves, num_threads_,
    share_state_->feature_hist_offsets(),
    config_->min_data_in_leaf, config_->min_sum_hessian_in_leaf, gpu_device_id_, config_->gpu_use_dp, config_->use_discretized_grad));
  if (nccl_comm_ != nullptr) {
    cuda_histogram_constructor_->SetNCCL(nccl_thread_index_);
  }
  cuda_histogram_constructor_->Init(train_data_, share_state_.get());

  const auto& feature_hist_offsets = share_state_->feature_hist_offsets();
  num_total_bin_ = feature_hist_offsets.empty() ? 0 : static_cast<int>(feature_hist_offsets.back());
  cuda_data_partition_.reset(new CUDADataPartition(
    train_data_, num_total_bin_, config_->num_leaves, num_threads_,
    config_->use_discretized_grad,
    cuda_histogram_constructor_->cuda_hist_pointer()));
  if (nccl_comm_ != nullptr) {
    cuda_data_partition_->SetNCCL(nccl_thread_index_, nccl_comm_);
  }
  cuda_data_partition_->Init();

  cuda_best_split_finder_.reset(new CUDABestSplitFinder(cuda_histogram_constructor_->cuda_hist(),
    train_data_, this->share_state_->feature_hist_offsets(), config_));
  if (nccl_comm_ != nullptr) {
    cuda_best_split_finder_->SetNCCL(nccl_thread_index_);
  }
  cuda_best_split_finder_->Init();

  leaf_best_split_feature_.resize(config_->num_leaves, -1);
  leaf_best_split_threshold_.resize(config_->num_leaves, 0);
  leaf_best_split_default_left_.resize(config_->num_leaves, 0);
  leaf_best_split_gain_.resize(config_->num_leaves, kMinScore);
  leaf_num_data_.resize(config_->num_leaves, 0);
  leaf_data_start_.resize(config_->num_leaves, 0);
  leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);

  AllocateBitset();

  num_cat_threshold_ = 0;

  if (config_->use_discretized_grad) {
    cuda_leaf_gradient_stat_buffer_.Resize(config_->num_leaves);
    cuda_leaf_hessian_stat_buffer_.Resize(config_->num_leaves);
    cuda_gradient_discretizer_.reset(new CUDAGradientDiscretizer(
      config_->grad_discretize_bins, config_->num_iterations, config_->seed, false, is_constant_hessian));
    cuda_gradient_discretizer_->SetNCCL(nccl_comm_);
    cuda_gradient_discretizer_->Init(num_data_);
  } else {
    cuda_gradient_discretizer_.reset(nullptr);
  }

  if (nccl_comm_ != nullptr) {
    leaf_to_hist_index_map_.resize(config_->num_leaves, -1);
    CUDASUCCESS_OR_FATAL(cudaStreamCreate(&nccl_stream_));
    global_num_data_in_leaf_.resize(config_->num_leaves, 0);
  }

  leaf_num_bits_in_histogram_bin_.resize(config_->num_leaves, 0);
  node_num_bits_in_histogram_bin_.resize(config_->num_leaves, 0);
  leaf_num_bits_in_histogram_acc_.resize(config_->num_leaves, 0);
  node_num_bits_in_histogram_acc_.resize(config_->num_leaves, 0);
}

void CUDASingleGPUTreeLearner::BeforeTrain() {
  const data_size_t root_num_data = cuda_data_partition_->root_num_data();
  const data_size_t* leaf_splits_init_indices =
    cuda_data_partition_->use_bagging() ? cuda_data_partition_->cuda_data_indices() : nullptr;
  cuda_data_partition_->BeforeTrain();
  if (config_->use_discretized_grad) {
    cuda_gradient_discretizer_->DiscretizeGradients(num_data_, gradients_, hessians_);
    cuda_histogram_constructor_->BeforeTrain(
      reinterpret_cast<const score_t*>(cuda_gradient_discretizer_->discretized_gradients_and_hessians()), nullptr);
    cuda_smaller_leaf_splits_->InitValues(
      config_->lambda_l1,
      config_->lambda_l2,
      reinterpret_cast<const int16_t*>(cuda_gradient_discretizer_->discretized_gradients_and_hessians()),
      leaf_splits_init_indices,
      cuda_data_partition_->cuda_data_indices(),
      root_num_data,
      cuda_histogram_constructor_->cuda_hist_pointer(),
      &leaf_sum_hessians_[0],
      cuda_gradient_discretizer_->grad_scale(),
      cuda_gradient_discretizer_->hess_scale());
  } else {
    cuda_histogram_constructor_->BeforeTrain(gradients_, hessians_);
    cuda_smaller_leaf_splits_->InitValues(
      config_->lambda_l1,
      config_->lambda_l2,
      gradients_,
      hessians_,
      leaf_splits_init_indices,
      cuda_data_partition_->cuda_data_indices(),
      root_num_data,
      cuda_histogram_constructor_->cuda_hist_pointer(),
      &leaf_sum_hessians_[0]);
  }
  leaf_num_data_[0] = root_num_data;
  cuda_larger_leaf_splits_->InitValues();
  col_sampler_.ResetByTree();
  cuda_best_split_finder_->BeforeTrain(col_sampler_.is_feature_used_bytree());
  leaf_data_start_[0] = 0;
  smaller_leaf_index_ = 0;
  larger_leaf_index_ = -1;

  if (nccl_comm_ != nullptr) {
    leaf_to_hist_index_map_[0] = 0;
    for (int leaf_index = 1; leaf_index < config_->num_leaves; ++leaf_index) {
      leaf_to_hist_index_map_[leaf_index] = -1;
    }
    global_num_data_in_leaf_[0] = global_num_data_;
  }

  leaf_num_bits_in_histogram_bin_.clear();
  node_num_bits_in_histogram_bin_.clear();
  leaf_num_bits_in_histogram_bin_.resize(config_->num_leaves, 0);
  node_num_bits_in_histogram_bin_.resize(config_->num_leaves, 0);
  leaf_num_bits_in_histogram_acc_.clear();
  node_num_bits_in_histogram_acc_.clear();
  leaf_num_bits_in_histogram_acc_.resize(config_->num_leaves, 0);
  node_num_bits_in_histogram_acc_.resize(config_->num_leaves, 0);
  SetNumBitsInHistogramBin(0, -1);
}

void CUDASingleGPUTreeLearner::AddPredictionToScore(const Tree* tree, double* out_score) const {
  cuda_data_partition_->UpdateTrainScore(tree, out_score);
}

Tree* CUDASingleGPUTreeLearner::Train(const score_t* gradients,
  const score_t* hessians, bool /*is_first_tree*/) {
  gradients_ = gradients;
  hessians_ = hessians;
  global_timer.Start("CUDASingleGPUTreeLearner::BeforeTrain", nccl_thread_index_);
  BeforeTrain();
  global_timer.Stop("CUDASingleGPUTreeLearner::BeforeTrain", nccl_thread_index_);
  const bool track_branch_features = !(config_->interaction_constraints_vector.empty());
  std::unique_ptr<CUDATree> tree(new CUDATree(config_->num_leaves, track_branch_features,
    config_->linear_tree, gpu_device_id_, has_categorical_feature_));
  for (int i = 0; i < config_->num_leaves - 1; ++i) {
    global_timer.Start("CUDASingleGPUTreeLearner::ConstructHistogramForLeaf", nccl_thread_index_);
    const data_size_t global_num_data_in_smaller_leaf = nccl_comm_ != nullptr ?
      global_num_data_in_leaf_[smaller_leaf_index_] :
      leaf_num_data_[smaller_leaf_index_];
    const data_size_t global_num_data_in_larger_leaf = nccl_comm_ != nullptr ?
      (larger_leaf_index_ < 0 ? 0 : global_num_data_in_leaf_[larger_leaf_index_]) :
      (larger_leaf_index_ < 0 ? 0 : leaf_num_data_[larger_leaf_index_]);
    const data_size_t num_data_in_smaller_leaf = leaf_num_data_[smaller_leaf_index_];
    const data_size_t num_data_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_num_data_[larger_leaf_index_];
    const double sum_hessians_in_smaller_leaf = leaf_sum_hessians_[smaller_leaf_index_];
    const double sum_hessians_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_sum_hessians_[larger_leaf_index_];
    cuda_histogram_constructor_->ConstructHistogramForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      global_num_data_in_smaller_leaf,
      global_num_data_in_larger_leaf,
      num_data_in_smaller_leaf,
      num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf,
      sum_hessians_in_larger_leaf,
      leaf_num_bits_in_histogram_bin_[smaller_leaf_index_]);
    global_timer.Stop("CUDASingleGPUTreeLearner::ConstructHistogramForLeaf", nccl_thread_index_);

    global_timer.Start("CUDASingleGPUTreeLearner::NCCLReduceHistogram", nccl_thread_index_);
    if (nccl_comm_ != nullptr) {
      NCCLReduceHistogram();
    }
    global_timer.Stop("CUDASingleGPUTreeLearner::NCCLReduceHistogram", nccl_thread_index_);

    uint8_t parent_num_bits_bin = 0;
    uint8_t smaller_num_bits_bin = 0;
    uint8_t larger_num_bits_bin = 0;
    if (larger_leaf_index_ != -1) {
      const int parent_leaf_index = std::min(smaller_leaf_index_, larger_leaf_index_);
      parent_num_bits_bin = node_num_bits_in_histogram_bin_[parent_leaf_index];
      smaller_num_bits_bin = leaf_num_bits_in_histogram_bin_[smaller_leaf_index_];
      larger_num_bits_bin = leaf_num_bits_in_histogram_bin_[larger_leaf_index_];
    } else {
      parent_num_bits_bin = leaf_num_bits_in_histogram_bin_[0];
      smaller_num_bits_bin = leaf_num_bits_in_histogram_bin_[0];
      larger_num_bits_bin = leaf_num_bits_in_histogram_bin_[0];
    }
    cuda_histogram_constructor_->SubtractHistogramForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      config_->use_discretized_grad,
      parent_num_bits_bin,
      smaller_num_bits_bin,
      larger_num_bits_bin);

    global_timer.Start("CUDASingleGPUTreeLearner::FindBestSplitsForLeaf", nccl_thread_index_);
    const uint8_t smaller_leaf_num_bits_bin = leaf_num_bits_in_histogram_bin_[smaller_leaf_index_];
    const uint8_t smaller_leaf_num_bits_acc = leaf_num_bits_in_histogram_acc_[smaller_leaf_index_];
    const uint8_t larger_leaf_num_bits_bin = larger_leaf_index_ < 0 ? 32 : leaf_num_bits_in_histogram_bin_[larger_leaf_index_];
    const uint8_t larger_leaf_num_bits_acc = larger_leaf_index_ < 0 ? 32 : leaf_num_bits_in_histogram_acc_[larger_leaf_index_];
    if (config_->use_discretized_grad) {
      cuda_best_split_finder_->FindBestSplitsForLeaf(
        cuda_smaller_leaf_splits_->GetCUDAStruct(),
        cuda_larger_leaf_splits_->GetCUDAStruct(),
        smaller_leaf_index_, larger_leaf_index_,
        global_num_data_in_smaller_leaf,
        global_num_data_in_larger_leaf,
        sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf,
        cuda_gradient_discretizer_->grad_scale(),
        cuda_gradient_discretizer_->hess_scale(),
        smaller_leaf_num_bits_bin,
        larger_leaf_num_bits_bin,
        smaller_leaf_num_bits_acc,
        larger_leaf_num_bits_acc);
    } else {
      cuda_best_split_finder_->FindBestSplitsForLeaf(
        cuda_smaller_leaf_splits_->GetCUDAStruct(),
        cuda_larger_leaf_splits_->GetCUDAStruct(),
        smaller_leaf_index_, larger_leaf_index_,
        global_num_data_in_smaller_leaf,
        global_num_data_in_larger_leaf,
        sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf,
        nullptr, nullptr,
        smaller_leaf_num_bits_bin,
        larger_leaf_num_bits_bin,
        smaller_leaf_num_bits_acc,
        larger_leaf_num_bits_acc);
    }
    global_timer.Stop("CUDASingleGPUTreeLearner::FindBestSplitsForLeaf", nccl_thread_index_);
    global_timer.Start("CUDASingleGPUTreeLearner::FindBestFromAllSplits", nccl_thread_index_);
    const CUDASplitInfo* best_split_info = nullptr;
    if (larger_leaf_index_ >= 0) {
      best_split_info = cuda_best_split_finder_->FindBestFromAllSplits(
        tree->num_leaves(),
        smaller_leaf_index_,
        larger_leaf_index_,
        &leaf_best_split_feature_[smaller_leaf_index_],
        &leaf_best_split_threshold_[smaller_leaf_index_],
        &leaf_best_split_default_left_[smaller_leaf_index_],
        &leaf_best_split_gain_[smaller_leaf_index_],
        &leaf_best_split_feature_[larger_leaf_index_],
        &leaf_best_split_threshold_[larger_leaf_index_],
        &leaf_best_split_default_left_[larger_leaf_index_],
        &leaf_best_split_gain_[larger_leaf_index_],
        &best_leaf_index_,
        &num_cat_threshold_);
    } else {
      best_split_info = cuda_best_split_finder_->FindBestFromAllSplits(
        tree->num_leaves(),
        smaller_leaf_index_,
        larger_leaf_index_,
        &leaf_best_split_feature_[smaller_leaf_index_],
        &leaf_best_split_threshold_[smaller_leaf_index_],
        &leaf_best_split_default_left_[smaller_leaf_index_],
        &leaf_best_split_gain_[smaller_leaf_index_],
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &best_leaf_index_,
        &num_cat_threshold_);
    }
    global_timer.Stop("CUDASingleGPUTreeLearner::FindBestFromAllSplits", nccl_thread_index_);
    global_timer.Start("CUDASingleGPUTreeLearner::CPUFindBestFromAllSplits", nccl_thread_index_);
    best_leaf_index_ = -1;
    double best_gain = kMinScore;
    for (int leaf_index = 0; leaf_index < tree->num_leaves(); ++leaf_index) {
      if (leaf_best_split_gain_[leaf_index] > best_gain) {
        best_gain = leaf_best_split_gain_[leaf_index];
        best_leaf_index_ = leaf_index;
      }
    }
    global_timer.Stop("CUDASingleGPUTreeLearner::CPUFindBestFromAllSplits", nccl_thread_index_);
    if (best_leaf_index_ == -1) {
      Log::Warning("No further splits with positive gain, training stopped with %d leaves.", (i + 1));
      break;
    }

    best_split_info += best_leaf_index_;
    global_timer.Start("CUDASingleGPUTreeLearner::Split", nccl_thread_index_);
    if (num_cat_threshold_ > 0) {
      ConstructBitsetForCategoricalSplit(best_split_info);
    }

    int right_leaf_index = 0;
    if (train_data_->FeatureBinMapper(leaf_best_split_feature_[best_leaf_index_])->bin_type() == BinType::CategoricalBin) {
      right_leaf_index = tree->SplitCategorical(best_leaf_index_,
                                       train_data_->RealFeatureIndex(leaf_best_split_feature_[best_leaf_index_]),
                                       train_data_->FeatureBinMapper(leaf_best_split_feature_[best_leaf_index_])->missing_type(),
                                       best_split_info,
                                       cuda_bitset_,
                                       cuda_bitset_len_,
                                       cuda_bitset_inner_,
                                       cuda_bitset_inner_len_);
    } else {
      right_leaf_index = tree->Split(best_leaf_index_,
                                       train_data_->RealFeatureIndex(leaf_best_split_feature_[best_leaf_index_]),
                                       train_data_->RealThreshold(leaf_best_split_feature_[best_leaf_index_],
                                        leaf_best_split_threshold_[best_leaf_index_]),
                                       train_data_->FeatureBinMapper(leaf_best_split_feature_[best_leaf_index_])->missing_type(),
                                       best_split_info);
    }

    leaf_best_split_gain_[best_leaf_index_] = kMinScore;
    leaf_best_split_gain_[right_leaf_index] = kMinScore;

    double sum_left_gradients = 0.0f;
    double sum_right_gradients = 0.0f;
    cuda_data_partition_->Split(best_split_info,
                                best_leaf_index_,
                                right_leaf_index,
                                leaf_best_split_feature_[best_leaf_index_],
                                leaf_best_split_threshold_[best_leaf_index_],
                                cuda_bitset_inner_,
                                static_cast<int>(cuda_bitset_inner_len_),
                                leaf_best_split_default_left_[best_leaf_index_],
                                leaf_num_data_[best_leaf_index_],
                                leaf_data_start_[best_leaf_index_],
                                cuda_smaller_leaf_splits_->GetCUDAStructRef(),
                                cuda_larger_leaf_splits_->GetCUDAStructRef(),
                                &leaf_num_data_[best_leaf_index_],
                                &leaf_num_data_[right_leaf_index],
                                &leaf_data_start_[best_leaf_index_],
                                &leaf_data_start_[right_leaf_index],
                                &leaf_sum_hessians_[best_leaf_index_],
                                &leaf_sum_hessians_[right_leaf_index],
                                &sum_left_gradients,
                                &sum_right_gradients,
                                global_num_data_in_leaf_.data() + best_leaf_index_,
                                global_num_data_in_leaf_.data() + right_leaf_index);
    //CheckSplitValid(leaf_best_split_feature_[best_leaf_index_], best_leaf_index_, right_leaf_index, sum_left_gradients, sum_right_gradients);
    if (nccl_comm_ != nullptr) {
      smaller_leaf_index_ = (global_num_data_in_leaf_[best_leaf_index_] < global_num_data_in_leaf_[right_leaf_index] ? best_leaf_index_ : right_leaf_index);
      larger_leaf_index_ = (smaller_leaf_index_ == best_leaf_index_ ? right_leaf_index : best_leaf_index_);
      const int best_leaf_hist_index = leaf_to_hist_index_map_[best_leaf_index_];
      leaf_to_hist_index_map_[smaller_leaf_index_] = right_leaf_index;
      leaf_to_hist_index_map_[larger_leaf_index_] = best_leaf_hist_index;
    } else {
      smaller_leaf_index_ = (leaf_num_data_[best_leaf_index_] < leaf_num_data_[right_leaf_index] ? best_leaf_index_ : right_leaf_index);
      larger_leaf_index_ = (smaller_leaf_index_ == best_leaf_index_ ? right_leaf_index : best_leaf_index_);
    }

    SetNumBitsInHistogramBin(best_leaf_index_, right_leaf_index);

    global_timer.Stop("CUDASingleGPUTreeLearner::Split", nccl_thread_index_);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  if (config_->use_discretized_grad && config_->discretized_grad_renew) {
    global_timer.Start("CUDASingleGPUTreeLearner::RenewDiscretizedTreeLeaves", nccl_thread_index_);
    RenewDiscretizedTreeLeaves(tree.get());
    global_timer.Stop("CUDASingleGPUTreeLearner::RenewDiscretizedTreeLeaves", nccl_thread_index_);
  }
  tree->ToHost();
  return tree.release();
}

void CUDASingleGPUTreeLearner::ResetTrainingData(
  const Dataset* train_data,
  bool is_constant_hessian) {
  SerialTreeLearner::ResetTrainingData(train_data, is_constant_hessian);
  CHECK_EQ(num_features_, train_data_->num_features());
  cuda_histogram_constructor_->ResetTrainingData(train_data, share_state_.get());
  cuda_data_partition_->ResetTrainingData(train_data,
    static_cast<int>(share_state_->feature_hist_offsets().back()),
    cuda_histogram_constructor_->cuda_hist_pointer());
  cuda_best_split_finder_->ResetTrainingData(
    cuda_histogram_constructor_->cuda_hist(),
    train_data,
    share_state_->feature_hist_offsets());
  cuda_smaller_leaf_splits_->Resize(num_data_);
  cuda_larger_leaf_splits_->Resize(num_data_);
  CHECK_EQ(is_constant_hessian, share_state_->is_constant_hessian);
}

void CUDASingleGPUTreeLearner::ResetConfig(const Config* config) {
  const int old_num_leaves = config_->num_leaves;
  SerialTreeLearner::ResetConfig(config);
  // TODO: be careful here when using with NCCLGBDT
  if (config_->gpu_device_id >= 0 && config_->gpu_device_id != gpu_device_id_) {
    Log::Fatal("Changing gpu device ID by resetting configuration parameter is not allowed for CUDA tree learner.");
  }
  num_threads_ = OMP_NUM_THREADS();
  if (config_->num_leaves != old_num_leaves) {
    leaf_best_split_feature_.resize(config_->num_leaves, -1);
    leaf_best_split_threshold_.resize(config_->num_leaves, 0);
    leaf_best_split_default_left_.resize(config_->num_leaves, 0);
    leaf_num_data_.resize(config_->num_leaves, 0);
    leaf_data_start_.resize(config_->num_leaves, 0);
    leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);
  }
  cuda_histogram_constructor_->ResetConfig(config);
  cuda_best_split_finder_->ResetConfig(config);
  cuda_data_partition_->ResetConfig(config);
}

void CUDASingleGPUTreeLearner::SetBaggingData(const Dataset* /*subset*/,
  const data_size_t* used_indices, data_size_t num_data) {
  cuda_data_partition_->SetUsedDataIndices(used_indices, num_data);
}

void CUDASingleGPUTreeLearner::RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                                         const double* /*score*/, data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const {
  CHECK(tree->is_cuda_tree());
  CUDATree* cuda_tree = reinterpret_cast<CUDATree*>(tree);
  if (obj != nullptr && obj->IsRenewTreeOutput()) {
    CHECK_LE(cuda_tree->num_leaves(), data_partition_->num_leaves());
    const data_size_t* bag_mapper = nullptr;
    if (total_num_data != num_data_) {
      CHECK_EQ(bag_cnt, num_data_);
      bag_mapper = bag_indices;
    }
    std::vector<int> n_nozeroworker_perleaf(tree->num_leaves(), 1);
    int num_machines = Network::num_machines();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      const double output = static_cast<double>(tree->LeafOutput(i));
      data_size_t cnt_leaf_data = leaf_num_data_[i];
      std::vector<data_size_t> index_mapper(cnt_leaf_data, -1);
      CopyFromCUDADeviceToHost<data_size_t>(index_mapper.data(),
        cuda_data_partition_->cuda_data_indices() + leaf_data_start_[i],
        static_cast<size_t>(cnt_leaf_data), __FILE__, __LINE__);
      if (cnt_leaf_data > 0) {
        const double new_output = obj->RenewTreeOutput(output, residual_getter, index_mapper.data(), bag_mapper, cnt_leaf_data);
        tree->SetLeafOutput(i, new_output);
      } else {
        CHECK_GT(num_machines, 1);
        tree->SetLeafOutput(i, 0.0);
        n_nozeroworker_perleaf[i] = 0;
      }
    }
    if (num_machines > 1) {
      std::vector<double> outputs(tree->num_leaves());
      for (int i = 0; i < tree->num_leaves(); ++i) {
        outputs[i] = static_cast<double>(tree->LeafOutput(i));
      }
      outputs = Network::GlobalSum(&outputs);
      n_nozeroworker_perleaf = Network::GlobalSum(&n_nozeroworker_perleaf);
      for (int i = 0; i < tree->num_leaves(); ++i) {
        tree->SetLeafOutput(i, outputs[i] / n_nozeroworker_perleaf[i]);
      }
    }
    cuda_tree->SyncLeafOutputFromHostToCUDA();
  }
}

Tree* CUDASingleGPUTreeLearner::FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const {
  std::unique_ptr<CUDATree> cuda_tree(new CUDATree(old_tree));
  cuda_leaf_gradient_stat_buffer_.SetValue(0);
  cuda_leaf_hessian_stat_buffer_.SetValue(0);
  ReduceLeafStat(cuda_tree.get(), gradients, hessians, cuda_data_partition_->cuda_data_indices());
  cuda_tree->SyncLeafOutputFromCUDAToHost();
  return cuda_tree.release();
}

Tree* CUDASingleGPUTreeLearner::FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                                                  const score_t* gradients, const score_t* hessians) const {
  cuda_data_partition_->ResetByLeafPred(leaf_pred, old_tree->num_leaves());
  refit_num_data_ = static_cast<data_size_t>(leaf_pred.size());
  data_size_t buffer_size = static_cast<data_size_t>(old_tree->num_leaves());
  if (old_tree->num_leaves() > 2048) {
    const int num_block = (refit_num_data_ + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
    buffer_size *= static_cast<data_size_t>(num_block + 1);
  }
  if (static_cast<size_t>(buffer_size) > cuda_leaf_gradient_stat_buffer_.Size()) {
    cuda_leaf_gradient_stat_buffer_.Resize(buffer_size);
    cuda_leaf_hessian_stat_buffer_.Resize(buffer_size);
  }
  return FitByExistingTree(old_tree, gradients, hessians);
}

void CUDASingleGPUTreeLearner::ReduceLeafStat(
  CUDATree* old_tree, const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf) const {
  LaunchReduceLeafStatKernel(gradients, hessians, num_data_in_leaf, old_tree->cuda_leaf_parent(),
    old_tree->cuda_left_child(), old_tree->cuda_right_child(),
    old_tree->num_leaves(), refit_num_data_, old_tree->cuda_leaf_value_ref(), old_tree->shrinkage());
}

void CUDASingleGPUTreeLearner::ConstructBitsetForCategoricalSplit(
  const CUDASplitInfo* best_split_info) {
  LaunchConstructBitsetForCategoricalSplitKernel(best_split_info);
}

void CUDASingleGPUTreeLearner::AllocateBitset() {
  has_categorical_feature_ = false;
  categorical_bin_offsets_.clear();
  categorical_bin_offsets_.push_back(0);
  categorical_bin_to_value_.clear();
  for (int i = 0; i < train_data_->num_features(); ++i) {
    const BinMapper* bin_mapper = train_data_->FeatureBinMapper(i);
    if (bin_mapper->bin_type() == BinType::CategoricalBin) {
      has_categorical_feature_ = true;
      break;
    }
  }
  if (has_categorical_feature_) {
    int max_cat_value = 0;
    int max_cat_num_bin = 0;
    for (int i = 0; i < train_data_->num_features(); ++i) {
      const BinMapper* bin_mapper = train_data_->FeatureBinMapper(i);
      if (bin_mapper->bin_type() == BinType::CategoricalBin) {
        max_cat_value = std::max(bin_mapper->MaxCatValue(), max_cat_value);
        max_cat_num_bin = std::max(bin_mapper->num_bin(), max_cat_num_bin);
      }
    }
    // std::max(..., 1UL) to avoid error in the case when there are NaN's in the categorical values 
    const size_t cuda_bitset_max_size = std::max(static_cast<size_t>((max_cat_value + 31) / 32), 1UL);
    const size_t cuda_bitset_inner_max_size = std::max(static_cast<size_t>((max_cat_num_bin + 31) / 32), 1UL);
    AllocateCUDAMemory<uint32_t>(&cuda_bitset_, cuda_bitset_max_size, __FILE__, __LINE__);
    AllocateCUDAMemory<uint32_t>(&cuda_bitset_inner_, cuda_bitset_inner_max_size, __FILE__, __LINE__);
    const int max_cat_in_split = std::min(config_->max_cat_threshold, max_cat_num_bin / 2);
    const int num_blocks = (max_cat_in_split + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
    AllocateCUDAMemory<size_t>(&cuda_block_bitset_len_buffer_, num_blocks, __FILE__, __LINE__);

    for (int i = 0; i < train_data_->num_features(); ++i) {
      const BinMapper* bin_mapper = train_data_->FeatureBinMapper(i);
      if (bin_mapper->bin_type() == BinType::CategoricalBin) {
        categorical_bin_offsets_.push_back(bin_mapper->num_bin());
      } else {
        categorical_bin_offsets_.push_back(0);
      }
    }
    for (size_t i = 1; i < categorical_bin_offsets_.size(); ++i) {
      categorical_bin_offsets_[i] += categorical_bin_offsets_[i - 1];
    }
    categorical_bin_to_value_.resize(categorical_bin_offsets_.back(), 0);
    for (int i = 0; i < train_data_->num_features(); ++i) {
      const BinMapper* bin_mapper = train_data_->FeatureBinMapper(i);
      if (bin_mapper->bin_type() == BinType::CategoricalBin) {
        const int offset = categorical_bin_offsets_[i];
        for (int bin = 0; bin < bin_mapper->num_bin(); ++bin) {
          categorical_bin_to_value_[offset + bin] = static_cast<int>(bin_mapper->BinToValue(bin));
        }
      }
    }
    InitCUDAMemoryFromHostMemory<int>(&cuda_categorical_bin_offsets_, categorical_bin_offsets_.data(), categorical_bin_offsets_.size(), __FILE__, __LINE__);
    InitCUDAMemoryFromHostMemory<int>(&cuda_categorical_bin_to_value_, categorical_bin_to_value_.data(), categorical_bin_to_value_.size(), __FILE__, __LINE__);
  } else {
    cuda_bitset_ = nullptr;
    cuda_bitset_inner_ = nullptr;
  }
  cuda_bitset_len_ = 0;
  cuda_bitset_inner_len_ = 0;
}

void CUDASingleGPUTreeLearner::CheckSplitValid(
  const int inner_split_feature,
  const int left_leaf,
  const int right_leaf,
  const double split_sum_left_gradients,
  const double split_sum_right_gradients) {
  std::vector<data_size_t> left_data_indices(leaf_num_data_[left_leaf]);
  std::vector<data_size_t> right_data_indices(leaf_num_data_[right_leaf]);
  CopyFromCUDADeviceToHost<data_size_t>(left_data_indices.data(),
    cuda_data_partition_->cuda_data_indices() + leaf_data_start_[left_leaf],
    leaf_num_data_[left_leaf], __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<data_size_t>(right_data_indices.data(),
    cuda_data_partition_->cuda_data_indices() + leaf_data_start_[right_leaf],
    leaf_num_data_[right_leaf], __FILE__, __LINE__);
  double sum_left_gradients = 0.0f, sum_left_hessians = 0.0f;
  double sum_right_gradients = 0.0f, sum_right_hessians = 0.0f;
  for (size_t i = 0; i < left_data_indices.size(); ++i) {
    const data_size_t index = left_data_indices[i];
    sum_left_gradients += gradients_[index];
    sum_left_hessians += hessians_[index];
  }
  for (size_t i = 0; i < right_data_indices.size(); ++i) {
    const data_size_t index = right_data_indices[i];
    sum_right_gradients += gradients_[index];
    sum_right_hessians += hessians_[index];
  }
  Log::Warning("inner_split_feature = %d", inner_split_feature);
  Log::Warning("sum_left_gradients = %f, split_sum_left_gradients = %f", sum_left_gradients, split_sum_left_gradients);
  Log::Warning("sum_left_hessians = %f, leaf_sum_hessians_[%d] = %f", sum_left_hessians, left_leaf, leaf_sum_hessians_[left_leaf]);
  Log::Warning("sum_right_gradients = %f, split_sum_right_gradients = %f", sum_right_gradients, split_sum_right_gradients);
  Log::Warning("sum_right_hessians = %f, leaf_sum_hessians_[%d] = %f", sum_right_hessians, right_leaf, leaf_sum_hessians_[right_leaf]);

  /*if (train_data_->FeatureBinMapper(inner_split_feature)->bin_type() == BinType::CategoricalBin) {
    std::vector<uint32_t> host_bitset_inner(cuda_bitset_inner_len_);
    CopyFromCUDADeviceToHost<uint32_t>(host_bitset_inner.data(), cuda_bitset_inner_, cuda_bitset_inner_len_, __FILE__, __LINE__);
    std::vector<data_size_t> host_left_data_indices(leaf_num_data_[left_leaf]);
    std::vector<data_size_t> host_right_data_indices(leaf_num_data_[right_leaf]);
    CopyFromCUDADeviceToHost<data_size_t>(host_left_data_indices.data(), cuda_data_partition_->cuda_data_indices() + leaf_data_start_[left_leaf],
      static_cast<size_t>(leaf_num_data_[left_leaf]), __FILE__, __LINE__);
    CopyFromCUDADeviceToHost<data_size_t>(host_right_data_indices.data(), cuda_data_partition_->cuda_data_indices() + leaf_data_start_[right_leaf],
      static_cast<size_t>(leaf_num_data_[right_leaf]), __FILE__, __LINE__);
    BinIterator* iter = train_data_->FeatureIterator(inner_split_feature);
    for (size_t i = 0; i < host_left_data_indices.size(); ++i) {
      const data_size_t data_index = host_left_data_indices[i];
      const uint32_t bin = iter->RawGet(data_index);
      const bool to_left = Common::FindInBitset(host_bitset_inner.data(), cuda_bitset_inner_len_, bin);
      if (!to_left) {
        Log::Warning("error !!! bin = %d found in left");
      }
    }
    for (size_t i = 0; i < host_right_data_indices.size(); ++i) {
      const data_size_t data_index = host_right_data_indices[i];
      const uint32_t bin = iter->RawGet(data_index);
      const bool to_right = (bin == 0 || !Common::FindInBitset(host_bitset_inner.data(), cuda_bitset_inner_len_, bin));
      if (!to_right) {
        Log::Warning("error !!! bin = %d found in right");
      }
    }

    // construct histogram manually
    std::vector<hist_t> hist(500, 0.0f);
    for (size_t i = 0; i < host_left_data_indices.size(); ++i) {
      const data_size_t data_index = host_left_data_indices[i];
      const uint32_t bin = iter->RawGet(data_index);
      const score_t gradient = gradients_[data_index];
      const score_t hessian = hessians_[data_index];
      hist[2 * bin] += gradient;
      hist[2 * bin + 1] += hessian;
    }
    for (size_t i = 0; i < host_right_data_indices.size(); ++i) {
      const data_size_t data_index = host_right_data_indices[i];
      const uint32_t bin = iter->RawGet(data_index);
      const score_t gradient = gradients_[data_index];
      const score_t hessian = hessians_[data_index];
      hist[2 * bin] += gradient;
      hist[2 * bin + 1] += hessian;
    }

    Log::Warning("==================== manual histogram for leaf %d (====================", left_leaf);
    for (size_t i = 0; i < 100; ++i) {
      Log::Warning("bin %d, grad %f, hess %f", i, hist[2 * i], hist[2 * i + 1]);
    }
  }*/

  CHECK_LE(std::fabs(sum_left_gradients - split_sum_left_gradients), 1e-6f);
  CHECK_LE(std::fabs(sum_left_hessians - leaf_sum_hessians_[left_leaf]), 1e-6f);
  CHECK_LE(std::fabs(sum_right_gradients - split_sum_right_gradients), 1e-6f);
  CHECK_LE(std::fabs(sum_right_hessians - leaf_sum_hessians_[right_leaf]), 1e-6f);
}

void CUDASingleGPUTreeLearner::RenewDiscretizedTreeLeaves(CUDATree* cuda_tree) {
  cuda_data_partition_->ReduceLeafGradStat(
    gradients_, hessians_, cuda_tree,
    cuda_leaf_gradient_stat_buffer_.RawData(),
    cuda_leaf_hessian_stat_buffer_.RawData());
  LaunchCalcLeafValuesGivenGradStat(cuda_tree, cuda_data_partition_->cuda_data_indices());
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDASingleGPUTreeLearner::NCCLReduceHistogram() {
  if (config_->use_discretized_grad) {
    hist_t* smaller_leaf_hist_pointer = cuda_histogram_constructor_->cuda_hist_pointer() +
      leaf_to_hist_index_map_[smaller_leaf_index_] * num_total_bin_;
    const uint8_t bit_size = leaf_num_bits_in_histogram_bin_[smaller_leaf_index_];
    if (bit_size == 32) {
      NCCLCHECK(ncclAllReduce(
        reinterpret_cast<const int64_t*>(smaller_leaf_hist_pointer),
        reinterpret_cast<int64_t*>(smaller_leaf_hist_pointer),
        static_cast<size_t>(num_total_bin_),
        ncclInt64,
        ncclSum,
        *nccl_comm_,
        nccl_stream_
      ));
    } else if (bit_size <= 16) {
      NCCLCHECK(ncclAllReduce(
        reinterpret_cast<const int32_t*>(smaller_leaf_hist_pointer),
        reinterpret_cast<int32_t*>(smaller_leaf_hist_pointer),
        static_cast<size_t>(num_total_bin_),
        ncclInt32,
        ncclSum,
        *nccl_comm_,
        nccl_stream_
      ));
    }
  } else {
    hist_t* smaller_leaf_hist_pointer = cuda_histogram_constructor_->cuda_hist_pointer() +
        leaf_to_hist_index_map_[smaller_leaf_index_] * num_total_bin_ * 2;
      NCCLCHECK(ncclAllReduce(
        smaller_leaf_hist_pointer,
        smaller_leaf_hist_pointer,
        static_cast<size_t>(num_total_bin_) * 2,
        ncclFloat64,
        ncclSum,
        *nccl_comm_,
        nccl_stream_));
  }
  CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(nccl_stream_));
}

/*void CUDASingleGPUTreeLearner::DumpHistogram(const int leaf_index) const {
  const hist_t* hist = cuda_histogram_constructor_->cuda_hist_pointer() +
        leaf_to_hist_index_map_[leaf_index] * num_total_bin_ * 2;
  std::vector<hist_t> host_histogram(2 * num_total_bin_, 0.0f);
  CopyFromCUDADeviceToHost<hist_t>(host_histogram.data(), hist, 2 * num_total_bin_, __FILE__, __LINE__);
  std::ofstream fout(std::string("histogram_") + std::to_string(gpu_device_id_) + std::string("_") + std::to_string(leaf_index));
  for (int bin = 0; bin < num_total_bin_; ++bin) {
    fout << bin << " " << "grad" << " " << host_histogram[2 * bin] << " hess " << host_histogram[2 * bin + 1] << std::endl;
  }
}*/

void CUDASingleGPUTreeLearner::SetNumBitsInHistogramBin(const int left_leaf_index, const int right_leaf_index) {
  if (right_leaf_index == -1) {
    if (!config_->use_discretized_grad) {
      leaf_num_bits_in_histogram_bin_[left_leaf_index] = 32;
      leaf_num_bits_in_histogram_acc_[left_leaf_index] = 32;
      return;
    }
    const data_size_t num_data_in_left_leaf = (nccl_comm_ == nullptr) ? leaf_num_data_[left_leaf_index] : global_num_data_in_leaf_[left_leaf_index];
    const uint64_t max_stat = static_cast<uint64_t>(num_data_in_left_leaf) * static_cast<uint64_t>(config_->grad_discretize_bins);
    const uint64_t max_stat_per_bin = static_cast<uint64_t>(num_data_in_left_leaf) * static_cast<uint64_t>(config_->grad_discretize_bins)
      / static_cast<uint64_t>(config_->per_bin_div);
    if (max_stat_per_bin < 256) {
      leaf_num_bits_in_histogram_bin_[left_leaf_index] = 8;
    } else if (max_stat_per_bin < 65536) {
      leaf_num_bits_in_histogram_bin_[left_leaf_index] = 16;
    } else {
      leaf_num_bits_in_histogram_bin_[left_leaf_index] = 32;
    }
    if (max_stat < 256) {
      leaf_num_bits_in_histogram_acc_[left_leaf_index] = 8;
    } else if (max_stat < 65536) {
      leaf_num_bits_in_histogram_acc_[left_leaf_index] = 16;
    } else {
      leaf_num_bits_in_histogram_acc_[left_leaf_index] = 32;
    }
  } else {
    if (!config_->use_discretized_grad) {
      node_num_bits_in_histogram_bin_[left_leaf_index] = 32;
      leaf_num_bits_in_histogram_bin_[left_leaf_index] = 32;
      leaf_num_bits_in_histogram_bin_[right_leaf_index] = 32;
      node_num_bits_in_histogram_acc_[left_leaf_index] = 32;
      leaf_num_bits_in_histogram_acc_[left_leaf_index] = 32;
      leaf_num_bits_in_histogram_acc_[right_leaf_index] = 32;
      return;
    }
    const data_size_t num_data_in_left_leaf = (nccl_comm_ == nullptr) ? leaf_num_data_[left_leaf_index] : global_num_data_in_leaf_[left_leaf_index];
    const data_size_t num_data_in_right_leaf = (nccl_comm_ == nullptr) ? leaf_num_data_[right_leaf_index] : global_num_data_in_leaf_[right_leaf_index];
    const uint64_t max_stat_left = static_cast<uint64_t>(num_data_in_left_leaf) * static_cast<uint64_t>(config_->grad_discretize_bins);
    const uint64_t max_stat_right = static_cast<uint64_t>(num_data_in_right_leaf) * static_cast<uint64_t>(config_->grad_discretize_bins);
    const uint64_t max_stat_left_per_bin = static_cast<uint64_t>(num_data_in_left_leaf) * static_cast<uint64_t>(config_->grad_discretize_bins) /
      static_cast<uint64_t>(config_->per_bin_div);
    const uint64_t max_stat_right_per_bin = static_cast<uint64_t>(num_data_in_right_leaf) * static_cast<uint64_t>(config_->grad_discretize_bins) /
      static_cast<uint64_t>(config_->per_bin_div);
    node_num_bits_in_histogram_bin_[left_leaf_index] = leaf_num_bits_in_histogram_bin_[left_leaf_index];
    node_num_bits_in_histogram_acc_[left_leaf_index] = leaf_num_bits_in_histogram_acc_[left_leaf_index];
    if (max_stat_left_per_bin < 256) {
      leaf_num_bits_in_histogram_bin_[left_leaf_index] = 8;
    } else if (max_stat_left_per_bin < 65536) {
      leaf_num_bits_in_histogram_bin_[left_leaf_index] = 16;
    } else {
      leaf_num_bits_in_histogram_bin_[left_leaf_index] = 32;
    }
    if (max_stat_right_per_bin < 256) {
      leaf_num_bits_in_histogram_bin_[right_leaf_index] = 8;
    } else if (max_stat_right_per_bin < 65536) {
      leaf_num_bits_in_histogram_bin_[right_leaf_index] = 16;
    } else {
      leaf_num_bits_in_histogram_bin_[right_leaf_index] = 32;
    }
    if (max_stat_left < 256) {
      leaf_num_bits_in_histogram_acc_[left_leaf_index] = 8;
    } else if (max_stat_left < 65536) {
      leaf_num_bits_in_histogram_acc_[left_leaf_index] = 16;
    } else {
      leaf_num_bits_in_histogram_acc_[left_leaf_index] = 32;
    }
    if (max_stat_right < 256) {
      leaf_num_bits_in_histogram_acc_[right_leaf_index] = 8;
    } else if (max_stat_right < 65536) {
      leaf_num_bits_in_histogram_acc_[right_leaf_index] = 16;
    } else {
      leaf_num_bits_in_histogram_acc_[right_leaf_index] = 32;
    }
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
