/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_BOOSTING_CUDA_NCCL_GBDT_HPP_
#define LIGHTGBM_BOOSTING_CUDA_NCCL_GBDT_HPP_

#ifdef USE_CUDA

#include "../gbdt.h"
#include <LightGBM/objective_function.h>
#include <LightGBM/network.h>
#include "cuda_score_updater.hpp"
#include <pthread.h>

namespace LightGBM {

template <typename GBDT_T>
class NCCLGBDT: public GBDT_T {
 public:
  NCCLGBDT();

  ~NCCLGBDT();

  void Init(const Config* gbdt_config, const Dataset* train_data,
            const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override;

  void Boosting() override;

  void RefitTree(const std::vector<std::vector<int>>& /*tree_leaf_prediction*/) override {
    Log::Fatal("RefitTree is not supported for NCCLGBDT.");
  }

  bool TrainOneIter(const score_t* gradients, const score_t* hessians) override;

  const double* GetTrainingScore(int64_t* /*out_len*/) override {
    Log::Fatal("GetTrainingScore is not supported for NCCLGBDT.");
  }

  void ResetTrainingData(const Dataset* /*train_data*/, const ObjectiveFunction* /*objective_function*/,
                         const std::vector<const Metric*>& /*training_metrics*/) override {
    Log::Fatal("ResetTrainingData is not supported for NCCLGBDT.");
  }

  void ResetConfig(const Config* /*gbdt_config*/) override {
    Log::Fatal("ResetConfig is not supported for NCCLGBDT.");
  }

 private:
  struct BoostingThreadData {
    int gpu_index;
    ObjectiveFunction* gpu_objective_function;
    score_t* gradients;
    score_t* hessians;
    const double* score;

    BoostingThreadData() {
      gpu_index = 0;
      gpu_objective_function = nullptr;
    }
  };

  struct TrainTreeLearnerThreadData {
    int gpu_index;
    TreeLearner* gpu_tree_learner;
    const score_t* gradients;
    const score_t* hessians; 
    bool is_first_time;
    int class_id;
    data_size_t num_data_in_gpu;
    std::unique_ptr<Tree> tree;

    TrainTreeLearnerThreadData() {
      gpu_index = 0;
      gpu_tree_learner = nullptr;
      gradients = nullptr;
      hessians = nullptr;
      is_first_time = false;
      class_id = 0;
      num_data_in_gpu = 0;
      tree.reset(nullptr);
    }
  };

  struct UpdateScoreThreadData {
    int gpu_index;
    ScoreUpdater* gpu_score_updater;
    TreeLearner* gpu_tree_learner;
    Tree* tree;
    int cur_tree_id;

    UpdateScoreThreadData() {
      gpu_index = 0;
      gpu_score_updater = nullptr;
      gpu_tree_learner = nullptr;
      tree = nullptr;
      cur_tree_id = 0;
    }
  };

  static void* BoostingThread(void* thread_data);

  static void* TrainTreeLearnerThread(void* thread_data);

  static void* UpdateScoreThread(void* thread_data);

  void Bagging(int /*iter*/) override {
    Log::Fatal("Bagging is not supported for NCCLGBDT.");
  }

  void InitNCCL();

  double BoostFromAverage(int class_id, bool update_scorer) override;

  void UpdateScore(const std::vector<std::unique_ptr<Tree>>& tree, const int cur_tree_id);

  void UpdateScore(const Tree* /*tree*/, const int /*cur_tree_id*/) {
    Log::Fatal("UpdateScore is not supported for NCCLGBDT.");
  }

  void RollbackOneIter() override {
    Log::Fatal("RollbackOneIter is not supported for NCCLGBDT.");
  }

  std::vector<double> EvalOneMetric(const Metric* metric, const double* score, const data_size_t num_data) const override;

  void SetCUDADevice(int gpu_id) const {
    if (gpu_list_.empty()) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_id));
    } else {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_list_[gpu_id]));
    }
  }

  int GetCUDADevice(int gpu_id) const {
    if (gpu_list_.empty()) {
      return gpu_id;
    } else {
      return gpu_list_[gpu_id];
    }
  }

  int num_gpu_;
  int num_threads_;
  int master_gpu_device_id_;
  int master_gpu_index_;
  std::vector<int> gpu_list_;
  std::vector<std::unique_ptr<ObjectiveFunction>> per_gpu_objective_functions_;
  std::vector<std::unique_ptr<ScoreUpdater>> per_gpu_train_score_updater_;
  std::vector<std::unique_ptr<CUDAVector<score_t>>> per_gpu_gradients_;
  std::vector<std::unique_ptr<CUDAVector<score_t>>> per_gpu_hessians_;
  std::vector<std::unique_ptr<Dataset>> per_gpu_datasets_;
  std::vector<data_size_t> per_gpu_data_start_;
  std::vector<data_size_t> per_gpu_data_end_;
  std::vector<pthread_t> host_threads_;
  std::vector<BoostingThreadData> boosting_thread_data_;
  std::vector<TrainTreeLearnerThreadData> train_tree_learner_thread_data_;
  std::vector<UpdateScoreThreadData> update_score_thread_data_;
  std::vector<int> nccl_gpu_rank_;
  std::vector<ncclComm_t> nccl_communicators_;
  std::vector<std::unique_ptr<TreeLearner>> per_gpu_tree_learners_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_BOOSTING_CUDA_NCCL_GBDT_HPP_
