#include "boosting.h"
#include "util.h"
#include "mpi_util.h"

#include <cmath>
#include <algorithm>
#include <glog/logging.h>

Booster::Booster(unsigned int n_iter, double shrinkage, unsigned int max_depth,
    unsigned int num_bins, unsigned int num_split_candidates,
    unsigned int eval_frequency):
  n_iter_(n_iter), shrinkage_(shrinkage),
  max_depth_(max_depth),
  num_bins_(num_bins),
  num_split_candidates_(num_split_candidates),
  loss_function_(new TwoClassLogisticRegression),
  metric_(new Accuracy),
  eval_frequency_(eval_frequency) { };

void Booster::Train(const DataMatrix& data) {
  Train(data, data);
};

void Booster::Train(const DataMatrix& data, const DataMatrix& data_monitor) {
  models_.clear();

  // TODO: What if there's a list of monitor datasets (needs to have a
  // DataPointer typedef then)? Or that data is itself data_monitor?
  double base_response = loss_function_->Baseline(data.GetTargets());
  double base_response_monitor = loss_function_->Baseline(data_monitor.GetTargets());

  //PEEK_VECTOR(cached_response_, 20);

  // For printing out rank information in monitor logging
  mpi::environment& env = MPIHandle::Get().env;
  mpi::communicator& world = MPIHandle::Get().world;

  for (unsigned int i = 0; i < n_iter_; ++i) {
    //LOG(INFO) << "Boosting iteration " << i;
    if (i == 0) {
      cached_response_ = std::vector<double>(data.Size(), base_response);
      cached_response_monitor_ = std::vector<double>(data_monitor.Size(),
          base_response_monitor);
    }
    else {
      std::vector<double> gradients;
      loss_function_->Gradient(data.GetTargets(), cached_response_, gradients);
      //PEEK_VECTOR(gradients, 20)
      LOG(INFO) << "Training a tree";
      models_.push_back(Tree(max_depth_, num_bins_, num_split_candidates_, i));
      models_.back().Train(data, gradients);

      // Discard the baseline response used to initiate the gradient
      // computation (turns out to give the correct / better output)
      // TODO: Refactor this
      if (i == 1) {
        cached_response_ = std::vector<double>(data.Size(), 0.0);
        cached_response_monitor_ = std::vector<double>(data_monitor.Size(),
            0.0);
      }

      UpdateCachedResponse(models_.size()-1, data, cached_response_);
      UpdateCachedResponse(models_.size()-1, data_monitor, cached_response_monitor_);
    }
    //PEEK_VECTOR(cached_response_, 20);
    if (eval_frequency_ != 0 && i % eval_frequency_ == 0) {
      std::vector<double> predictions;
      loss_function_->Output(cached_response_, predictions);
      double score = metric_->Evaluate(predictions, data);

      std::vector<double> predictions_monitor;
      loss_function_->Output(cached_response_monitor_, predictions_monitor);
      double score_monitor = metric_->Evaluate(predictions_monitor, data_monitor);

      printf("[%d] %s", i*eval_frequency_, metric_->Name());
      printf(" %.6f", score);
      printf(" %.6f", score_monitor);
      printf("\n");

      // rank iter metric_name train_metric monitor_metric
      printf("MONITOR %d %d %s %f %f\n",
          world.rank(),
          i*eval_frequency_,
          metric_->Name(),
          score, score_monitor);
    }
  }
};

// Update cached_response_ using the model specified by model_index
void Booster::UpdateCachedResponse(unsigned int model_index,
  const DataMatrix& data, std::vector<double>& response) const {

  CHECK(model_index < models_.size()) <<
      "The specified model_index is out of bound";

  CHECK(data.Size() == response.size()) <<
      "Data and cached response does not match in size";

  for (unsigned int j = 0; j < response.size(); ++j) {
    response[j] += shrinkage_ *
      models_[model_index].Predict(data.GetRow(j));
  }

};

std::vector<double> Booster::Predict(const DataMatrix& data) const {
  std::vector<double> responses(data.Size(), 0.0);
  for(unsigned int i = 0; i < models_.size(); ++i) {
    UpdateCachedResponse(i, data, responses);
  }
  std::vector<double> predictions;
  loss_function_->Output(responses, predictions);
  return predictions;
};

void Booster::Describe() {
  printf("Booster parameters:\n");
  printf("n_iter: %d\n", n_iter_);
  printf("shrinkage: %f\n", shrinkage_);
  printf("num_bins: %d\n", num_bins_);
  printf("num_split_candidates: %d\n", num_split_candidates_);
  printf("loss_function: %s\n", loss_function_->Name());
};
