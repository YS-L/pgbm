#include "boosting.h"
#include "util.h"

#include <cmath>
#include <algorithm>
#include <glog/logging.h>

Booster::Booster(unsigned int n_iter, double shrinkage, unsigned int eval_frequency):
  n_iter_(n_iter), shrinkage_(shrinkage),
  loss_function_(new TwoClassLogisticRegression),
  metric_(new Accuracy),
  eval_frequency_(eval_frequency) {
};

void Booster::Train(const DataMatrix& data) {
  models_.clear();

  double base_response = loss_function_->Baseline(data.GetTargets());

  cached_response_ = std::vector<double>(data.Size(), base_response);
  //PEEK_VECTOR(cached_response_, 20);

  for (unsigned int i = 0; i < n_iter_; ++i) {
    //LOG(INFO) << "Boosting iteration " << i;
    std::vector<double> gradients;
    loss_function_->Gradient(data.GetTargets(), cached_response_, gradients);
    //PEEK_VECTOR(gradients, 20)
    LOG(INFO) << "Training a tree";
    models_.push_back(Tree(6, 40, 80));
    models_.back().Train(data, gradients);
    for(unsigned int j = 0; j < cached_response_.size(); ++j) {
      cached_response_[j] += shrinkage_ *
        models_.back().Predict(data.GetRow(j));
    }
    //PEEK_VECTOR(cached_response_, 0);
    if (i % eval_frequency_ == 0) {
      std::vector<double> predictions;
      loss_function_->Output(cached_response_, predictions);
      double score = metric_->Evaluate(predictions, data);
      printf("[%d] %s: %.6f\n", i*eval_frequency_, metric_->Name(), score);
    }
  }
};

std::vector<double> Booster::Predict(const DataMatrix& data) const {
  std::vector<double> responses;
  responses.reserve(data.Size());
  for(unsigned int i = 0; i < data.Size(); ++i) {
    double val = 0;
    for(unsigned int j = 0; j < models_.size(); ++j) {
      val += shrinkage_ * models_[j].Predict(data.GetRow(i));
    }
    responses.push_back(val);
  };
  std::vector<double> predictions;
  loss_function_->Output(responses, predictions);
  return predictions;
};
