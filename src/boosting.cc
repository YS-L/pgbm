#include "boosting.h"

#include <cmath>
#include <algorithm>
#include <glog/logging.h>

Booster::Booster(unsigned int n_iter, double shrinkage):
  n_iter_(n_iter), shrinkage_(shrinkage) { };

void Booster::Train(const DataMatrix& data) {
  models_.clear();

  std::vector<double> y = data.GetTargets();
  for(unsigned int i = 0; i < y.size(); ++i) {
    if (std::fabs(y[i]) < 10e-5) {
      y[i] = -1;
    }
  }
  double sum = std::accumulate(y.begin(), y.end(), 0.0);
  double y_mean = sum / y.size();
  double base_response = 0.5 * std::log((1 + y_mean) / (1 - y_mean));

  cached_response_ = std::vector<double>(data.Size(), base_response);

  for (unsigned int i = 0; i < n_iter_; ++i) {
    std::vector<double> gradients;
    ComputeGradient(data, cached_response_, gradients);
    // TODO: Tree parameters
    models_.push_back(Tree());
    models_.back().Train(data, gradients);
    for(unsigned int j = 0; j < cached_response_.size(); ++j) {
      cached_response_[j] += shrinkage_ *
        models_.back().Predict(data.GetRow(i));
    }
  }
};

std::vector<double> Booster::Predict(const DataMatrix& data) const {
  std::vector<double> responses;
  responses.reserve(data.Size());
  for(unsigned int i = 0; i < data.Size(); ++i) {
    double val = 0;
    for(unsigned int i = 0; i < models_.size(); ++i) {
      val += shrinkage_ * models_[i].Predict(data.GetRow(i));
    }
    responses.push_back(val);
  };
  std::vector<double> predictions;
  OutputTransform(responses, predictions);
  return predictions;
};

// Computes the negative gradients
// TODO: To be taken care of by a separate object
void Booster::ComputeGradient(const DataMatrix& data,
    const std::vector<double> current_response,
    std::vector<double>& gradients) const {

  gradients.clear();
  gradients.reserve(data.Size());

  std::vector<double> y = data.GetTargets();
  for(unsigned int i = 0; i < y.size(); ++i) {
    if (std::fabs(y[i]) < 10e-5) {
      y[i] = -1;
    }
  }
  for(unsigned int i = 0; i < data.Size(); ++i) {
    double grad = 2*y[i] / (1 + std::log(2*y[i]*current_response[i]));
    gradients.push_back(grad);
  }
};

void Booster::OutputTransform(const std::vector<double>& response,
    std::vector<double>& output) const {
  output = std::vector<double>(response.size());
  std::vector<std::vector<double> > probs(2);
  for(unsigned int i = 0; i < response.size(); ++i) {
    double prob_positive = 1 / (1 + std::exp(-2*response[i]));
    double prob_negative = 1 / (1 + std::exp(2*response[i]));
    if (prob_positive > prob_negative) {
      output[i] = 1;
    }
    else {
      output[i] = 0;
    }
  }
};
