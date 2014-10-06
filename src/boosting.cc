#include "boosting.h"
#include "util.h"

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
  PEEK_VECTOR(cached_response_, 20);

  for (unsigned int i = 0; i < n_iter_; ++i) {
    LOG(INFO) << "Boosting iteration " << i;
    std::vector<double> gradients;
    ComputeGradient(data, cached_response_, gradients);
    PEEK_VECTOR(gradients, 20)
    LOG(INFO) << "Training a tree";
    models_.push_back(Tree(6, 40, 80));
    models_.back().Train(data, gradients);
    for(unsigned int j = 0; j < cached_response_.size(); ++j) {
      cached_response_[j] += shrinkage_ *
        models_.back().Predict(data.GetRow(j));
    }
    PEEK_VECTOR(cached_response_, 0);
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

  std::vector<double> current_transformed_response;
  OutputTransform(current_response, current_transformed_response);
  for(unsigned int i = 0; i < data.Size(); ++i) {
    double grad = 2*y[i] / (1 + std::exp(2*y[i]*current_transformed_response[i]));
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
    double Fx = 0.5 * std::log(prob_positive / prob_negative);
    output[i] = Fx;
  }
};
