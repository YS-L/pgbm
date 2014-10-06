#include "loss.h"

#include <algorithm>
#include <glog/logging.h>

void Loss::Output(
    const std::vector<double>& response,
    std::vector<double>& output) const {
  output = response;
};


void TwoClassLogisticRegression::Gradient(
    const std::vector<double>& targets,
    const std::vector<double>& current_response,
    std::vector<double>& gradients) const {

  gradients.clear();
  gradients.reserve(targets.size());

  for(unsigned int i = 0; i < targets.size(); ++i) {
    double y_i;
    if (std::fabs(targets[i]) < 10e-5) {
      y_i = -1;
    }
    else {
      y_i = 1;
    }
    double grad = 2*y_i / (1 + std::exp(2*y_i*current_response[i]));
    gradients.push_back(grad);
  }
};

double TwoClassLogisticRegression::Baseline(
    const std::vector<double>& targets) const {

  std::vector<double> y = targets;
  for(unsigned int i = 0; i < y.size(); ++i) {
    if (std::fabs(y[i]) < 10e-5) {
      y[i] = -1;
    }
  }
  double sum = std::accumulate(y.begin(), y.end(), 0.0);
  double y_mean = sum / y.size();
  double base_response = 0.5 * std::log((1 + y_mean) / (1 - y_mean));
  return base_response;
};

void TwoClassLogisticRegression::Output(
    const std::vector<double>& response,
    std::vector<double>& output) const {

  output = std::vector<double>(response.size());
  std::vector<std::vector<double> > probs(2);
  for(unsigned int i = 0; i < response.size(); ++i) {
    double prob_positive = 1 / (1 + std::exp(-2*response[i]));
    double prob_negative = 1 / (1 + std::exp(2*response[i]));
    // TODO: Need extra information on how the labels are stored
    output[i] = (prob_positive > prob_negative)?1.0:0.0;
  }
};

