#include "eval.h"
#include "data.h"

#include <cmath>
#include <glog/logging.h>

double Accuracy::Evaluate(const std::vector<double>& predictions,
                          const DataMatrix& data) const {
  int n_matched = 0;
  const std::vector<double>& targets = data.GetTargets();
  CHECK(targets.size() == predictions.size()) <<
      "Length of predictions does not match with targets";
  for(unsigned int i = 0; i < predictions.size(); ++i) {
    n_matched += (int)(std::fabs(predictions[i] - targets[i]) < 10e-6);
  }
  double accuracy = (double)n_matched / targets.size();
  return accuracy;
};
