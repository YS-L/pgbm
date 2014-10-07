#include "gtest/gtest.h"
#include "data.h"
#include "eval.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <glog/logging.h>

#define BOSTON "bin/data/boston.svmlight"

TEST(EvalTest, Accuracy) {
  DataMatrix data;
  data.Load(BOSTON);
  std::vector<double> predictions;
  Accuracy metric;

  predictions = data.GetTargets();
  ASSERT_NEAR(1.0, metric.evaluate(predictions, data), 1e-6);

  for(unsigned int i = 0; i < 10; ++i) {
    if (predictions[i] > 10e-6) {
      predictions[i] = 0.0;
    } else {
      predictions[i] = 1.0;
    }
  }
  double expected_accuracy = (predictions.size()-10.0)/predictions.size();
  ASSERT_NEAR(expected_accuracy, metric.evaluate(predictions, data), 1e-6);
};
