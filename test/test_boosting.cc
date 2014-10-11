#include "gtest/gtest.h"
#include "data.h"
#include "boosting.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <glog/logging.h>

#define IRIS "bin/data/iris.svmlight"
#define BOSTON "bin/data/boston.svmlight"

TEST(BoosterTest, Iris) {
  DataMatrix data;
  data.Load(IRIS);
  ASSERT_EQ(150, (int)data.Size());

  Booster booster(50, 0.1, 8, 100, 100);
  booster.Train(data);

  LOG(INFO) << "Predicting...";

  std::vector<double> preds = booster.Predict(data);
  std::vector<double> targets = data.GetTargets();
  double hit = 0.0;
  for(unsigned int i = 0; i < preds.size(); ++i) {
    if (std::fabs(preds[i] - targets[i]) == 0) {
      hit += 1;
    }
    if (i > 120 || true) {
      printf("%f vs %f\n", targets[i], preds[i]);
    }
  }
  double acc = hit / preds.size();
  LOG(INFO) << "ACC: " << acc;
};
