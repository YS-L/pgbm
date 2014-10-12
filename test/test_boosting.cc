#include "gtest/gtest.h"
#include "data.h"
#include "boosting.h"
#include "eval.h"

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

  std::vector<double> preds = booster.Predict(data);
  Accuracy metric;
  double score = metric.Evaluate(preds, data);

  printf("Iris training accuracy: %.6f\n", score);

  ASSERT_NEAR(1.0, score, 1e-6);
};

TEST(BoosterTest, SUSY) {
  DataMatrix data_train;
  data_train.Load("bin/data/susy.svmlight.train.1k");
  ASSERT_EQ(1000, (int)data_train.Size());

  Booster booster(20, 0.05, 7, 80, 80, 1);
  booster.Train(data_train);

  DataMatrix data_eval;
  data_eval.Load("bin/data/susy.svmlight.eval.1k");
  ASSERT_EQ(1000, (int)data_eval.Size());
  std::vector<double> predictions = booster.Predict(data_eval);
  Accuracy metric;
  double score = metric.Evaluate(predictions, data_eval);

  printf("SUSY validation accuracy: %.6f\n", score);

  ASSERT_NEAR(0.754, score, 1e-6);
};
