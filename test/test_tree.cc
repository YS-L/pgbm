#include "gtest/gtest.h"
#include "data.h"
#include "tree.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <glog/logging.h>

#define BOSTON "bin/data/boston.svmlight"

TEST(TreeTest, Boston) {
  DataMatrix data;
  data.Load(BOSTON);
  ASSERT_EQ(506, (int)data.Size());

  Tree tree(50, 80, 80);
  tree.Train(data);

  LOG(INFO) << "Predicting...";

  std::vector<double> preds = tree.Predict(data);
  std::vector<double> targets = data.GetTargets();
  double err = 0.0;
  for(unsigned int i = 0; i < preds.size(); ++i) {
    err += (preds[i]-targets[i])*(preds[i]-targets[i]) / preds.size();
    if (i < 40)
      printf("%f vs %f\n", targets[i], preds[i]);
  }
  err = std::sqrt(err);
  LOG(INFO) << "RMSE: " << err;
};
