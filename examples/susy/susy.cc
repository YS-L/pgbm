#include "data.h"
#include "boosting.h"
#include "eval.h"

#include <glog/logging.h>

int main(int argc, char** argv) {

  LOG(INFO) << "Susy started";

  DataMatrix data_train;
  data_train.Load("../../Scripts/susy/susy.svmlight.train.5k");
  LOG(INFO) << "Training data size: " << data_train.Size()
            << " x "
            << data_train.Dimension();

  Booster booster(20, 0.05, 7, 80, 80, 1);
  booster.Train(data_train);

  DataMatrix data_eval;
  data_eval.Load("../../Scripts/susy/susy.svmlight.eval.5k");
  LOG(INFO) << "Training data size: " << data_eval.Size()
            << " x "
            << data_eval.Dimension();
  std::vector<double> predictions = booster.Predict(data_eval);
  Accuracy metric;
  double score = metric.Evaluate(predictions, data_eval);

  printf("Evaluation score: %.6f\n", score);

  return 0;
};
