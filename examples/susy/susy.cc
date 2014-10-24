#include "data.h"
#include "boosting.h"
#include "eval.h"
#include "mpi_util.h"

#include <cstdlib>
#include <string>
#include <glog/logging.h>

int main(int argc, char** argv) {

  LOG(INFO) << "Susy started";

  mpi::environment& env = MPIHandle::Get().env;
  mpi::communicator& world = MPIHandle::Get().world;

  char filename_data_train[1000];
  sprintf(filename_data_train,
      "../../Scripts/susy/susy.svmlight.train.50k.p%d", world.rank()+1);

  //char filename_data_eval[1000];

  DataMatrix data_train;
  data_train.Load(filename_data_train);
  LOG(INFO) << "Training data size: " << data_train.Size()
            << " x "
            << data_train.Dimension();

  DataMatrix data_eval;
  data_eval.Load("../../Scripts/susy/susy.svmlight.eval.50k");
  LOG(INFO) << "Training data size: " << data_eval.Size()
            << " x "
            << data_eval.Dimension();

  Booster booster(5, 0.05, 7, 80, 80, 1);
  booster.Train(data_train, data_eval);

  std::vector<double> predictions = booster.Predict(data_eval);
  Accuracy metric;
  double score = metric.Evaluate(predictions, data_eval);

  booster.Describe();
  printf("[Rank %d] Evaluation score: %.6f\n", world.rank(), score);

  return 0;
};
