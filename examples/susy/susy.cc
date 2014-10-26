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
  //char filename_data_eval[1000];
  sprintf(filename_data_train, "../../Scripts/susy/susy.svmlight.train.4500k");

  int num_k_total_samples = 1000;
  int num_per_node = (int)((float)num_k_total_samples*1000 / world.size());
  int num_skips = world.rank() * num_per_node;

  LOG(INFO) << "Number of samples per node: " << num_per_node;
  LOG(INFO) << "Skips for this node: " << num_skips;

  DataMatrix data_train;
  data_train.Load(filename_data_train, num_skips, num_per_node);
  LOG(INFO) << "Training data size: " << data_train.Size()
            << " x "
            << data_train.Dimension();

  DataMatrix data_eval;
  data_eval.Load("../../Scripts/susy/susy.svmlight.eval.50k");
  LOG(INFO) << "Validation data size: " << data_eval.Size()
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
