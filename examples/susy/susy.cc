#include "data.h"
#include "boosting.h"
#include "eval.h"
#include "mpi_util.h"
#include "util.h"

#include <cstdlib>
#include <string>
#include <glog/logging.h>
#include <gflags/gflags.h>
// TODO: Move somewhere else?
#include <boost/mpi/timer.hpp>

//DEFINE_bool(benchmark_build_tree, false, "To benchmark time required for building a tree");
DEFINE_int32(num_trees, 5, "Number of trees to build, i.e. number of iterations in boosting");
DEFINE_double(shrinkage, 0.05, "Shrinkage parameter");
DEFINE_int32(num_samples, 500, "Number of samples to train on in terms of thousands");
DEFINE_int32(max_depth, 7, "Maximum level of tree depth");
DEFINE_int32(bin_size, 80, "Number of bins to use in histograms");
DEFINE_int32(num_split_candidates, 80, "Number of candidate interpolation points to consider for each split");
DEFINE_int32(monitor_frequency, 1, "Evaluate current model on the monitoring dataset every this number of iteration(s); do not monitor if 0");

int main(int argc, char** argv) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "Susy started";

  mpi::environment& env = MPIHandle::Get().env;
  mpi::communicator& world = MPIHandle::Get().world;

  char filename_data_train[1000];
  char filename_data_validate[1000];
  sprintf(filename_data_train, "../../Scripts/susy/susy.svmlight.train.4500k");
  sprintf(filename_data_validate, "../../Scripts/susy/susy.svmlight.eval.50k");

  int num_k_total_samples = FLAGS_num_samples;
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
  data_eval.Load(filename_data_validate);
  LOG(INFO) << "Validation data size: " << data_eval.Size()
            << " x "
            << data_eval.Dimension();

  world.barrier();
  boost::mpi::timer timer;

  Booster booster(
      FLAGS_num_trees,
      FLAGS_shrinkage,
      FLAGS_max_depth,
      FLAGS_bin_size,
      FLAGS_num_split_candidates,
      FLAGS_monitor_frequency
  );

  booster.Train(data_train, data_eval);

  world.barrier();

  if ( world.rank() == 0 ) {
    LOG_STATS("elapsed_time_training", timer.elapsed());
  }

  std::vector<double> predictions = booster.Predict(data_eval);
  Accuracy metric;
  double score = metric.Evaluate(predictions, data_eval);

  if ( world.rank() == 0 ) {
    booster.Describe();
    printf("[Rank %d] Validation score: %.6f\n", world.rank(), score);
    LOG_STATS("validation_score", score);
  }

  return 0;
};
