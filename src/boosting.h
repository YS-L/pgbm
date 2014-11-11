#ifndef BOOSTING_H_
#define BOOSTING_H_

#include "tree.h"
#include "loss.h"
#include "eval.h"

#include <vector>
#include <memory>

class DataMatrix;

class Booster {

public:
  Booster(unsigned int n_iter, double shrinkage, unsigned int max_depth=6,
      unsigned int num_bins=50, unsigned int num_split_candidates=50,
      unsigned int eval_frequency=1);

  void Train(const DataMatrix& data);
  void Train(const DataMatrix& data, const DataMatrix& data_monitor);
  std::vector<double> Predict(const DataMatrix& data) const;
  void Describe();

private:

  void BoostSingleIteration(const DataMatrix& data);
  void UpdateCachedResponse(unsigned int model_index, const DataMatrix& data,
      std::vector<double>& response) const;

  unsigned int n_iter_;
  double shrinkage_;
  unsigned int max_depth_;
  unsigned int num_bins_;
  unsigned int num_split_candidates_;
  std::vector<Tree> models_;
  double base_response_;
  std::vector<double> cached_response_;
  std::vector<double> cached_response_monitor_;
  std::shared_ptr<Loss> loss_function_;
  std::shared_ptr<Metric> metric_;
  unsigned int eval_frequency_;
};

#endif
