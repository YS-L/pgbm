#ifndef BOOSTING_H_
#define BOOSTING_H_

#include "tree.h"
#include "loss.h"

#include <vector>
#include <memory>

class DataMatrix;

class Booster {

public:
  Booster(unsigned int n_iter, double shrinkage);

  void Train(const DataMatrix& data);
  std::vector<double> Predict(const DataMatrix& data) const;

private:

  void BoostSingleIteration(const DataMatrix& data);

  unsigned int n_iter_;
  double shrinkage_;
  std::vector<Tree> models_;
  std::vector<double> cached_response_;
  std::shared_ptr<Loss> loss_function_;

};

#endif
