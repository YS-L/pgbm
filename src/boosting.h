#ifndef BOOSTING_H_
#define BOOSTING_H_

#include "tree.h"

#include <vector>

class DataMatrix;

class Booster {

public:
  Booster(unsigned int n_iter, double shrinkage);

  void Train(const DataMatrix& data);
  std::vector<double> Predict(const DataMatrix& data) const;

private:

  void BoostSingleIteration(const DataMatrix& data);
  void ComputeGradient(const DataMatrix& data,
      const std::vector<double> current_response,
      std::vector<double>& gradients) const;
  void OutputTransform(const std::vector<double>& response,
      std::vector<double>& output) const;

  unsigned int n_iter_;
  double shrinkage_;
  std::vector<Tree> models_;
  std::vector<double> cached_response_;

};

#endif
