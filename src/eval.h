#ifndef EVAL_H_
#define EVAL_H_

#include <vector>

class DataMatrix;

class Metric {

public:

  virtual ~Metric() { };

  virtual double evaluate(const std::vector<double>& predictions,
                          const DataMatrix& data) const = 0;

};

class Accuracy {

public:

  virtual double evaluate(const std::vector<double>& predictions,
                          const DataMatrix& data) const;

};

#endif
