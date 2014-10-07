#ifndef EVAL_H_
#define EVAL_H_

#include <vector>

class DataMatrix;

class Metric {

public:

  virtual ~Metric() { };

  virtual double Evaluate(const std::vector<double>& predictions,
                          const DataMatrix& data) const = 0;

  virtual const char* Name() const = 0;

};

class Accuracy: public Metric {

public:

  virtual double Evaluate(const std::vector<double>& predictions,
                          const DataMatrix& data) const;

  virtual const char* Name() const {
    return "Accuracy";
  }
};

#endif
