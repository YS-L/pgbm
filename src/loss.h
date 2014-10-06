#ifndef LOSS_H_
#define LOSS_H_

#include <vector>

class Loss {

public:

  virtual ~Loss() {};

  // Computes the negative gradients
  virtual void Gradient(const std::vector<double>& targets,
                        const std::vector<double>& current_response,
                        std::vector<double>& gradients) const = 0;

  virtual double Baseline(const std::vector<double>& targets) const = 0;

  virtual void Output(const std::vector<double>& responses,
                      std::vector<double>& transformed) const;

};

class TwoClassLogisticRegression: public Loss {

public:

  virtual void Gradient(const std::vector<double>& targets,
                        const std::vector<double>& current_response,
                        std::vector<double>& gradients) const;

  virtual double Baseline(const std::vector<double>& targets) const;

  virtual void Output(const std::vector<double>& responses,
                      std::vector<double>& transformed) const;

};

#endif
