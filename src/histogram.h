#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include <vector>
#include <utility>

class Histogram {

public:

  Histogram(int num_bins);

  struct BinVal {
    double m;
    double y;
  };

  struct Bin {
    double p;
    BinVal val;

    bool operator<(const Bin& rhs) const {
      return p < rhs.p;
    }
  };

  void Update(double x, double y);
  void Merge(const Histogram& hist);
  BinVal Interpolate(double x) const;

  unsigned int get_num_bins() const {
    return bins_.size();
  };

  const std::vector<Bin> get_bins() const {
    return bins_;
  }

private:
  typedef std::vector<Bin> HistogramType;
  typedef HistogramType::iterator HistogramTypeIter;
  typedef HistogramType::const_iterator HistogramTypeConstIter;
  void Trim();

  int max_num_bins_;
  HistogramType bins_;

};

#endif
