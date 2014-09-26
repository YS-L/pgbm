#include "histogram.h"
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <iostream>

Histogram::Histogram(int num_bins): max_num_bins_(num_bins) {
};

void Histogram::Update(double x, double y) {
  BinVal val;
  val.m = 1;
  val.y = y;
  Bin bin;
  bin.p = x;
  bin.val = val;
  HistogramTypeIter insert_location = std::lower_bound(bins_.begin(), bins_.end(), bin);
  if (insert_location != bins_.end() && insert_location->p == x) {
    insert_location->val.m += 1;
    insert_location->val.y += y;
  }
  else {
    bins_.insert(insert_location, bin);
    Trim();
  }
};

void Histogram::Merge(const Histogram& other) {
  for (unsigned int i = 0; i < other.bins_.size(); ++i) {
    bins_.push_back(other.bins_[i]);
  }
  std::sort(bins_.begin(), bins_.end());
  Trim();
};

// Compute the additional offset to be added to the accumulated sum
double trapezoid_estimate(double s, double p1, double p2, double v1, double v2) {
  double vs = v1 + (v2 - v1) * (s - p1) / (p2 - p1);
  double as = (v1 + vs) * (s - p1) / 2;
  double a2 = (v1 + v2) * (p2 - p1) / 2;
  double R = (v1 + v2) / 2;
  double res = (as / a2) * R;
  return res;
};

Histogram::BinVal Histogram::Interpolate(double x) const {
  // TODO: assert the x in between p_min and p_max
  // TODO: cache summation in frozen mode
  // TODO: For completeness sake, need to handle -ve and +ve infinity 
  Bin bin;
  bin.p = x;
  HistogramTypeConstIter it2 = std::upper_bound(bins_.begin(), bins_.end(), bin);
  HistogramTypeConstIter it1 = it2 - 1;
  double m_summed = 0.0;
  double y_summed = 0.0;
  for (HistogramTypeConstIter it = bins_.begin(); it != it2; ++it) {
    if (it != it1) {
      m_summed += it->val.m;
      y_summed += it->val.y;
    }
    else{
      m_summed += it->val.m / 2.0;
      y_summed += it->val.y / 2.0;
    }
  }
  double m_offset = trapezoid_estimate(x, it1->p, it2->p,
                                       it1->val.m, it2->val.m);
  double y_offset = trapezoid_estimate(x, it1->p, it2->p,
                                       it1->val.y, it2->val.y);
  BinVal val;
  val.m = m_summed + m_offset;
  val.y = y_summed + y_offset;
  return val;
};

void Histogram::Trim() {
  while (bins_.size() > max_num_bins_) {
    // TODO: Handle ties in minimum distance
    double min_dist = std::numeric_limits<double>::max();
    unsigned int combine_location = 0;
    for (unsigned int i = 0; i < bins_.size() - 1; ++i) {
      double dist = std::fabs(bins_[i].p - bins_[i+1].p);
      if (dist < min_dist) {
        min_dist = dist;
        combine_location = i;
      }
    }
    //std::cout << "combine_location: " << combine_location << std::endl;
    //std::cout << "min_dist: " << min_dist << std::endl;
    int i = combine_location;
    int j = combine_location + 1;
    bins_[i].p = ((bins_[i].p * bins_[i].val.m) +
                  (bins_[j].p * bins_[j].val.m)) /
                 (bins_[i].val.m + bins_[j].val.m);
    bins_[i].val.m += bins_[j].val.m;
    bins_[i].val.y += bins_[j].val.m;
    bins_.erase(bins_.begin() + j);
  }
};

// Implements Algorithm 4 in Ben-Haim-10 paper
std::vector<double> Histogram::Uniform(int N) const {
  // N corresponds to tilde B in the paper
  std::vector<double> results;
  double sum_m = 0.0;
  for (unsigned int i = 0; i < bins_.size(); ++i) {
    sum_m += bins_[i].val.m;
  }
  std::vector<double> cumsums(bins_.size(), 0.0);
  for (unsigned int i = 0; i < bins_.size(); ++i) {
    // Special case of Sum / Interpolate where the target x is always at the
    // center of a bin -- can be computed inplace directly
    if (i == 0) {
      cumsums[i] = bins_[i].val.m / 2.0;
    }
    else {
      cumsums[i] = cumsums[i-1]
                   + bins_[i-1].val.m / 2.0
                   + bins_[i].val.m / 2.0;
    }
  }
  for (unsigned int j = 1; j <= N - 1; ++j) {
    double s = ((double)j / N) * sum_m;
    // This is (i+1) -- the first element that is larger than s
    std::vector<double>::iterator it = std::upper_bound(cumsums.begin(),
                                                        cumsums.end(), s);
    unsigned int i = it - cumsums.begin() - 1;
    double d = s - cumsums[i];
    double a = bins_[i+1].val.m - bins_[i].val.m;
    double z;
    if (std::fabs(a) > 10e-8) {
      double b = 2.0 * bins_[i].val.m;
      double c = -2.0 * d;
      z = (- b + sqrt(b*b - 4*a*c)) / (2*a);
    }
    else {
      z = d / bins_[i].val.m;
    }
    double uj = bins_[i].p + (bins_[i+1].p - bins_[i].p)*z;
    results.push_back(uj);
  }
  return results;
};
