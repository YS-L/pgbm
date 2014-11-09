#include "histogram.h"
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <glog/logging.h>

Histogram::Histogram(unsigned int num_bins):
  max_num_bins_(num_bins),
  bins_pod_(0),
  bins_pending_pod_(false),
  dirty_(true),
  cumsums_(num_bins+1) { };

Histogram::~Histogram() {
  if (bins_pod_ != 0) {
    //delete [] bins_pod_;
  }
};

void Histogram::Update(double x, double y) {
  SyncPodBins();
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
  dirty_ = true;
  CHECK(bins_.size() > 0) << "Bin size is 0 after Update";
};

void Histogram::Merge(const Histogram& other) {
  SyncPodBins();
  for (unsigned int i = 0; i < other.bins_.size(); ++i) {
    bins_.push_back(other.bins_[i]);
  }
  std::sort(bins_.begin(), bins_.end());
  Trim();
  dirty_ = true;
};

// Compute the additional offset to be added to the accumulated sum
double trapezoid_estimate(double s, double p1, double p2, double v1, double v2) {
  // Bottom base w.r.t trepezoid truncated at s
  // TODO: Remove max() if not handling boundary cases here
  double vs = v1 + (v2 - v1) * std::max((s - p1), 0.0) / (p2 - p1);
  double as = (v1 + vs) * std::max((s - p1), 0.0) / 2;
  double a2 = (v1 + v2) * (p2 - p1) / 2;
  double R = (v1 + v2) / 2;
  //LOG(INFO) << "as : a2: -->" << as << " vs " <<a2;
  double res = (as / a2) * R;
  return res;
};

Histogram::BinVal Histogram::Interpolate(double x) const {
  SyncPodBins();
  // TODO: assert the x in between p_min and p_max
  // TODO: For completeness sake, need to handle -ve and +ve infinity, as it2
  // below might hit the boundaries
  Bin bin;
  bin.p = x;
  HistogramTypeConstIter it2 = std::upper_bound(bins_.begin(), bins_.end(), bin);
  HistogramTypeConstIter it1 = it2 - 1;
  //LOG(INFO) << "it2 index: " << it2 - bins_.begin() << "; is end: " << (it2 == bins_.end())?1:0;
  //LOG(INFO) << "bins_ max: " << it2 - bins_.begin();
  double m_summed = 0.0;
  double y_summed = 0.0;
  PrecomputeCumsums();
  if (it2 != bins_.begin()) {
    unsigned int index_upto = it2 - bins_.begin() - 1;
    m_summed = cumsums_[index_upto].m;
    y_summed = cumsums_[index_upto].y;
  }

  double m_offset, y_offset;
  // Take care of two critical bin locations: first and last
  if (it2 == bins_.begin()) {
    //LOG(INFO) << "Handling special case of neg_inf!";
    double p_neg_inf = bins_[0].p - (bins_[1].p - bins_[0].p);
    //LOG(INFO) << "p_0: " << it2->p << " m_0: " << it2->val.m;
    if (x < p_neg_inf) {
      m_offset = 0;
      y_offset = 0;
    }
    else {
      m_offset = trapezoid_estimate(x, p_neg_inf, it2->p, 0.0, it2->val.m);
      y_offset = trapezoid_estimate(x, p_neg_inf, it2->p, 0.0, it2->val.y);
    }
    //LOG(INFO) << "p_neg_inf: " << p_neg_inf << " offsets: " << m_offset << " " << y_offset;
  }
  else if (it2 == bins_.end()) {
    //LOG(INFO) << "Handling special case of pos_inf!";
    double p_pos_inf = bins_[bins_.size()-1].p + (bins_[bins_.size()-1].p -
        bins_[bins_.size()-2].p);
    if (x > p_pos_inf) {
      m_offset = it2->val.m / 2;
      y_offset = it2->val.y / 2;
    }
    else{
      m_offset = trapezoid_estimate(x, it2->p, p_pos_inf, it1->val.m, 0.0);
      y_offset = trapezoid_estimate(x, it2->p, p_pos_inf, it1->val.y, 0.0);
    }
    //LOG(INFO) << "p_pos_inf: " << p_pos_inf << " offsets: " << m_offset << " " << y_offset;
  }
  else {
    m_offset = trapezoid_estimate(x, it1->p, it2->p, it1->val.m, it2->val.m);
    y_offset = trapezoid_estimate(x, it1->p, it2->p, it1->val.y, it2->val.y);
  }

  BinVal val;
  val.m = m_summed + m_offset;
  val.y = y_summed + y_offset;
  return val;
};

Histogram::BinVal Histogram::InterpolateInf() const {
  PrecomputeCumsums();
  //const BinVal& binval = cumsums_[bins_.size()];
  const BinVal& binval = cumsums_.back();
  return binval;
};

void Histogram::Trim() {
  SyncPodBins();
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
    bins_[i].val.y += bins_[j].val.y;
    bins_.erase(bins_.begin() + j);
  }
};

// Implements Algorithm 4 in Ben-Haim-10 paper
std::vector<double> Histogram::Uniform(int N) const {
  SyncPodBins();
  // N corresponds to tilde B in the paper
  CHECK(N >= 1) << "Histogram's Uniform routine requires N >= 1";
  //CHECK(bins_.size() >= 2) << "Histogrom does not have enough bins";
  std::vector<double> results;

  PrecomputeCumsums();
  // Last of cumsums_ adds up all bins
  //double sum_m = cumsums_[bins_.size()].m;
  double sum_m = cumsums_.back().m;

  BinVal binval;
  for (int j = 1; j <= N - 1; ++j) {
    double s = ((double)j / N) * sum_m;
    binval.m = s;
    // This is (i+1) -- the first element that is larger than s; and cumsums_
    // is not all sorted until saturated
    std::vector<BinVal>::iterator it = std::upper_bound(
        cumsums_.begin(), cumsums_.begin() + bins_.size() - 1, binval,
        [](const BinVal& a, const BinVal& b) -> bool { return a.m < b.m; });

    if (it == cumsums_.begin()) {
      continue;
    }

    unsigned int i = it - cumsums_.begin() - 1;
    double d = s - cumsums_[i].m;
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

void Histogram::PrecomputeCumsums() const {
  // Pre-compute the cummulative sums, where
  // - cumsums_[i] means summation up to 1/2 of the i-th bin
  // - Last element of cumsums_ sums up valus in all the bins

  // Only recompute when necessary
  if (!dirty_) {
    return;
  }

  //cumsums_.resize(bins_.size()+1);

  for (unsigned int i = 0; i < bins_.size(); ++i) {
    if (i == 0) {
      cumsums_[i].m = bins_[i].val.m / 2.0;
      cumsums_[i].y = bins_[i].val.y / 2.0;
    }
    else {
      cumsums_[i].m = cumsums_[i-1].m
                      + bins_[i-1].val.m / 2.0
                      + bins_[i].val.m / 2.0;

      cumsums_[i].y = cumsums_[i-1].y
                      + bins_[i-1].val.y / 2.0
                      + bins_[i].val.y / 2.0;
    }
  }

  // Sums at +ve infinity (i.e. sum up all)
  // Note: Here how the +ve infinity cumsum is accessed has to be standardize
  // -- it should be accessed via cumsums_.back() (instead of
  // cumsums_[bins_.size()]).
  /*
  unsigned int index_inf = bins_.size();
  cumsums_[index_inf].m = cumsums_[index_inf-1].m + bins_.back().val.m/2.0;
  cumsums_[index_inf].y = cumsums_[index_inf-1].y + bins_.back().val.y/2.0;
  */
  unsigned int index_inf = max_num_bins_;
  cumsums_[index_inf].m = cumsums_[bins_.size()-1].m + bins_.back().val.m/2.0;
  cumsums_[index_inf].y = cumsums_[bins_.size()-1].y + bins_.back().val.y/2.0;
  dirty_ = false;
};

void Histogram::SyncPodBins() const {
  if (bins_pending_pod_) {
    bins_pending_pod_ = false;
    dirty_ = true;
    bins_.clear();
    bins_.reserve(max_num_bins_+1); // +1 just in case, probably no need
    for (unsigned int i = 0; i < bins_pod_size_; ++i) {
      // or directly resize up there?
      bins_.push_back(bins_pod_[i]);
    }
  }
};
