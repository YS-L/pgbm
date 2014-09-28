#include "tree.h"

#include <limits>
#include <utility>
#include <glog/logging.h>

Tree::Tree(unsigned int max_depth, unsigned int n_bins, unsigned int n_splits):
  max_depth_(max_depth), n_bins_(n_bins), n_splits_(n_splits) {
};

void Tree::Train(const DataMatrix& data) {
  Node root;
  root.id = 0;
  root.samples.reserve(data.Size());
  root.depth = 0;
  for (unsigned int i = 0; i < data.Size(); ++i) {
    root.samples.push_back(i);
  }
  processing_queue_.push(root);
  while (!processing_queue_.empty()) {
    ProcessNode(data, processing_queue_.front());
    processing_queue_.pop();
  }
};

void Tree::ProcessNode(const DataMatrix& data, const Node& node) {

  if (node.depth > max_depth_) {
    return;
  }

  // Find the best splits for current node among all the features
  SplitResult best_result;
  best_result.cost = std::numeric_limits<double>::max();
  unsigned int best_feature;
  std::vector<unsigned int> feature_keys = data.GetFeatureKeys();
  for (unsigned int i = 0; i < feature_keys.size(); ++i) {
    unsigned int fkey = feature_keys[i];
    const auto& column = data.GetColumn(fkey);
    const auto& targets = data.GetTargets();
    auto histogram = ComputeHistogram(column, targets, node.samples);
    SplitResult result = FindBestSplit(histogram);
    if (result.cost < best_result.cost) {
      // TODO: Options to check for number of observations per leaf, etc. Or
      // check for these within FindBestSplit using the histogram?
      best_result = result;
      best_feature = i;
    }
  }

  // Finalize current node and push next nodes into the queue
  Node current, next_left, next_right;

  current = node;
  current.id = nodes_.size();
  current.feature_index = best_feature;
  current.threshold = best_result.threshold;
  current.is_leaf = false;

  next_left.depth = current.depth + 1;
  next_left.label = best_result.label_left;
  next_left.is_leaf = true;
  next_right.depth = current.depth + 1;
  next_right.label = best_result.label_right;
  next_right.is_leaf = true;

  auto features = data.GetColumn(current.feature_index);
  for(unsigned int i = 0; i < current.samples.size(); ++i) {
    unsigned int sample_idx = current.samples[i];
    if (features[sample_idx].value < current.threshold) {
      next_left.samples.push_back(sample_idx);
    }
    else {
      next_right.samples.push_back(sample_idx);
    }
  }
  // After split is decided, no longer need to retain the sample indices
  current.samples.clear();
  nodes_.push_back(current);

  processing_queue_.push(next_left);
  processing_queue_.push(next_right);
};

Tree::SplitResult Tree::FindBestSplit(const Histogram& histogram) const {
  std::vector<double> candidates = histogram.Uniform(n_splits_);
  Histogram::BinVal val_inf = histogram.InterpolateInf();
  double best_cost = std::numeric_limits<double>::min();
  double best_threshold, best_y_left, best_y_right;
  for (unsigned int i = 0; i < candidates.size(); ++i) {
    double s = candidates[i];
    // TODO: Make sure that cache is in action
    Histogram::BinVal val_s = histogram.Interpolate(s);
    double l_s = val_s.y;
    double m_s = val_s.m;
    double l_inf = val_inf.y;
    double m_inf = val_inf.m;
    double y_left = l_s / m_s;
    double y_right = (l_inf - l_s) / (m_inf - m_s);
    double cost = -(l_s*l_s)/(m_s*m_s) - (l_inf-l_s)*(l_inf-l_s)/(m_inf-m_s);
    if (cost < best_cost) {
      best_cost = cost;
      best_threshold = s;
      best_y_left = y_left;
      best_y_right = y_right;
    }
  }
  SplitResult res;
  res.threshold = best_threshold;
  res.label_left = best_y_left;
  res.label_right = best_y_right;
  return res;
};

Histogram Tree::ComputeHistogram(
    const std::vector<DataMatrix::FeaturePoint>& column,
    const std::vector<double>& targets,
    const std::vector<unsigned int>& samples) const {
  Histogram histogram(n_bins_);
  for (unsigned int i = 0; i < samples.size(); ++i) {
    unsigned int sample_idx = samples[i];
    double fval = column[sample_idx].value;
    double tval = targets[sample_idx];
    histogram.Update(fval, tval);
  }
  return histogram;
};

double Tree::TraverseTree(const DataMatrix::SamplePoint& sample) const {
  unsigned int id = 0;
  while (true) {
    if (nodes_[id].is_leaf) {
      return nodes_[id].label;
    }
    unsigned int fidx = nodes_[id].feature_index;
    double thres = nodes_[id].threshold;
    if (sample.features.at(fidx) <= thres) {
      id = nodes_[id].left_id;
    }
    else {
      id = nodes_[id].right_id;
    }
  };
};
