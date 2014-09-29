#include "tree.h"

#include <limits>
#include <utility>
#include <glog/logging.h>

Tree::Tree(unsigned int max_depth, unsigned int n_bins, unsigned int n_splits):
  max_depth_(max_depth), n_bins_(n_bins), n_splits_(n_splits) {
};

void Tree::Train(const DataMatrix& data) {
  Train(data, data.GetTargets());
};

void Tree::Train(const DataMatrix& data, const std::vector<double>& targets) {
  InitializeRootNode(data);
  while (!processing_queue_.empty()) {
    ProcessNode(data, targets, processing_queue_.front());
    processing_queue_.pop();
  }
};

std::vector<double> Tree::Predict(const DataMatrix& data) const {
  std::vector<double> predictions;
  predictions.reserve(data.Size());
  for (unsigned int i = 0; i < data.Size(); ++i) {
    predictions.push_back(Predict(data.GetRow(i)));
  }
  return predictions;
};

double Tree::Predict(const DataMatrix::SamplePoint& sample) const {
  return TraverseTree(sample);
};

void Tree::InitializeRootNode(const DataMatrix& data) {
  nodes_.clear();
  Node root;
  root.samples.reserve(data.Size());
  root.depth = 0;
  // NOTE: Even if the almost unncessary label is to be filled, it has to be
  // determined by the master later.
  for (unsigned int i = 0; i < data.Size(); ++i) {
    root.samples.push_back(i);
  }
  unsigned int root_id = AddNode(root);
  processing_queue_.push(root_id);
};

// Add a node and return it's unique id
unsigned int Tree::AddNode(Node& node) {
  node.id = nodes_.size();
  nodes_.push_back(node);
  return node.id;
};

void Tree::ProcessNode(const DataMatrix& data,
    const std::vector<double>& targets, unsigned int node_id) {

  Node& node = nodes_[node_id];

  if (node.depth >= max_depth_) {
    return;
  }

  // Find the best splits for current node among all the features
  SplitResult best_result;
  best_result.can_split = false;
  best_result.cost = std::numeric_limits<double>::max();
  unsigned int best_feature;
  std::vector<unsigned int> feature_keys = data.GetFeatureKeys();
  for (unsigned int i = 0; i < feature_keys.size(); ++i) {
    unsigned int fkey = feature_keys[i];
    const auto& column = data.GetColumn(fkey);
    auto histogram = ComputeHistogram(column, targets, node.samples);
    SplitResult result = FindBestSplit(histogram);
    if (best_result.can_split && result.cost < best_result.cost) {
      // TODO: Options to check for number of observations per leaf, etc. Or
      // check for these within FindBestSplit using the histogram?
      best_result = result;
      best_feature = i;
    }
  }

  if (!best_result.can_split) {
    return;
  }

  // TODO: Currently always split until reaching the required depth
  node.is_leaf = false;
  node.feature_index = best_feature;

  // Prepare the children nodes if can split
  Node next_left, next_right;
  next_left.depth = node.depth + 1;
  next_left.label = best_result.label_left;
  next_left.is_leaf = true;
  next_right.depth = node.depth + 1;
  next_right.label = best_result.label_right;
  next_right.is_leaf = true;

  auto features = data.GetColumn(node.feature_index);
  for(unsigned int i = 0; i < node.samples.size(); ++i) {
    unsigned int sample_idx = node.samples[i];
    if (features[sample_idx].value < node.threshold) {
      next_left.samples.push_back(sample_idx);
    }
    else {
      next_right.samples.push_back(sample_idx);
    }
  }

  // After split is decided and samples divided to the children, no longer need
  // to retain the sample indices
  node.samples.clear();

  unsigned int left_id = AddNode(next_left);
  unsigned int right_id = AddNode(next_right);

  // Link parent and childrens
  node.left_id = left_id;
  node.right_id = right_id;

  processing_queue_.push(left_id);
  processing_queue_.push(right_id);
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
  res.can_split = true;
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
