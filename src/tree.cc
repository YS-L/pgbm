#include "tree.h"

#include <cstdlib>
#include <cmath>
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
  //LOG(INFO) << "Predicting for sample #" << i;
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
  //LOG(INFO) << "Added node with id " << node.id;
  return node.id;
};

void Tree::ProcessNode(const DataMatrix& data,
    const std::vector<double>& targets, unsigned int node_id) {

  //LOG(INFO) << "Processing node " << node_id << " (" << nodes_[node_id].id << ")";

  if (nodes_[node_id].depth >= max_depth_) {
    //LOG(INFO) << "Done by depth";
    return;
  }

  // Find the best splits for current node among all the features
  SplitResult best_result;
  best_result.can_split = false;
  best_result.cost = std::numeric_limits<double>::max();
  unsigned int best_feature;
  std::vector<unsigned int> feature_keys = data.GetFeatureKeys();
  for (unsigned int i = 0; i < feature_keys.size(); ++i) {
    //LOG(INFO) << "Computing histogram for feature " << i;
    unsigned int fkey = feature_keys[i];
    const auto& column = data.GetColumn(fkey);
    auto histogram = ComputeHistogram(column, targets, nodes_[node_id].samples);
    SplitResult result = FindBestSplit(histogram);
    if (result.can_split && result.cost < best_result.cost) {
      // TODO: Options to check for number of observations per leaf, etc. Or
      // check for these within FindBestSplit using the histogram?
      best_result = result;
      best_feature = i;
    }
  }

  if (!best_result.can_split) {
    return;
  }

  LOG(INFO) << "Node [" << node_id <<"]: Making a split on feature " << best_feature << " with threshold " << best_result.threshold << "; label_left: " << best_result.label_left << " label_right: " << best_result.label_right;

  // TODO: Currently always split until reaching the required depth
  nodes_[node_id].is_leaf = false;
  nodes_[node_id].feature_index = best_feature;
  nodes_[node_id].threshold = best_result.threshold;

  // Prepare the children nodes if can split
  // TODO: Uncessary copies here? Need to add first and update in place?
  Node next_left, next_right;
  next_left.depth = nodes_[node_id].depth + 1;
  next_left.label = best_result.label_left;
  next_left.is_leaf = true;
  next_right.depth = nodes_[node_id].depth + 1;
  next_right.label = best_result.label_right;
  next_right.is_leaf = true;

  auto features = data.GetColumn(nodes_[node_id].feature_index);
  for(unsigned int i = 0; i < nodes_[node_id].samples.size(); ++i) {
    unsigned int sample_idx = nodes_[node_id].samples[i];
    if (features[sample_idx].value < nodes_[node_id].threshold) {
      next_left.samples.push_back(sample_idx);
    }
    else {
      next_right.samples.push_back(sample_idx);
    }
  }

  // After split is decided and samples divided to the children, no longer need
  // to retain the sample indices
  nodes_[node_id].samples.clear();

  unsigned int left_id = AddNode(next_left);
  unsigned int right_id = AddNode(next_right);

  // Link parent and childrens
  nodes_[node_id].left_id = left_id;
  nodes_[node_id].right_id = right_id;
  //LOG(INFO) << "Node " << nodes_[node_id].id << "'s left: " << nodes_[node_id].left_id << "; right: " << nodes_[node_id].right_id;

  processing_queue_.push(left_id);
  processing_queue_.push(right_id);

  //LOG(INFO) << "Done processing node " << node_id;
};

Tree::SplitResult Tree::FindBestSplit(const Histogram& histogram) const {
  //LOG(INFO) << "Finding best split";
  std::vector<double> candidates = histogram.Uniform(n_splits_);
  //CHECK(candidates.size() > 0) << "Empty split (uniformed) candidates";
  Histogram::BinVal val_inf = histogram.InterpolateInf();
  double l_inf = val_inf.y;
  double m_inf = val_inf.m;
  SplitResult best_split;
  best_split.cost = std::numeric_limits<double>::max();
  best_split.can_split = false;

  // Minimum samples required to initiate a split
  if (m_inf <= 2) {
    return best_split;
  }

  for (unsigned int i = 0; i < candidates.size(); ++i) {
    double s = candidates[i];
    // TODO: Make sure that cache is in action
    Histogram::BinVal val_s = histogram.Interpolate(s);
    double l_s = val_s.y;
    double m_s = val_s.m;
    double y_left = l_s / m_s;
    double y_right = (l_inf - l_s) / (m_inf - m_s);
    // (Estimated) Minimum observations at leaf
    if (!(m_s > 0 && (m_inf-m_s) > 0)) {
      continue;
    }
    // No need to split if node is already pure
    // TODO: Can be checked out of this loop ?
    if (std::fabs(y_left - y_right) < 10e-6) {
      continue;
    }

    CHECK((m_inf - m_s) > 0) << "Deviding by zero in right side at s = " << s << ": i = " << i << "; m_inf: " << m_inf << "; m_s: " << m_s;
    double cost = - (l_s*l_s)/(m_s) - (l_inf-l_s)*(l_inf-l_s)/(m_inf-m_s);
    if (cost < best_split.cost) {
      best_split.cost = cost;
      best_split.threshold = s;
      best_split.label_left = y_left;
      best_split.label_right = y_right;
      best_split.can_split = true;
    }
  }
  //LOG(INFO) << "Done finding best split: " << best_threshold;
  return best_split;
};

Histogram Tree::ComputeHistogram(
    const std::vector<DataMatrix::FeaturePoint>& column,
    const std::vector<double>& targets,
    const std::vector<unsigned int>& samples) const {
  //LOG(INFO) << "Computing histogram";
  Histogram histogram(n_bins_);
  for (unsigned int i = 0; i < samples.size(); ++i) {
    unsigned int sample_idx = samples[i];
    double fval = column[sample_idx].value;
    double tval = targets[sample_idx];
    histogram.Update(fval, tval);
  }
  //LOG(INFO) << "Done computing histogram";
  return histogram;
};

double Tree::TraverseTree(const DataMatrix::SamplePoint& sample) const {
  unsigned int id = 0;
  while (true) {
    //LOG(INFO) << "Traversing: Now at node " << id;
    if (nodes_[id].is_leaf) {
      return nodes_[id].label;
    }
    unsigned int fidx = nodes_[id].feature_index;
    //LOG(INFO) << "Considering feature " << fidx << " at node " << id;
    double thres = nodes_[id].threshold;
    if (sample.features.at(fidx) <= thres) {
      id = nodes_[id].left_id;
    }
    else {
      id = nodes_[id].right_id;
    }
  };
};
