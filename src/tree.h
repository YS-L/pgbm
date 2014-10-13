#ifndef TREE_H_
#define TREE_H_

#include "histogram.h"
#include "data.h"

#include <vector>
#include <queue>

class Tree {

public:

  struct Node {
    // ID of the current node
    unsigned int id;
    // Indices of the samples reaching the tree
    std::vector<unsigned int> samples;
    // Feature used for splitting
    unsigned int feature_index;
    // Threshold feature value for splitting
    double threshold;
    // Target value
    double label;
    // Is the node a leaf?
    bool is_leaf;
    // ID of the left node
    unsigned int left_id;
    // ID of the right node
    unsigned int right_id;
    // Current depth
    unsigned int depth;
  };

  Tree(unsigned int max_depth=3, unsigned int n_bins=40, unsigned int n_splits=20);

  void Train(const DataMatrix& data);
  void Train(const DataMatrix& data, const std::vector<double>& targets);
  std::vector<double> Predict(const DataMatrix& data) const;
  double Predict(const DataMatrix::SamplePoint& sample) const;

private:

  struct CandidateInfo {
    Histogram histogram;
    unsigned int node_id;
    unsigned int feature_index;
  };

  // Result of splitting on a single feature optimally
  struct SplitResult {
    double cost;
    double threshold;
    double label_left;
    double label_right;
    double label_self; // When can_split is false
    bool can_split;
    unsigned int id_left;
    unsigned int id_right;
    unsigned int feature_index;
  };

  void ProcessCurrentNodes(const DataMatrix& data, const std::vector<double>& targets);
  void InitializeRootNode(const DataMatrix& data);
  void FinalizeAndSplitNode(const DataMatrix& data, const SplitResult& result, Node& parent);
  SplitResult FindBestSplit(const Histogram& histogram) const;
  Histogram ComputeHistogram(
      const std::vector<DataMatrix::FeaturePoint>& column,
      const std::vector<double>& targets,
      const std::vector<unsigned int>& samples) const;
  double TraverseTree(const DataMatrix::SamplePoint& sample) const;

  unsigned int max_depth_;
  unsigned int n_bins_;
  unsigned int n_splits_;
  std::vector<Node> nodes_;
  unsigned int current_node_id_;
  std::vector<unsigned int> current_queue_;
  std::vector<unsigned int> next_queue_;
  unsigned int current_depth_;
};

#endif
