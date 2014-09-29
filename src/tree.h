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

  Tree(unsigned int max_depth=3, unsigned int n_bins=40, unsigned int n_splits=80);

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
    unsigned int threshold;
    double label_left;
    double label_right;
    double label_self; // When can_split is false
    bool can_split;
  };

  void ProcessNode(const DataMatrix& data, const std::vector<double>& targets,
      unsigned int node_id);
  void InitializeRootNode(const DataMatrix& data);
  void SplitNode(const DataMatrix& data, const Node& parent,
                 const SplitResult& result);
  unsigned int AddNode(Node& node);
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
  std::queue<unsigned int> processing_queue_;
};

#endif
