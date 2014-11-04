#include "tree.h"
#include "mpi_util.h"

#include <cstdlib>
#include <cmath>
#include <limits>
#include <utility>
#include <glog/logging.h>

Tree::Tree(unsigned int max_depth, unsigned int n_bins, unsigned int n_splits):
  max_depth_(max_depth), n_bins_(n_bins), n_splits_(n_splits),
  current_node_id_(0),
  current_depth_(0)
{
};

void Tree::Train(const DataMatrix& data) {
  Train(data, data.GetTargets());
};

void Tree::Train(const DataMatrix& data, const std::vector<double>& targets) {
  InitializeRootNode(data);
  while (current_depth_ < max_depth_) {
    ProcessCurrentNodes(data, targets);
    current_depth_ += 1;
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
  nodes_.resize(1 << (max_depth_+1));
  unsigned int root_id = current_node_id_++;
  Node& root = nodes_[root_id];
  root.samples.reserve(data.Size());
  root.depth = 0;
  // NOTE: Even if the almost unncessary label is to be filled, it has to be
  // determined by the master later.
  for (unsigned int i = 0; i < data.Size(); ++i) {
    root.samples.push_back(i);
  }
  current_queue_.push_back(root_id);
};

void Tree::ProcessCurrentNodes(const DataMatrix& data, const std::vector<double>& targets) {

  // Histograms are stored by features. In the distributed version, a message
  // completely covers a feature (over multiple nodes), so that the number of
  // messages sent does not grow with tree depth.
  // histograms[x][y]: Histogram for feature x at node y
  const std::vector<unsigned int> feature_keys = data.GetFeatureKeys();
  std::vector<std::vector<Histogram> > histograms(feature_keys.size(),
      std::vector<Histogram>(current_queue_.size(), Histogram(n_bins_)));

  mpi::environment& env = MPIHandle::Get().env;
  mpi::communicator& world = MPIHandle::Get().world;

  //LOG(INFO) << "Rank: " << world.rank();
  //LOG(INFO) << "Environment intialized: " << env.initialized();

  // In case of slaves
  int total_reqs_push_histograms = histograms.size();
  mpi::request reqs_push_histograms[total_reqs_push_histograms];
  std::vector<HistogramsPerFeature> messages(total_reqs_push_histograms);

  for (unsigned int i = 0; i < feature_keys.size(); ++i) {
    unsigned int fkey = feature_keys[i];
    const auto& column = data.GetColumn(fkey);
    for (unsigned int j = 0; j < current_queue_.size(); ++j) {
      histograms[i][j] = ComputeHistogram(column, targets, nodes_[current_queue_[j]].samples);
    }
    messages[i].histograms = histograms[i];
    messages[i].feature_index = i; // TODO: i or fkey?
    if (world.rank() >= 1) {
      int tag = (world.rank()-1)*feature_keys.size() + i;
      //LOG(INFO) << "Push tag: " << tag;
      reqs_push_histograms[i] = world.isend(0, tag, messages[i]);
      // TODO: Need to run a dummy test() to kick start the process?
      reqs_push_histograms[i].test();
    }
  }
  if (world.rank() >= 1) {
    LOG(INFO) << "Finalize pushing";
    mpi::wait_all(reqs_push_histograms, reqs_push_histograms + total_reqs_push_histograms);
    LOG(INFO) << "Pushed all histograms";
  }

  if (world.rank() == 0 && world.size() > 1) {
    LOG(INFO) << "Pulling";
    MPI_PullHistograms(histograms);
  }

  next_queue_.clear();

  // This way ordering of nodes in the next processing queue is defined by the
  // master process.
  std::vector<SplitResult> best_splits_by_nodes(current_queue_.size());

  if ( world.rank() == 0 ) {
    for (unsigned int j = 0; j < current_queue_.size(); ++j) {
      SplitResult best_result;
      best_result.can_split = false;
      best_result.cost = std::numeric_limits<double>::max();
      for (unsigned int i = 0; i < feature_keys.size(); ++i) {
        SplitResult result = FindBestSplit(histograms[i][j]);
        if (result.can_split && result.cost < best_result.cost) {
          // TODO: Options to check for number of observations per leaf, etc. Or
          // check for these within FindBestSplit using the histogram?
          best_result = result;
          best_result.feature_index = i;
        }
      }
      // TODO: Make a ref?
      best_splits_by_nodes[j] = best_result;
    }
    MPI_PushBestSplits(best_splits_by_nodes);
  }
  else {
    LOG(INFO) << "Receiving split results...";
    MPI_PullBestSplits(best_splits_by_nodes);
  }

  for (unsigned int j = 0; j < current_queue_.size(); ++j) {

    SplitResult& best_result = best_splits_by_nodes[j];

    if (!best_result.can_split) {
      return;
    }
    best_result.id_left = current_node_id_++;
    best_result.id_right = current_node_id_++;

    //LOG(INFO) << "Node [" << current_queue_[j] <<"]: Making a split on feature " << best_result.feature_index << " with threshold " << best_result.threshold << "; label_left: " << best_result.label_left << " label_right: " << best_result.label_right;

    FinalizeAndSplitNode(data, best_result, nodes_[current_queue_[j]]);

    //LOG(INFO) << "Done processing node " << current_queue_[j];
    next_queue_.push_back(best_result.id_left);
    next_queue_.push_back(best_result.id_right);
  }

  current_queue_.swap(next_queue_);

  LOG(INFO) << "ProcessCurrentNodes: Done depth " << current_depth_;
};

void Tree::MPI_PullHistograms(std::vector<std::vector<Histogram> >& histograms) const {
  // TODO: or pass in env and world?
  mpi::environment& env = MPIHandle::Get().env;
  mpi::communicator& world = MPIHandle::Get().world;

  int total_reqs = (world.size()-1) * histograms.size();
  //LOG(INFO) << "MPI_PullHistograms: # total reqs: " << total_reqs;
  mpi::request reqs[total_reqs];
  std::vector<HistogramsPerFeature> messages(total_reqs);
  for (unsigned int i = 0; i < histograms.size(); ++i) {
    for (int k = 1; k < world.size(); ++k) {
      int cur_idx = i*(world.size()-1) + (k-1);
      int tag = (k-1)*histograms.size() + i;
      //LOG(INFO) << "MPI_PullHistograms: Current message index to pull: " << cur_idx;
      reqs[cur_idx] = world.irecv(k, tag, messages[cur_idx]);
      // TODO: This causes blocking?
      //reqs[cur_idx].test();
    }
  }
  LOG(INFO) << "MPI_PullHistograms: Start waiting for slave histograms";
  mpi::wait_all(reqs, reqs + total_reqs);
  LOG(INFO) << "MPI_PulHistograms: Finished waiting in pull";
  for (int i = 0; i < total_reqs; i++) {
    unsigned int feature_index = messages[i].feature_index;
    std::vector<Histogram>& h = messages[i].histograms;
    for (unsigned int j = 0; j < h.size(); ++j) {
      histograms[feature_index][j].Merge(h[j]);
    }
  }
};

void Tree::MPI_PushHistograms(const std::vector<std::vector<Histogram> >& histograms) const {
  mpi::environment& env = MPIHandle::Get().env;
  mpi::communicator& world = MPIHandle::Get().world;
};

void Tree::MPI_PushBestSplits(const std::vector<SplitResult>& best_splits_by_nodes) const {
  LOG(INFO) << "MPI_PushBestSplits: Pushing best splits";
  mpi::environment& env = MPIHandle::Get().env;
  mpi::communicator& world = MPIHandle::Get().world;
  mpi::request reqs[world.size()-1];
  for (int k = 1; k < world.size(); ++k) {
    reqs[k-1] = world.isend(k, MPI_TagBestSplits(), best_splits_by_nodes);
    // TODO: Empty optional and double test()/wait()?
    reqs[k-1].test();
  }
  mpi::wait_all(reqs, reqs + world.size() - 1);
  LOG(INFO) << "MPI_PushBestSplits: Pushes finalized";
};

void Tree::MPI_PullBestSplits(std::vector<SplitResult>& best_splits_by_nodes) const {
  LOG(INFO) << "MPI_PullBestSplits: Pulling best splits";
  mpi::environment& env = MPIHandle::Get().env;
  mpi::communicator& world = MPIHandle::Get().world;
  world.recv(0, MPI_TagBestSplits(), best_splits_by_nodes);
  LOG(INFO) << "MPI_PullBestSplits: Pull finalized";
};

int Tree::MPI_TagBestSplits() const {
  return 1001;
};

void Tree::FinalizeAndSplitNode(const DataMatrix& data, const SplitResult& result, Node& parent) {

  // Finalize current node
  parent.is_leaf = false;
  parent.feature_index = result.feature_index;
  parent.threshold = result.threshold;

  // Reference is OK since nodes_ will not be reallocated
  Node& next_left = nodes_[result.id_left];
  Node& next_right = nodes_[result.id_right];

  next_left.id = result.id_left;
  next_left.depth = parent.depth + 1;
  next_left.label = result.label_left;
  next_left.is_leaf = true;
  next_right.id = result.id_right;
  next_right.depth = parent.depth + 1;
  next_right.label = result.label_right;
  next_right.is_leaf = true;

  auto features = data.GetColumn(parent.feature_index);
  for(unsigned int i = 0; i < parent.samples.size(); ++i) {
    unsigned int sample_idx = parent.samples[i];
    if (features[sample_idx].value < parent.threshold) {
      next_left.samples.push_back(sample_idx);
    }
    else {
      next_right.samples.push_back(sample_idx);
    }
  }

  // After split is decided and samples divided to the children, no longer need
  // to retain the sample indices
  parent.samples.clear();

  // Link parent and childrens
  parent.left_id = result.id_left;
  parent.right_id = result.id_right;
  //LOG(INFO) << "Node " << nodes_[node_id].id << "'s left: " << nodes_[node_id].left_id << "; right: " << nodes_[node_id].right_id;
};

Tree::SplitResult Tree::FindBestSplit(const Histogram& histogram) const {
  //LOG(INFO) << "Finding best split";

  SplitResult best_split;
  best_split.cost = std::numeric_limits<double>::max();
  best_split.can_split = false;

  // Possible that the histogram is empty because of no sample at all (TO CHECK)
  if (histogram.get_num_bins() == 0) {
    return best_split;
  }

  std::vector<double> candidates = histogram.Uniform(n_splits_);
  //CHECK(candidates.size() > 0) << "Empty split (uniformed) candidates";
  Histogram::BinVal val_inf = histogram.InterpolateInf();
  double l_inf = val_inf.y;
  double m_inf = val_inf.m;

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
  //LOG(INFO) << "Done finding best split with cost: " << best_split.cost;
  return best_split;
};

Histogram Tree::ComputeHistogram(
    const std::vector<DataMatrix::FeaturePoint>& column,
    const std::vector<double>& targets,
    const std::vector<unsigned int>& samples) const {
  //TODO: What if samples is empty? Now in the next step find best split will
  //ignore the resulted empty histogram.

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
