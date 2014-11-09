#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include "util.h"

#include <vector>
#include <utility>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/array.hpp>
#include <boost/mpi/datatype.hpp>
#include <glog/logging.h>

class Histogram {

public:

  Histogram(unsigned int num_bins=10);

  // TODO
  ~Histogram();
  //Histogram(Histogram const&) = delete;
  //Histogram& operator=(Histogram const&) = delete;

  class BinVal {
  public:
    double m;
    double y;

  private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
      ar & m;
      ar & y;
    }
  };

  class Bin {
  public:
    double p;
    BinVal val;

    bool operator<(const Bin& rhs) const {
      return p < rhs.p;
    }

  private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
      ar & p;
      ar & val;
    }
  };

  void Update(double x, double y);
  void Merge(const Histogram& hist);
  BinVal Interpolate(double x) const;
  BinVal InterpolateInf() const;
  std::vector<double> Uniform(int N) const;

  unsigned int get_num_bins() const {
    return bins_.size();
  };

  const Vector<Bin>& get_bins() const {
    return bins_;
  }

private:

  friend class boost::serialization::access;
  /*
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & max_num_bins_;
    ar & bins_;
  }
  */

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & max_num_bins_;

    // Note that bins_ and bins_.size() are not available during loading
    if (Archive::is_saving::value) {
      bins_pod_size_ = bins_.size();
    }
    ar & bins_pod_size_;

    if (bins_pod_ != 0) {
      delete [] bins_pod_;
      bins_pod_ = 0;
    }
    bins_pod_ = new Bin[bins_pod_size_];

    if (Archive::is_saving::value) {
      //LOG(INFO) << "----> Saving archive";
      for (unsigned int i = 0; i < bins_.size(); ++i) {
        bins_pod_[i] = bins_[i];
      }
    }
    else {
      //LOG(INFO) << "----> Loading archive";
      bins_pending_pod_ = true;
    }
    ar & boost::serialization::make_array<Bin>(bins_pod_, bins_pod_size_);
  }

  /*
  template<class Archive>
  void save(Archive & ar, const unsigned int version) const {
    ar << max_num_bins_;
    //ar << bins_;
    bins_pod_size_ = bins_.size();
    ar << bins_pod_size_;
    for (unsigned int i = 0; i < bins_.size(); ++i) {
      bins_pod_[i] = bins_[i];
    }
    ar << boost::serialization::make_array<Bin>(bins_pod_, bins_pod_size_);
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version) {
    ar >> max_num_bins_;
    //ar >> bins_;
    bins_pending_pod_ = true;
    ar >> bins_pod_size_;
    boost::serialization::array<Bin> tmp_array;
    ar >> tmp_array;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();
  */

  //typedef std::vector<Bin> HistogramType;
  typedef Vector<Bin> HistogramType;
  typedef HistogramType::iterator HistogramTypeIter;
  typedef HistogramType::const_iterator HistogramTypeConstIter;
  void Trim();
  void PrecomputeCumsums() const;
  void SyncPodBins() const;

  unsigned int max_num_bins_;
  mutable HistogramType bins_;
  mutable Bin* bins_pod_;
  mutable unsigned int bins_pod_size_;
  mutable bool bins_pending_pod_;
  mutable bool dirty_;
  mutable std::vector<BinVal> cumsums_;

};


BOOST_IS_MPI_DATATYPE(Histogram::BinVal);
BOOST_IS_MPI_DATATYPE(Histogram::Bin);
//BOOST_IS_MPI_DATATYPE(Histogram);

#endif
