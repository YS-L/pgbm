#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include <vector>
#include <utility>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/mpi/datatype.hpp>

class Histogram {

public:

  Histogram(unsigned int num_bins=10);

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

  const std::vector<Bin>& get_bins() const {
    return bins_;
  }

private:

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & max_num_bins_;
    ar & bins_;
  }

  typedef std::vector<Bin> HistogramType;
  typedef HistogramType::iterator HistogramTypeIter;
  typedef HistogramType::const_iterator HistogramTypeConstIter;
  void Trim();
  void PrecomputeCumsums() const;

  unsigned int max_num_bins_;
  HistogramType bins_;
  mutable bool dirty_;
  mutable std::vector<BinVal> cumsums_;

};

BOOST_IS_MPI_DATATYPE(Histogram::BinVal);
BOOST_IS_MPI_DATATYPE(Histogram::Bin);

#endif
