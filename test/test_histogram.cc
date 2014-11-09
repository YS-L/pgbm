#include "gtest/gtest.h"
#include "histogram.h"
#include "util.h"
#include <algorithm>
#include <vector>
#include <cstdio>

TEST(HistogramTest, Update) {
  unsigned int N = 10;
  Histogram hist(N);
  for (unsigned int i = 1; i <= 2 * N; ++i) {
    hist.Update(i, 1);
    ASSERT_EQ(std::min(i, N), hist.get_num_bins());
  }
  Vector<Histogram::Bin> bins = hist.get_bins();
  for (unsigned int i = 0; i < bins.size(); ++i) {
    printf("p:%f -> (%f, %f)\n", bins[i].p, bins[i].val.m, bins[i].val.y);
  }
};

TEST(HistogramTest, BenHaimAppendixA) {
  unsigned int N = 5;
  Histogram hist(N);
  Vector<Histogram::Bin> bins;
  std::vector<int> sequence = {23,19,10,16,36,2,9,32,30,45};
  for (int i = 0; i < 5; ++i) {
    hist.Update(sequence[i], 1);
  }
  hist.Update(sequence[5], 1);
  bins = hist.get_bins();
  for (unsigned int i = 0; i < bins.size(); ++i) {
    printf("p:%f -> (%f, %f)\n", bins[i].p, bins[i].val.m, bins[i].val.y);
  }
  hist.Update(sequence[6], 1);
  bins = hist.get_bins();
  printf("-------------\n");
  for (unsigned int i = 0; i < bins.size(); ++i) {
    printf("p:%f -> (%f, %f)\n", bins[i].p, bins[i].val.m, bins[i].val.y);
  }

  Histogram hist2(N);
  hist2.Update(sequence[7], 1);
  hist2.Update(sequence[8], 1);
  hist2.Update(sequence[9], 1);
  hist.Merge(hist2);

  bins = hist.get_bins();
  printf("-------------\n");
  for (unsigned int i = 0; i < bins.size(); ++i) {
    printf("p:%f -> (%f, %f)\n", bins[i].p, bins[i].val.m, bins[i].val.y);
  }

  Histogram::BinVal M15 = hist.Interpolate(15);
  printf("Number of points smaller than 15: %f, %f\n", M15.m, M15.y);

  ASSERT_NEAR(3.275, M15.m, 0.01);
  ASSERT_NEAR(3.275, M15.y, 0.01);

  std::vector<double> uniform_points = hist.Uniform(3);
  ASSERT_EQ(2, (int)uniform_points.size());
  printf("-------------\n");
  for (unsigned int i = 0; i < uniform_points.size(); ++i) {
    printf("%f ", uniform_points[i]);
  }
  printf("\n");

  printf("Expected # of points in each uniform region: ");
  double p, p_prev=0;
  for (unsigned int i = 0; i < uniform_points.size(); ++i) {
    p = hist.Interpolate(uniform_points[i]).m;
    printf("%f ", p - p_prev);
    p_prev = p;
  }
  printf("\n");

  ASSERT_NEAR(15.21, uniform_points[0], 0.02);
  ASSERT_NEAR(28.98, uniform_points[1], 0.02);

};

TEST(HistogramTest, Uniform) {
  unsigned int N = 5;
  Histogram hist(N);
  std::vector<Histogram::Bin> bins;
  std::vector<int> sequence = {23,19,10,16,36,2,9,32,30,45};
  for (unsigned int i = 0; i < sequence.size(); ++i) {
    hist.Update(sequence[i], 1);
  }
  // TODO: About >= 25 will fail, since now points before first bin is crudely
  // discarded
  unsigned int B = 20;
  std::vector<double> uniform_points = hist.Uniform(B);
  ASSERT_EQ(B-1, uniform_points.size());
};
