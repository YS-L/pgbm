#include "gtest/gtest.h"
#include "data.h"
#include <algorithm>
#include <vector>
#include <cstdio>

#define IRIS "bin/data/iris.svmlight"

TEST(DataTest, LoadIris) {
  DataMatrix d;
  ASSERT_EQ(0, d.Load(IRIS));
  ASSERT_EQ(4, (int)d.Dimension());
  ASSERT_EQ(150, (int)d.Size());

  auto row = d.GetRow(0);
  ASSERT_EQ(4, (int)row.features.size());
  ASSERT_NEAR(5.1, row.features.at(0), 0.001);
  ASSERT_NEAR(3.5, row.features.at(1), 0.001);

  auto col = d.GetColumn(0);
  ASSERT_EQ(150, (int)col.size());

  std::vector<int> col_sindices;
  std::vector<double> col_fvals;
  // TODO: Test more systematically with googlemock?
  for (unsigned int i = 0; i < col.size(); ++i) {
    col_sindices.push_back(col[i].sample_index);
    col_fvals.push_back(col[i].value);
  }
  ASSERT_EQ(0, col_sindices[0]);
  ASSERT_EQ(1, col_sindices[1]);
  ASSERT_NEAR(5.1, col_fvals[0], 0.001);
  ASSERT_NEAR(4.9, col_fvals[1], 0.001);
};

TEST(DataTest, UpdateTargets) {
  DataMatrix d;
  ASSERT_EQ(0, d.Load(IRIS));
  std::vector<double> new_targets(d.Size(), 0.0);
  d.SetTargets(new_targets);
  ASSERT_EQ(new_targets, d.GetTargets());
};

TEST(DataTest, LoadIrisPartial) {
  DataMatrix d;

  // Load only 50 samples
  ASSERT_EQ(0, d.Load(IRIS, 0, 50));
  ASSERT_EQ(4, (int)d.Dimension());
  ASSERT_EQ(50, (int)d.Size());

  // Skip 1 row, check that 2nd row becomes the 1st row
  ASSERT_EQ(0, d.Load(IRIS, 1, 50));
  ASSERT_EQ(4, (int)d.Dimension());
  ASSERT_EQ(50, (int)d.Size());

  auto row = d.GetRow(0);
  ASSERT_EQ(4, (int)row.features.size());
  ASSERT_NEAR(4.9, row.features.at(0), 0.001);
  ASSERT_NEAR(3.0, row.features.at(1), 0.001);
};
