#include "gtest/gtest.h"
#include "data.h"
#include <algorithm>
#include <vector>
#include <cstdio>

#define IRIS "bin/data/iris.svmlight"

TEST(DataTest, LoadIris) {
  DataMatrix d;
  ASSERT_EQ(0, d.Load(IRIS));
  ASSERT_EQ(4, d.Dimension());
  ASSERT_EQ(150, d.Size());

  auto row = d.GetRow(0);
  ASSERT_EQ(4, row.size());
  std::vector<int> row_findices;
  std::vector<double> row_fvals;
  // TODO: Test more systematically with googlemock?
  for (unsigned int i = 0; i < row.size(); ++i) {
    row_findices.push_back(row[i].feature_index);
    row_fvals.push_back(row[i].value);
  }
  ASSERT_EQ(0, row_findices[0]);
  ASSERT_EQ(1, row_findices[1]);
  ASSERT_NEAR(5.1, row_fvals[0], 0.001);
  ASSERT_NEAR(3.5, row_fvals[1], 0.001);

  auto col = d.GetColumn(0);
  ASSERT_EQ(150, col.size());

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
