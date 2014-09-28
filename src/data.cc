#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <glog/logging.h>

#define LINE_BUFFER 50000

DataMatrix::DataMatrix() { }

int DataMatrix::Load(const char *filename) {
  // Column wise storage
  column_data_.clear();
  // Row wise storage
  row_data_.clear();
  targets_.clear();

  FILE* fp = fopen(filename, "r");
  if (fp == NULL) {
      printf("Error in reading file %s\n", filename);
      return -1;
  }

  int sample_idx = 0;
  char buffer[LINE_BUFFER];
  char buffer_ftoken[LINE_BUFFER];

  while (fgets(buffer, LINE_BUFFER, fp)) {

    buffer[strlen(buffer)-1] = '\0';
    //printf("Read line #%d\n", sample_idx);

    SamplePoint sample_point;
    row_data_.push_back(sample_point);

    char *pch;
    pch = strtok(buffer, " ");

    int count_ftoken = 0;
    while (pch != NULL)
    {
      if (count_ftoken == 0) {
        //Label
        double target = atof(pch);
        targets_.push_back(target);
        //printf("Target: %f\n", target);
      }
      else {
        strcpy(buffer_ftoken, pch);
        int i;
        for (i = 0; buffer_ftoken[i] != ':'; ++i);
        buffer_ftoken[i] = '\0';
        int fidx = atoi(buffer_ftoken);
        double fval = atof(&buffer_ftoken[i+1]);
        //printf("%d: %f\n", fidx, fval);
        FeaturePoint fpoint;
        fpoint.sample_index = sample_idx;
        fpoint.value = fval;
        if (column_data_.find(fidx) == column_data_.end()) {
          column_data_[fidx] = std::vector<FeaturePoint>();
        }
        column_data_[fidx].push_back(fpoint);

        // NOTE: sample_index might be redundant in row data, need SamplePoint?
        row_data_.back().features.insert(std::make_pair(fidx, fval));
      }
      pch = strtok (NULL, " ");
      count_ftoken += 1;
      //printf ("%s\n",pch);
    }
    sample_idx += 1;
  }
  fclose(fp);
};

unsigned int DataMatrix::Size() {
  return row_data_.size();
};

unsigned int DataMatrix::Dimension() {
  return column_data_.size();
};

const DataMatrix::SamplePoint& DataMatrix::GetRow(unsigned int index) const {
  CHECK(index < row_data_.size()) << "Row index out of bound";
  return row_data_[index];
};

const std::vector<DataMatrix::FeaturePoint>& DataMatrix::GetColumn(unsigned int index) const {
  // TODO: Efficiency?
  CHECK(column_data_.find(index) != column_data_.end()) << "Feature index not found";
  return column_data_.at(index);
};

const std::vector<double>& DataMatrix::GetTargets() const {
  return targets_;
};

void DataMatrix::SetTargets(const std::vector<double>& targets) {
  targets_ = targets;
};
