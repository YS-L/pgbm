#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <glog/logging.h>

#define LINE_BUFFER 50000

DataMatrix::DataMatrix() { }

/* Load data stored in svmlight format
 *
 * @param[in] filename Data file's name
 * @param[in] skips Number of rows to skip from the beginning, default is 0.
 * @param[in] max_num_samples Number of samples to load if > -1 (default).
 * @return Returns 0 on success.
 */
int DataMatrix::Load(const char *filename, int skips, int max_num_samples) {
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
  int num_line_read = 0;
  char buffer[LINE_BUFFER];
  char buffer_ftoken[LINE_BUFFER];


  while (fgets(buffer, LINE_BUFFER, fp)) {

    num_line_read += 1;

    if (num_line_read <= skips) {
      continue;
    }

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

    if (max_num_samples >= 0 && sample_idx >= max_num_samples) {
      break;
    }
  }
  fclose(fp);
  return 0;
};

unsigned int DataMatrix::Size() const {
  return row_data_.size();
};

unsigned int DataMatrix::Dimension() const {
  return column_data_.size();
};

const DataMatrix::SamplePoint& DataMatrix::GetRow(unsigned int index) const {
  CHECK(index < row_data_.size()) << "Row index out of bound";
  return row_data_[index];
};

const std::vector<DataMatrix::SamplePoint>& DataMatrix::GetRows() const {
  return row_data_;
};

const std::map<unsigned int, std::vector<DataMatrix::FeaturePoint> >& DataMatrix::GetColumns() const {
  return column_data_;
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


std::vector<unsigned int> DataMatrix::GetFeatureKeys() const {
  std::vector<unsigned int> keys;
  for (auto it = column_data_.begin(); it != column_data_.end(); ++it) {
    keys.push_back(it->first);
  }
  return keys;
};
