#include "reader.h"

#include <cstdlib>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <math.h>
#include <sstream>

using namespace std; // using std

DEFINE_double(reader_num_columns, 784, "The number of columns in the reader");

#define EPS 1e-6

namespace io {

bool CsvReader::read(int read_rows) {
  ifstream file(file_path_.c_str());
  CHECK(file.is_open());
  string line;
  int num_rows = 0;
  // constructs the dense mat
  while (getline(file, line)) {
    if (num_rows++ == read_rows)
      break;
    vector<double> vec;
    vector<sparse_pair> sparse_vec;
    stringstream ss(line);
    double val;
    int idx = 0;
    while (ss >> val) {
      if (idx == label_index_) {
        labels_.push_back((int)val);
      } else {
        vec.push_back(val);
      }
      idx++;
      // ignore comma and space while reading
      while (ss.peek() == ' ' || ss.peek() == ',') {
        ss.ignore();
      }
    }
    dense_mat_.push_back(vec);
  }
  num_rows_ = dense_mat_.size();
  num_cols_ = dense_mat_[0].size();
  // constructs the sparse mat
  for (int i = 0; i < num_rows_; i++) {
    vector<sparse_pair> sparse_vec;
    for (int j = 0; j < num_cols_; j++) {
      if (fabs(dense_mat_[i][j]) > EPS) {
        sparse_vec.push_back(make_pair(j, dense_mat_[i][j]));
      }
    }
    sparse_mat_.push_back(std::move(sparse_vec));
  }
  // randomly generates the labels if <label_index_> is -1.
  if (label_index_ == -1) {
    for (int i = 0; i < num_rows_; i++) {
      labels_.push_back(rand() % 2);
    }
  }

  file.close();
  return true;
}

void CsvReader::sample_mini_batch(int mini_batch_size) {
  dense_mini_batch_.clear();
  sparse_mini_batch_.clear();
  mini_batch_labels_.clear();
  for (int i = 0; i < mini_batch_size; i++) {
    int row_index = rand() % num_rows_;
    dense_mini_batch_.push_back(dense_mat_[row_index]);
    sparse_mini_batch_.push_back(sparse_mat_[row_index]);
    mini_batch_labels_.push_back(labels_[row_index]);
  }
  return;
}

bool LibsvmReader::read(int read_rows) {
  ifstream file(file_path_.c_str());
  CHECK(file.is_open());
  string line;
  int num_rows = 0;
  while (getline(file, line)) {
    if (num_rows++ == read_rows)
      break;
    stringstream ss(line);
    string val;
    ss >> val;
    labels_.push_back(stoi(val));
    vector<sparse_pair> sparse_vec;
    while (ss >> val) {
      std::size_t delim = val.find(':');
      int pos = stoi(val.substr(0, delim));
      if (pos >= FLAGS_reader_num_columns)
        continue;
      double value = stod(val.substr(delim + 1, val.length()));
      sparse_vec.push_back(make_pair(pos, value));
    }
    sparse_mat_.push_back(std::move(sparse_vec));
  }
  num_rows_ = sparse_mat_.size();
  num_cols_ = FLAGS_reader_num_columns;
  // constructs the dense mat
  for (int i = 0; i < num_rows_; i++) {
    // Initialize an all zero vector with dimension FLAGS_reader_num_columns.
    std::vector<double> dense_row(FLAGS_reader_num_columns, 0);
    for (int j = 0; j < sparse_mat_[i].size(); j++) {
      dense_row[sparse_mat_[i][j].first] = sparse_mat_[i][j].second;
    }
    dense_mat_.push_back(dense_row);
  }

  file.close();
  return true;
}

void LibsvmReader::sample_mini_batch(int mini_batch_size, int mini_batch_id) {
  dense_mini_batch_.clear();
  sparse_mini_batch_.clear();
  mini_batch_labels_.clear();
  const int start_row_index = mini_batch_id * mini_batch_size;
  const int end_row_index = (mini_batch_id + 1) * mini_batch_size;
  for (int i = start_row_index; i < end_row_index; i++) {
    dense_mini_batch_.push_back(dense_mat_[i]);
    sparse_mini_batch_.push_back(sparse_mat_[i]);
    mini_batch_labels_.push_back(labels_[i]);
  }
}

} // namespace io
