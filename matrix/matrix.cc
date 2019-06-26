#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <set>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <unordered_map>
#include <vector>

#include "matrix.h"

#define NUM_CHUNKS 4
#define INIT_PAIRS_REF_THRESHOLD 5
#define NUM_ALIGN_BYTES 64

DECLARE_string(methods);
DECLARE_string(o);
DECLARE_int32(iters);
DECLARE_bool(verbose);

DEFINE_bool(preprocess_optimization, true, "If enabled, we enable the "
    "preprocess optimization that can preprocess the init pairs for better "
    "cache performance especially on mat times mat operations. It's a bit hacky, "
    "and feel free to turn off it.");

using namespace std;

namespace core {

// some utility functions and class used here.
namespace {
int max_elements(const std::vector<std::vector<int>> two_d_vectors) {
  CHECK(!two_d_vectors.empty());
  CHECK(!two_d_vectors[0].empty());
  int max_element = two_d_vectors[0][0];
  for (const auto &vector : two_d_vectors) {
    for (auto element : vector) {
      max_element = std::max(max_element, element);
    }
  }
  return max_element;
}

int num_elements(const std::vector<std::vector<int>> two_d_vectors) {
  int num = 0;
  for (const auto &vec : two_d_vectors) {
    num += vec.size();
  }
  return num;
}

std::unordered_map<double, uint32_t>
distinct_values(const std::vector<double> &values) {
  std::unordered_map<double, uint32_t> distinct_values;
  uint32_t index = 0;
  for (double value : values) {
    if (distinct_values.find(value) == distinct_values.end()) {
      distinct_values[value] = index++;
    }
  }
  return std::move(distinct_values);
}

std::unordered_map<double, uint32_t>
two_d_distinct_values(const std::vector<std::vector<double>> &values) {
  std::unordered_map<double, uint32_t> distinct_values;
  uint32_t index = 0;
  for (const auto &vec : values) {
    for (double value : vec) {
      if (distinct_values.find(value) == distinct_values.end()) {
        distinct_values[value] = index++;
      }
    }
  }
  return std::move(distinct_values);
}

// Computations for MatTimesOtherMat operations.
void mat_times_other_mat_impl(int num_init_pairs, bool* preprocessed_init_pairs, int num_dict_nodes,
    int num_rows, int dim, const uint32_t *row_sizes, const uint32_t* row_start_indexes,
    const uint32_t *codes, const DictNode3 *dict_nodes, bool* computed, double *dict_nodes_values, 
    double *results) {

  #pragma omp parallel proc_bind(spread) num_threads(4)
  {
    #pragma omp for schedule (static) ordered nowait
    for (int i = 0; i < num_rows; i++) {
      double *x = (double *)__builtin_assume_aligned(results + i * dim, NUM_ALIGN_BYTES);
      uint32_t code_idx = row_start_indexes[i];
      for (int j = 0; j < row_sizes[i]; j++) {
        const uint32_t code_num = codes[code_idx + j];
        if (code_num < num_init_pairs && preprocessed_init_pairs[code_num]) {
          continue;
        } 
        if (code_num < num_init_pairs || computed[code_num-num_init_pairs]) {
          double *v = (double *)__builtin_assume_aligned(
              dict_nodes_values + dim * code_num, NUM_ALIGN_BYTES);
  
          for (int k = 0; k < dim; k++) {
            x[k] += v[k];
          }
        } else {
          computed[code_num-num_init_pairs] = true;
          const DictNode3 &dict_node = dict_nodes[code_num];
          const uint32_t parent_code = dict_node.parent_idx;
          double *y = (double *)__builtin_assume_aligned(
              dict_nodes_values + dim * parent_code, NUM_ALIGN_BYTES);
          double *u = (double *)__builtin_assume_aligned(
              dict_nodes_values + dim * dict_node.last_idx, NUM_ALIGN_BYTES);
          double *w = (double *)__builtin_assume_aligned(
              dict_nodes_values + dim * code_num, NUM_ALIGN_BYTES);
  
          for (int k = 0; k < dim; k++) {
            x[k] += y[k] + u[k];
            w[k] = y[k] + u[k];
          } 
        }
      }
    }
  }
}

// Computations for OtherMatTimesMat operations.
void other_mat_times_mat_impl(int num_init_pairs, bool* preprocessed_init_pairs, int num_dict_nodes,
    int num_rows, int dim, const uint32_t *row_sizes, 
    const uint32_t* row_start_indexes, const uint32_t *codes,
    double *other_mat,
    const DictNode3 *dict_nodes, bool* referred, double *dict_nodes_counts) {

  #pragma omp parallel proc_bind(spread) num_threads(4)
  {
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_rows; ++i) {
      double *y = (double *)__builtin_assume_aligned(other_mat + i * dim, NUM_ALIGN_BYTES);
      uint32_t code_idx = row_start_indexes[i];
      for (int j = 0; j < row_sizes[i]; ++j) {
        const uint32_t code_num = codes[code_idx++];
        if (code_num < num_init_pairs && preprocessed_init_pairs[code_num]) continue;
        // __builtin_prefetch((char *)(dict_nodes_counts + codes[code_idx + 2] * dim), 0, 0);
        double *x = (double *)__builtin_assume_aligned(
            dict_nodes_counts + code_num * dim, NUM_ALIGN_BYTES);
        if (!referred[code_num]) {
          referred[code_num] = true;
          for (int k = 0; k < dim; k++) {
            x[k] = y[k];
          } 
        } else {
          for (int k = 0; k < dim; k++) {
            x[k] += y[k];
          }
        }
      }
    }
    #pragma omp barrier

    #pragma omp for schedule (static) ordered nowait
    for (int i = num_dict_nodes - 1; i >= num_init_pairs; --i) {
      if (!referred[i])
        continue;
      const DictNode3 dict_node = dict_nodes[i];
      double *x = (double *)__builtin_assume_aligned(
          dict_nodes_counts + dict_node.last_idx * dim, NUM_ALIGN_BYTES);
      double *y = (double *)__builtin_assume_aligned(
          dict_nodes_counts + dict_node.parent_idx * dim, NUM_ALIGN_BYTES);
      double *z =
          (double *)__builtin_assume_aligned(dict_nodes_counts + i * dim, NUM_ALIGN_BYTES);
      for (int j = 0; j < dim; j++) {
        x[j] += z[j];
        y[j] += z[j];
      }
    }
  }
}

} // namespace

int8_t determine_num_bytes(int max_number) {
  if (max_number < 256) { // 2^8
    return 1;
  } else if (max_number < 65536) { // 2^16
    return 2;
  } else if (max_number < 16777216) { // 2^24
    return 3;
  } else {
    return 4;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////
// LogicalCompressedMat code
/////////////////////////////////////////////////////////////////////////////////////////
uint32_t* LogicalCompressedMat::row_start_indexes_ = new uint32_t[1024];
DictNode3* LogicalCompressedMat::dict_nodes_ = new DictNode3[1024 * 1024];
double* LogicalCompressedMat::dict_nodes_values_ = new double[20 * 1024 * 1024];
char * LogicalCompressedMat::buffer_ = new char[10 * 1024 * 1024];
double* LogicalCompressedMat::results_buffer_ = (double*)aligned_alloc(NUM_ALIGN_BYTES, sizeof(double) * 1024 * 1024);
double* LogicalCompressedMat::other_mat_buffer_ = (double*)aligned_alloc(NUM_ALIGN_BYTES, sizeof(double) * 1024 * 1024);
bool* LogicalCompressedMat::preprocessed_init_pairs_ = (bool*)aligned_alloc(NUM_ALIGN_BYTES,
    sizeof(bool) * 1024 * 1024);
uint32_t* LogicalCompressedMat::init_pairs_row_counts_ = (uint32_t*)aligned_alloc(NUM_ALIGN_BYTES,
    sizeof(uint32_t) * 1024 * 1024);
uint32_t* LogicalCompressedMat::init_pairs_row_indexes_ = (uint32_t*)aligned_alloc(NUM_ALIGN_BYTES,
    sizeof(uint32_t) * 1024 * 1024 * INIT_PAIRS_REF_THRESHOLD);


int64_t LogicalCompressedMat::size() {
  // # of bytes used to store the header.
  int64_t mat_size = sizeof(int) * 4;
  mat_size += init_column_indexes_.size() * sizeof(int);
  mat_size += init_values_.size() * sizeof(double);
  mat_size += row_sizes_.size() * sizeof(int);
  mat_size += codes_.size() * sizeof(int);
  return mat_size;
}

std::string LogicalCompressedMat::serialize_as_string() {
  std::string ret_string;
  ret_string.resize(size());
  int64_t copy_index = 0;
  memcpy(&ret_string[copy_index], &num_init_pairs_, sizeof(int));
  copy_index += sizeof(int);
  memcpy(&ret_string[copy_index], &num_rows_, sizeof(int));
  copy_index += sizeof(int);
  memcpy(&ret_string[copy_index], &num_cols_, sizeof(int));
  copy_index += sizeof(int);
  memcpy(&ret_string[copy_index], &num_dict_nodes_, sizeof(int));
  copy_index += sizeof(int);
  memcpy(&ret_string[copy_index], &init_column_indexes_[0],
         init_column_indexes_.size() * sizeof(int));
  copy_index += init_column_indexes_.size() * sizeof(int);
  memcpy(&ret_string[copy_index], &init_values_[0],
         init_values_.size() * sizeof(double));
  copy_index += init_values_.size() * sizeof(double);
  memcpy(&ret_string[copy_index], &row_sizes_[0],
         row_sizes_.size() * sizeof(int));
  copy_index += row_sizes_.size() * sizeof(int);
  memcpy(&ret_string[copy_index], &codes_[0], codes_.size() * sizeof(int));
  copy_index += codes_.size() * sizeof(int);
  CHECK(copy_index == size());

  return std::move(ret_string);
}

LogicalCompressedMat
LogicalCompressedMat::CreateFromString(const std::string &data) {
  LogicalCompressedMat mat;

  int64_t copy_index = 0;
  memcpy(&mat.num_init_pairs_, &data[copy_index], sizeof(int));
  copy_index += sizeof(int);
  memcpy(&mat.num_rows_, &data[copy_index], sizeof(int));
  copy_index += sizeof(int);
  memcpy(&mat.num_cols_, &data[copy_index], sizeof(int));
  copy_index += sizeof(int);
  memcpy(&mat.num_dict_nodes_, &data[copy_index], sizeof(int));
  copy_index += sizeof(int);

  mat.init_column_indexes_.resize(mat.num_init_pairs_);
  memcpy(&mat.init_column_indexes_[0], &data[copy_index],
         mat.init_column_indexes_.size() * sizeof(int));
  copy_index += mat.init_column_indexes_.size() * sizeof(int);

  mat.init_values_.resize(mat.num_init_pairs_);
  memcpy(&mat.init_values_[0], &data[copy_index],
         mat.init_values_.size() * sizeof(double));
  copy_index += mat.init_values_.size() * sizeof(double);

  mat.row_sizes_.resize(mat.num_rows_);
  memcpy(&mat.row_sizes_[0], &data[copy_index],
         mat.row_sizes_.size() * sizeof(int));
  copy_index += mat.row_sizes_.size() * sizeof(int);

  CHECK((data.size() - copy_index) % sizeof(int) == 0);
  mat.codes_.resize((data.size() - copy_index) / sizeof(int));
  memcpy(&mat.codes_[0], &data[copy_index], mat.codes_.size() * sizeof(int));

  return std::move(mat);
}

bool LogicalCompressedMat::operator==(const LogicalCompressedMat &right) const {
  if (num_init_pairs_ != right.num_init_pairs_)
    return false;
  if (num_rows_ != right.num_rows_)
    return false;
  if (num_cols_ != right.num_cols_)
    return false;
  if (num_dict_nodes_ != right.num_dict_nodes_)
    return false;
  if (init_column_indexes_ != right.init_column_indexes_)
    return false;
  if (init_values_ != right.init_values_)
    return false;
  if (row_sizes_ != right.row_sizes_)
    return false;
  if (codes_ != right.codes_)
    return false;
  return true;
}

void LogicalCompressedMat::LightDecompression() const {
  if (light_decompressed_) return;

  // First populates dict_nodes for init_data.
  for (size_t i = 0; i < num_init_pairs_; i++) {
    dict_nodes_[i].first_idx = i;
  }

  uint32_t seq_num = num_init_pairs_;
  int code_idx = 0;
  for (size_t i = 0; i < num_rows_; i++) {
    uint32_t last_code_num = codes_[code_idx++];
    for (size_t j = 1; j < row_sizes_[i]; ++j) {
      const uint32_t code_num = codes_[code_idx++];

      // Inserts the new node by appending the last node.
      dict_nodes_[seq_num].first_idx = dict_nodes_[last_code_num].first_idx;
      dict_nodes_[seq_num].last_idx = dict_nodes_[code_num].first_idx;
      dict_nodes_[seq_num].parent_idx = last_code_num;
      ++seq_num;
      last_code_num = code_num;
    }
  }

  uint32_t nnz = 0;
  for (int i = 0; i < num_rows_; i++) {
    row_start_indexes_[i] = nnz;
    nnz += row_sizes_[i];
  }

  if (FLAGS_preprocess_optimization) {
    memset(reinterpret_cast<char*>(preprocessed_init_pairs_), 1,
        sizeof(bool) * num_init_pairs_);
    memset(reinterpret_cast<char*>(init_pairs_row_counts_), 0,
        sizeof(uint32_t) * num_init_pairs_);
    for (int i = 0; i < num_rows_; i++) {
      for (int j = 0; j < row_sizes_[i]; j++) {
        const uint32_t code = codes_[row_start_indexes_[i] + j];
        if (code >= num_init_pairs_) {
          const DictNode3& dict_node = dict_nodes_[code];
          preprocessed_init_pairs_[dict_node.last_idx] = false;
          if (dict_node.parent_idx < num_init_pairs_) {
            preprocessed_init_pairs_[dict_node.parent_idx] = false;
          }
        } else {
          if (init_pairs_row_counts_[code] == INIT_PAIRS_REF_THRESHOLD) {
            // Reach the limit.
            preprocessed_init_pairs_[code] = false;
          } else {
            init_pairs_row_indexes_[code * INIT_PAIRS_REF_THRESHOLD +
              init_pairs_row_counts_[code]] = i;
            init_pairs_row_counts_[code] ++;
          }
        }
      }
    }
  } else {
    memset(reinterpret_cast<char*>(preprocessed_init_pairs_), 0,
        sizeof(bool) * num_init_pairs_);
  }

  light_decompressed_ = true;
}

LogicalCompressedMat LogicalCompressedMat::CreateLogicalCompressedMat(
    const std::vector<std::vector<io::sparse_pair>> &sparse_matrix,
    const std::vector<io::sparse_pair> &init_pairs, int dim) {
  LogicalCompressedMat logical_compressed_mat;

  compress::LZWCompresser<io::sparse_pair, io::pair_hash<int, double>>
      compresser(sparse_matrix, dim);
  compresser.compress_data();
  std::vector<std::vector<int>> codes = compresser.get_codes();
  std::vector<io::sparse_pair> init_data = compresser.get_init_data();

  logical_compressed_mat.num_init_pairs_ = init_data.size();
  logical_compressed_mat.num_rows_ = codes.size();
  logical_compressed_mat.num_cols_ = dim;
  logical_compressed_mat.num_dict_nodes_ = compresser.get_seq_num();

  for (const auto &pair : init_data) {
    logical_compressed_mat.init_column_indexes_.push_back(pair.first);
    logical_compressed_mat.init_values_.push_back(pair.second);
  }

  for (const auto &row : codes) {
    logical_compressed_mat.row_sizes_.push_back(row.size());
    for (const auto &col : row) {
      logical_compressed_mat.codes_.push_back(col);
    }
  }

  return std::move(logical_compressed_mat);
}

std::vector<double> LogicalCompressedMat::MatMultiplyVec(
    const std::vector<double> &vec) const {
  std::vector<double> results(num_rows_, 0);

  bool *computed = (bool *)(buffer_);
  memset((char *)computed, 1, sizeof(bool) * num_init_pairs_);
  memset((char *)(computed + num_init_pairs_), 0,
      sizeof(bool) * (num_dict_nodes_ - num_init_pairs_));

  LightDecompression();

  for (size_t i = 0; i < num_init_pairs_; i++) {
    dict_nodes_values_[i] = init_values_[i] * vec[init_column_indexes_[i]];
  }

  int code_index = 0;
  uint32_t seq_num = num_init_pairs_;
  for (size_t i = 0; i < num_rows_; i++) {
    for (int j = 0; j < row_sizes_[i]; j++) {
      uint32_t code_num = codes_[code_index++];
      if (!computed[code_num]) {
        computed[code_num] = true;
        const DictNode3 &dict_node = dict_nodes_[code_num];
        dict_nodes_values_[code_num] = dict_nodes_values_[dict_node.parent_idx] +
                                    dict_nodes_values_[dict_node.last_idx];
      }
      results[i] += dict_nodes_values_[code_num];
    }
  }
  return std::move(results);
}

void LogicalCompressedMat::MatMultiplyOtherMatImpl(
    const std::vector<std::vector<double>> &other_mat,
    std::vector<std::vector<double>> *results) const {
  const int dim = other_mat[0].size();
  bool * computed = (bool*)(buffer_);
  memset((char*)computed, 0, num_dict_nodes_ - num_init_pairs_);
  memset((char*)results_buffer_, 0, sizeof(double) * dim * num_rows_);
  LightDecompression();

  #pragma omp parallel proc_bind(spread) num_threads(4)
  {
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_init_pairs_; ++i) {
      const uint32_t col_idx = init_column_indexes_[i];
      const double col_value = init_values_[i];
      double *y = (double *)__builtin_assume_aligned(
          other_mat[col_idx].data(), 16);

      if (preprocessed_init_pairs_[i]) {
        for (int j = 0; j < init_pairs_row_counts_[i]; j++) {
          const uint32_t row_number = init_pairs_row_indexes_[i * INIT_PAIRS_REF_THRESHOLD + j];
          double *x = (double *)__builtin_assume_aligned(results_buffer_ + row_number * dim, 16);
          for (int k=0; k < dim; k++) {
            x[k] += col_value * y[k];
          }
        }
      } else {
        double *x = (double *)__builtin_assume_aligned(
            dict_nodes_values_ + i * dim, 16);
        for (int j = 0; j < dim; ++j) {
          x[j] = col_value * y[j];
        }
      }
    }
  }

  mat_times_other_mat_impl(num_init_pairs_, preprocessed_init_pairs_, num_dict_nodes_, num_rows_, dim,
      &row_sizes_[0], row_start_indexes_, &codes_[0], dict_nodes_, computed, dict_nodes_values_, results_buffer_);

  #pragma omp parallel proc_bind(spread) num_threads(4)
  {
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < results->size(); i++) {
      for (int j = 0; j < dim; j++) {
        (*results)[i][j] = results_buffer_[i*dim + j];
      }
    }
  }
}

std::vector<std::vector<double>> LogicalCompressedMat::MatMultiplyOtherMat(
    const std::vector<std::vector<double>> &other_mat) const {
  const int dim = other_mat[0].size();

  std::vector<std::vector<double>> results(num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    results[i].resize(dim);
  }

  MatMultiplyOtherMatImpl(other_mat, &results);
  return std::move(results);
}

void LogicalCompressedMat::MatMultiplyOtherMatInPlace(
    const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {
  MatMultiplyOtherMatImpl(other_mat, results);
}

std::vector<double> LogicalCompressedMat::VecMultiplyMat(
    const std::vector<double> &vec) const {
  std::vector<double> results(num_cols_, 0);

  memset((char*) dict_nodes_values_, 0, sizeof(double) * num_dict_nodes_);
  bool *referred = (bool *)(buffer_);
  memset((char *)(referred), 0, sizeof(bool) * num_dict_nodes_);

  LightDecompression();

  uint32_t code_index = 0;
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < row_sizes_[i]; j++) {
      const uint32_t code_num = codes_[code_index++];
      referred[code_num] = true;
      dict_nodes_values_[code_num] += vec[i];
    }
  }
  for (int i = num_dict_nodes_ - 1; i >= num_init_pairs_; i--) {
    if (!referred[i])
      continue;
    dict_nodes_values_[dict_nodes_[i].last_idx] += dict_nodes_values_[i];
    dict_nodes_values_[dict_nodes_[i].parent_idx] += dict_nodes_values_[i];
  }

  for (int i = 0; i < num_init_pairs_; i++) {
    results[init_column_indexes_[i]] += dict_nodes_values_[i] * init_values_[i];
  }
  return std::move(results);
}

void LogicalCompressedMat::OtherMatMultiplyMatImpl(
    const std::vector<std::vector<double>> &other_mat,
    std::vector<std::vector<double>>* results) const {
  const int dim = other_mat[0].size();

  bool* referred = (bool*)(buffer_);
  memset((char*) referred, 0, num_dict_nodes_);

  LightDecompression();

  #pragma omp parallel proc_bind(spread) num_threads(4)
  {
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < other_mat.size(); i++) {
      for (int j = 0; j < dim; j++) {
        other_mat_buffer_[i * dim + j] = other_mat[i][j];
      }
    }
  }

  other_mat_times_mat_impl(num_init_pairs_, preprocessed_init_pairs_, num_dict_nodes_, num_rows_,
      dim, &row_sizes_[0], row_start_indexes_, &codes_[0], other_mat_buffer_, dict_nodes_,
      referred, dict_nodes_values_);

  memset((char*)results_buffer_, 0, sizeof(double) * results->size() * dim);

  #pragma omp parallel proc_bind(spread) num_threads(4)
  {
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_init_pairs_; ++i) {
      const int column_idx = init_column_indexes_[i];
      const double value = init_values_[i];
      double *y =
          (double *)__builtin_assume_aligned(results_buffer_ + column_idx * dim, 16);
      if (preprocessed_init_pairs_[i]) {
        for (int j = 0; j < init_pairs_row_counts_[i]; j++) {
          const uint32_t row_number = init_pairs_row_indexes_[i * INIT_PAIRS_REF_THRESHOLD + j];
          double *x = (double *)__builtin_assume_aligned(other_mat_buffer_ + row_number * dim, 16);
          for (int k = 0; k < dim; ++k) {
            y[k] += x[k] * value;
          }
        }
      } else {
        double *x =
          (double *)__builtin_assume_aligned(dict_nodes_values_ + i * dim, 16);
        for (int j = 0; j < dim; ++j) {
          y[j] += x[j] * value;
        }
      }
    }

    #pragma omp barrier
  
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < results->size(); i++) {
      for (int j = 0; j < dim; j++) {
        (*results)[i][j] = results_buffer_[i*dim + j];
      }
    }
  }
}

std::vector<std::vector<double>> LogicalCompressedMat::OtherMatMultiplyMat(
    const std::vector<std::vector<double>> &other_mat) const {
  const int dim = other_mat[0].size();

  std::vector<std::vector<double>> results(num_cols_);
  for (int i = 0; i < num_cols_; i++) {
    results[i].resize(dim);
  }
  OtherMatMultiplyMatImpl(other_mat, &results);
  return std::move(results);
}

void LogicalCompressedMat::OtherMatMultiplyMatInPlace(
    const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {
  // Initialize 'results' here.
  for (int i = 0; i < results->size(); i++) {
    for (int j = 0; j < (*results)[i].size(); j++) {
      (*results)[i][j] = 0;
    }
  }
  OtherMatMultiplyMatImpl(other_mat, results);
}

void LogicalCompressedMat::MatMultiplyScalar(int scalar) {
  for (int i = 0; i < num_init_pairs_; i++) {
    init_values_[i] *= scalar;
  }
}

void LogicalCompressedMat::MatSquare() {
  for (int i = 0; i < num_init_pairs_; i++) {
    init_values_[i] = init_values_[i] * init_values_[i];
  }
}

double LogicalCompressedMat::MatSum() const {
  double sum = 0;
  DictNode *dict_nodes = (DictNode *)(buffer_);

  // First populates the <dict_nodes> for init_data.
  for (int i = 0; i < num_init_pairs_; i++) {
    dict_nodes[i].col_value = init_values_[i];
    dict_nodes[i].value = dict_nodes[i].col_value;
  }

  uint32_t seq_num = num_init_pairs_;
  int code_index = 0;
  for (int i = 0; i < num_rows_; i++) {
    uint32_t code_num = codes_[code_index++];
    uint32_t last_code_num = code_num;
    sum += dict_nodes[code_num].value;
    uint32_t row_size = row_sizes_[i];
    for (int j = 1; j < row_size; j++) {
      code_num = codes_[code_index++];

      // Inserts the new code by appending the last node.
      dict_nodes[seq_num].col_value = dict_nodes[last_code_num].col_value;
      dict_nodes[seq_num++].value =
          dict_nodes[last_code_num].value + dict_nodes[code_num].col_value;

      sum += dict_nodes[code_num].value;

      last_code_num = code_num;
    }
  }

  return sum;
}

/////////////////////////////////////////////////////////////////////////////////////////
// CompressedMat code
/////////////////////////////////////////////////////////////////////////////////////////
// TODO(fengan): Use a more robust way to allocate buffer.
uint32_t *CompressedMat::init_column_indexes_ = new uint32_t[1024 * 1024];
uint32_t *CompressedMat::init_value_indexes_ = new uint32_t[1024 * 1024];
uint32_t *CompressedMat::row_sizes_ = new uint32_t[1024 * 1024];
uint32_t *CompressedMat::row_start_indexes_ = new uint32_t[1024];
uint32_t *CompressedMat::codes_ = new uint32_t [1024 * 1024];
DictNode3 *CompressedMat::dict_nodes_ = (DictNode3*)aligned_alloc(NUM_ALIGN_BYTES,
    sizeof(DictNode3) * 1024 * 1024);
double* CompressedMat::dict_nodes_values_ = (double*)aligned_alloc(NUM_ALIGN_BYTES,
    sizeof(double) * 30 * 1024 * 1024);
char *CompressedMat::buffer_ = (char*)aligned_alloc(NUM_ALIGN_BYTES, 10 * 1024 * 1024);
double* CompressedMat::results_buffer_ = (double*)aligned_alloc(NUM_ALIGN_BYTES,
    sizeof(double) * 1024 * 1024);
double* CompressedMat::other_mat_buffer_ = (double*)aligned_alloc(NUM_ALIGN_BYTES,
    sizeof(double) * 1024 * 1024);
bool* CompressedMat::preprocessed_init_pairs_ = (bool*)aligned_alloc(NUM_ALIGN_BYTES,
    sizeof(bool) * 1024 * 1024);
uint32_t* CompressedMat::init_pairs_row_counts_ = (uint32_t*)aligned_alloc(NUM_ALIGN_BYTES,
    sizeof(uint32_t) * 1024 * 1024);
uint32_t* CompressedMat::init_pairs_row_indexes_ = (uint32_t*)aligned_alloc(NUM_ALIGN_BYTES,
    sizeof(uint32_t) * 1024 * 1024 * INIT_PAIRS_REF_THRESHOLD);

void CompressedMat::physical_decoding() const {
  // Decode 'init_column_indexes_'.
  IntegerIterator init_column_indexes_iter(
      init_column_indexes_buffer_.data(), column_index_num_byte_);
  init_column_indexes_iter.GetInts(init_column_indexes_, num_init_pairs_);
  // Decode 'init_value_indexes_'.
  IntegerIterator init_value_indexes_iter(
      init_value_indexes_buffer_.data(), value_index_num_byte_);
  init_value_indexes_iter.GetInts(init_value_indexes_, num_init_pairs_);
  // Decode 'row_sizes_'.
  IntegerIterator row_sizes_iter(row_sizes_buffer_.data(), column_index_num_byte_);
  row_sizes_iter.GetInts(row_sizes_, num_rows_);
  // Populate 'row_start_indexes_'.
  int num_codes = 0;
  for (int i = 0; i < num_rows_; i++) {
    row_start_indexes_[i] = num_codes;
    num_codes += row_sizes_[i];
  }
  // Decode 'codes_'.
  IntegerIterator codes_iter(codes_buffer_.data(), code_num_byte_);
  codes_iter.GetInts(codes_, num_codes);
}

int64_t CompressedMat::size() {
  // # of bytes used to store the header.
  int64_t mat_size = sizeof(int8_t) * 3 + sizeof(int32_t) * 5;
  // Adds the buffer sizes.
  mat_size += init_column_indexes_buffer_.size();
  mat_size += init_value_indexes_buffer_.size();
  mat_size += values_buffer_.size();
  mat_size += row_sizes_buffer_.size();
  mat_size += codes_buffer_.size();
  return mat_size;
}

std::string CompressedMat::serialize_as_string() {
  std::string ret_string;
  ret_string.resize(size());
  int64_t index = 0;
  memcpy(&ret_string[index], (char *)&code_num_byte_, sizeof(uint8_t));
  index += sizeof(uint8_t);
  memcpy(&ret_string[index], (char *)&column_index_num_byte_, sizeof(uint8_t));
  index += sizeof(uint8_t);
  memcpy(&ret_string[index], (char *)&value_index_num_byte_, sizeof(uint8_t));
  index += sizeof(uint8_t);
  memcpy(&ret_string[index], (char *)&num_init_pairs_, sizeof(int));
  index += sizeof(int);
  memcpy(&ret_string[index], (char *)&num_double_values_, sizeof(int));
  index += sizeof(int);
  memcpy(&ret_string[index], (char *)&num_rows_, sizeof(int));
  index += sizeof(int);
  memcpy(&ret_string[index], (char *)&num_cols_, sizeof(int));
  index += sizeof(int);
  memcpy(&ret_string[index], (char *)&num_dict_nodes_, sizeof(int));
  index += sizeof(int);
  memcpy(&ret_string[index], &init_column_indexes_buffer_[0],
         init_column_indexes_buffer_.size());
  index += init_column_indexes_buffer_.size();
  memcpy(&ret_string[index], &init_value_indexes_buffer_[0],
         init_value_indexes_buffer_.size());
  index += init_value_indexes_buffer_.size();
  memcpy(&ret_string[index], &values_buffer_[0], values_buffer_.size());
  index += values_buffer_.size();
  memcpy(&ret_string[index], &row_sizes_buffer_[0], row_sizes_buffer_.size());
  index += row_sizes_buffer_.size();
  memcpy(&ret_string[index], &codes_buffer_[0], codes_buffer_.size());
  index += codes_buffer_.size();
  CHECK(index == size());

  return std::move(ret_string);
}

CompressedMat CompressedMat::CreateFromString(const std::string &data) {
  CompressedMat mat;
  mat.light_decompressed_ = false;
  int64_t index = 0;
  memcpy(&mat.code_num_byte_, &data[index], sizeof(int8_t));
  index += sizeof(int8_t);
  memcpy(&mat.column_index_num_byte_, &data[index], sizeof(int8_t));
  index += sizeof(int8_t);
  memcpy(&mat.value_index_num_byte_, &data[index], sizeof(int8_t));
  index += sizeof(int8_t);

  memcpy(&mat.num_init_pairs_, &data[index], sizeof(int));
  index += sizeof(int);
  memcpy(&mat.num_double_values_, &data[index], sizeof(int));
  index += sizeof(int);
  memcpy(&mat.num_rows_, &data[index], sizeof(int));
  index += sizeof(int);
  memcpy(&mat.num_cols_, &data[index], sizeof(int));
  index += sizeof(int);
  memcpy(&mat.num_dict_nodes_, &data[index], sizeof(int));
  index += sizeof(int);

  mat.init_column_indexes_buffer_.resize(mat.num_init_pairs_ *
                                         mat.column_index_num_byte_);
  memcpy(&mat.init_column_indexes_buffer_[0], &data[index],
         mat.init_column_indexes_buffer_.size());
  index += mat.init_column_indexes_buffer_.size();

  mat.init_value_indexes_buffer_.resize(mat.num_init_pairs_ *
                                        mat.value_index_num_byte_);
  memcpy(&mat.init_value_indexes_buffer_[0], &data[index],
         mat.init_value_indexes_buffer_.size());
  index += mat.init_value_indexes_buffer_.size();

  mat.values_buffer_.resize(mat.num_double_values_ * sizeof(double));
  memcpy(&mat.values_buffer_[0], &data[index], mat.values_buffer_.size());
  index += mat.values_buffer_.size();

  mat.row_sizes_buffer_.resize(mat.num_rows_ * mat.column_index_num_byte_);
  memcpy(&mat.row_sizes_buffer_[0], &data[index], mat.row_sizes_buffer_.size());
  index += mat.row_sizes_buffer_.size();

  // All the remaining bytes must belong to codes_buffer_
  mat.codes_buffer_.resize(data.size() - index);
  memcpy(&mat.codes_buffer_[0], &data[index], mat.codes_buffer_.size());
  index += mat.codes_buffer_.size();

  return std::move(mat);
}

bool CompressedMat::operator==(const CompressedMat &right) const {
  if (code_num_byte_ != right.code_num_byte_)
    return false;
  if (column_index_num_byte_ != right.column_index_num_byte_)
    return false;
  if (value_index_num_byte_ != right.value_index_num_byte_)
    return false;
  if (num_init_pairs_ != right.num_init_pairs_)
    return false;
  if (num_double_values_ != right.num_double_values_)
    return false;
  if (num_rows_ != right.num_rows_)
    return false;
  if (num_cols_ != right.num_cols_)
    return false;
  if (num_dict_nodes_ != right.num_dict_nodes_)
    return false;
  if (init_column_indexes_buffer_ != right.init_column_indexes_buffer_)
    return false;
  if (init_value_indexes_buffer_ != right.init_value_indexes_buffer_)
    return false;
  if (values_buffer_ != right.values_buffer_)
    return false;
  if (row_sizes_buffer_ != right.row_sizes_buffer_)
    return false;
  if (codes_buffer_ != right.codes_buffer_)
    return false;
  return true;
}

CompressedMat CompressedMat::CreateCompressedMat(
    const std::vector<std::vector<io::sparse_pair>> &sparse_matrix,
    const std::vector<io::sparse_pair> &init_pairs, int dim) {
  CompressedMat compressed_mat;

  compress::LZWCompresser<io::sparse_pair, io::pair_hash<int, double>>
      compresser(sparse_matrix, dim);
  compresser.compress_data();
  std::vector<std::vector<int>> codes = compresser.get_codes();
  std::vector<io::sparse_pair> init_data = compresser.get_init_data();
  std::vector<double> init_doubles;
  for (const auto &pair : init_data) {
    init_doubles.push_back(pair.second);
  }
  std::unordered_map<double, uint32_t> distinct_doubles =
      distinct_values(init_doubles);

  compressed_mat.light_decompressed_ = false;

  // Determines the type of <init_column_indexes> and <init_value_indexes>.
  compressed_mat.column_index_num_byte_ = determine_num_bytes(dim);
  compressed_mat.value_index_num_byte_ =
      determine_num_bytes(distinct_doubles.size());

  // First finds the maximal number in codes and then determines the type of
  // <codes>.
  int max_code = max_elements(codes);
  compressed_mat.code_num_byte_ = determine_num_bytes(max_code);

  compressed_mat.num_init_pairs_ = init_data.size();
  compressed_mat.num_double_values_ = distinct_doubles.size();
  compressed_mat.num_rows_ = codes.size();
  compressed_mat.num_cols_ = dim;
  compressed_mat.num_dict_nodes_ = compresser.get_seq_num();

  // Allocates buffers.
  compressed_mat.init_column_indexes_buffer_.resize(
      compressed_mat.num_init_pairs_ * compressed_mat.column_index_num_byte_);
  compressed_mat.init_value_indexes_buffer_.resize(
      compressed_mat.num_init_pairs_ * compressed_mat.value_index_num_byte_);
  compressed_mat.values_buffer_.resize(compressed_mat.num_double_values_ *
                                       sizeof(double));
  compressed_mat.row_sizes_buffer_.resize(
      compressed_mat.num_rows_ * compressed_mat.column_index_num_byte_);
  compressed_mat.codes_buffer_.resize(
      num_elements(codes) * compressed_mat.code_num_byte_ + /*buffer_space*/ 1);

  // Populates buffers. The bytes are populated assuming that the machine is
  // little-endian.
  for (int i = 0; i < init_data.size(); i++) {
    uint32_t column_index = init_data[i].first;
    double value = init_data[i].second;
    memcpy(&compressed_mat.init_column_indexes_buffer_
                [i * compressed_mat.column_index_num_byte_],
           (char *)&column_index, compressed_mat.column_index_num_byte_);
    memcpy(
        &compressed_mat
             .init_value_indexes_buffer_[i *
                                         compressed_mat.value_index_num_byte_],
        (char *)&(distinct_doubles[value]),
        compressed_mat.value_index_num_byte_);
  }
  for (const auto &pair : distinct_doubles) {
    memcpy(&compressed_mat.values_buffer_[pair.second * sizeof(double)],
           (char *)&pair.first, sizeof(double));
  }

  int code_count = 0;
  for (int i = 0; i < codes.size(); i++) {
    uint32_t row_size = codes[i].size();
    memcpy(&compressed_mat
                .row_sizes_buffer_[i * compressed_mat.column_index_num_byte_],
           (char *)&row_size, compressed_mat.column_index_num_byte_);
    for (int j = 0; j < row_size; j++) {
      uint32_t code = codes[i][j];
      memcpy(&compressed_mat
                  .codes_buffer_[code_count * compressed_mat.code_num_byte_],
             (char *)&code, compressed_mat.code_num_byte_);
      code_count++;
    }
  }
  return std::move(compressed_mat);
}

std::vector<double> CompressedMat::MatMultiplyVec(
    const std::vector<double> &vec) const {
  std::vector<double> results(num_rows_, 0);

  bool *computed = (bool *)(buffer_);
  memset((char *)computed, 1, sizeof(bool) * num_init_pairs_);
  memset((char *)(computed + num_init_pairs_), 0,
         sizeof(bool) * (num_dict_nodes_ - num_init_pairs_));
  double *values_buffer = (double *)(values_buffer_.data());

  LightDecompression();

  for (int i = 0; i < num_init_pairs_; i++) {
    dict_nodes_values_[i] = values_buffer[init_value_indexes_[i]] *
                         vec[init_column_indexes_[i]];
  }

  int run_index = 0;
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < row_sizes_[i]; j++) {
      const uint32_t code_num = codes_[run_index++];
      if (!computed[code_num]) {
        computed[code_num] = true;
        const DictNode3 &dict_node = dict_nodes_[code_num];
        dict_nodes_values_[code_num] = dict_nodes_values_[dict_node.parent_idx] +
                                  dict_nodes_values_[dict_node.last_idx];
      }
      results[i] += dict_nodes_values_[code_num];
    }
  }
  return std::move(results);
}

std::vector<std::vector<double>> CompressedMat::MatMultiplyOtherMat(
    const std::vector<std::vector<double>> &other_mat) const {
  const int dim = other_mat[0].size();
  std::vector<std::vector<double>> results(num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    results[i].resize(dim);
  }

  MatMultiplyOtherMatImpl(other_mat, &results);
  return std::move(results);
}

void CompressedMat::MatMultiplyOtherMatInPlace(
    const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {

  // auto start = std::chrono::system_clock::now();

  MatMultiplyOtherMatImpl(other_mat, results);

  // LOG(INFO) << "TOC, mat times other mat: "
  //           << std::chrono::duration_cast<std::chrono::nanoseconds>(
  //                  std::chrono::system_clock::now() - start)
  //                      .count() /
  //                  (1000.0 * 1000.0 * 1000.0)
  //           << " secs";
}

void CompressedMat::MatMultiplyOtherMatImpl(
    const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {

  const int dim = other_mat[0].size();
  double *values_buffer = (double *)(values_buffer_.data());
  bool *computed = (bool *)(buffer_);
  memset((char *)computed, 0, num_dict_nodes_ - num_init_pairs_);
  LightDecompression();

  memset((char*)results_buffer_, 0, sizeof(double) * results->size() * dim);

  #pragma omp parallel proc_bind(spread) num_threads(4)
  {
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < other_mat.size(); i++) {
      for (int j = 0; j < dim; j++) {
        other_mat_buffer_[i * dim + j] = other_mat[i][j];
      }
    }

    #pragma omp barrier

    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_init_pairs_; ++i) {
      const int col_index = init_column_indexes_[i];
      const double value = values_buffer[init_value_indexes_[i]];
      double *y = (double *)__builtin_assume_aligned(
          other_mat_buffer_ + col_index * dim, NUM_ALIGN_BYTES);
      if (preprocessed_init_pairs_[i]) {
        for (int j = 0; j < init_pairs_row_counts_[i]; j++) {
          const uint32_t row_number = init_pairs_row_indexes_[i * INIT_PAIRS_REF_THRESHOLD + j];
          double *x = (double *)__builtin_assume_aligned(results_buffer_ + row_number * dim, NUM_ALIGN_BYTES);
          for (int k=0; k < dim; k++) {
            x[k] += value * y[k];
          }
        }
      } else {
        double *x = (double *)__builtin_assume_aligned(
            dict_nodes_values_ + i * dim, NUM_ALIGN_BYTES);
        for (int j = 0; j < dim; ++j) {
          x[j] = value * y[j];
        }
      }
    }
  }
  mat_times_other_mat_impl(num_init_pairs_, preprocessed_init_pairs_, num_dict_nodes_, num_rows_, dim,
      row_sizes_, row_start_indexes_, codes_, dict_nodes_, computed,
      dict_nodes_values_, results_buffer_);

  #pragma omp parallel proc_bind(spread) num_threads(4)
  {
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < results->size(); i++) {
      for (int j = 0; j < dim; j++) {
        (*results)[i][j] = results_buffer_[i*dim + j];
      }
    }
  }
}

std::vector<double> CompressedMat::VecMultiplyMat(const std::vector<double> &vec)
  const {
  std::vector<double> results(num_cols_, 0);

  memset((char*)dict_nodes_values_, 0, sizeof(double) * num_dict_nodes_);
  bool *referred = (bool *)(buffer_);
  memset((char *)(referred), 0, sizeof(bool) * num_dict_nodes_);
  double *values_buffer = (double *)(values_buffer_.data());

  LightDecompression();

  int run_index = 0;
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < row_sizes_[i]; j++) {
      const uint32_t code_num = codes_[run_index++];
      referred[code_num] = true;
      dict_nodes_values_[code_num] += vec[i];
    }
  }
  for (int i = num_dict_nodes_ - 1; i >= num_init_pairs_; i--) {
    if (!referred[i])
      continue;
    dict_nodes_values_[dict_nodes_[i].last_idx] += dict_nodes_values_[i];
    dict_nodes_values_[dict_nodes_[i].parent_idx] += dict_nodes_values_[i];
  }

  for (int i = num_init_pairs_ - 1; i >= 0; i--) {
    results[init_column_indexes_[i]] +=
        dict_nodes_values_[i] * values_buffer[init_value_indexes_[i]];
  }
  return std::move(results);
}

std::vector<std::vector<double>> CompressedMat::OtherMatMultiplyMat(
    const std::vector<std::vector<double>> &other_mat) const {
  const int dim = other_mat[0].size();
  std::vector<std::vector<double>> results(num_cols_);
  for (int i = 0; i < num_cols_; i++) {
    results[i].resize(dim);
  }

  OtherMatMultiplyMatImpl(other_mat, &results);
  return std::move(results);
}

void CompressedMat::OtherMatMultiplyMatInPlace(
    const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {
  // auto start = std::chrono::system_clock::now();

  // Initialize 'results'.
  for (int i = 0; i < results->size(); i++) {
    for (int j = 0; j < (*results)[i].size(); j++) {
      (*results)[i][j] = 0;
    }
  }
  OtherMatMultiplyMatImpl(other_mat, results);

  // LOG(INFO) << "TOC, other mat times mat: "
  //           << std::chrono::duration_cast<std::chrono::nanoseconds>(
  //                  std::chrono::system_clock::now() - start)
  //                      .count() /
  //                  (1000.0 * 1000.0 * 1000.0)
  //           << " secs";
}

void CompressedMat::OtherMatMultiplyMatImpl(
    const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {
  const int dim = other_mat[0].size();

  const double* values_buffer = (double*) values_buffer_.data();

  LightDecompression();

  bool* referred = (bool*)(buffer_);
  memset((char*) referred, 0, num_dict_nodes_);

  #pragma omp parallel proc_bind(spread) num_threads(4)
  {
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < other_mat.size(); i++) {
      for (int j = 0; j < dim; j++) {
        other_mat_buffer_[i * dim + j] = other_mat[i][j];
      }
    }
  }

  other_mat_times_mat_impl(num_init_pairs_, preprocessed_init_pairs_, num_dict_nodes_, num_rows_,
      dim, row_sizes_, row_start_indexes_, codes_, other_mat_buffer_, dict_nodes_,
      referred, dict_nodes_values_);

  memset((char*)results_buffer_, 0, sizeof(double) * results->size() * dim);

  #pragma omp parallel proc_bind(spread) num_threads(4)
  {
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_init_pairs_; i++) {
      const int column_idx = init_column_indexes_[i];
      const double value = values_buffer[init_value_indexes_[i]];
      double *y =
          (double *)__builtin_assume_aligned(results_buffer_ + column_idx * dim, NUM_ALIGN_BYTES);
      if (preprocessed_init_pairs_[i]) {
        for (int j = 0; j < init_pairs_row_counts_[i]; j++) {
          const uint32_t row_number = init_pairs_row_indexes_[i * INIT_PAIRS_REF_THRESHOLD + j];
          double *x = (double *)__builtin_assume_aligned(other_mat_buffer_ + row_number * dim, NUM_ALIGN_BYTES);
          for (int k = 0; k < dim; ++k) {
            y[k] += x[k] * value;
          }
        }
      } else {
        double *x =
          (double *)__builtin_assume_aligned(dict_nodes_values_ + i * dim, NUM_ALIGN_BYTES);
        for (int j = 0; j < dim; ++j) {
          y[j] += x[j] * value;
        }
      }
    }
  
    #pragma omp barrier
  
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < results->size(); i++) {
      for (int j = 0; j < dim; j++) {
        (*results)[i][j] = results_buffer_[i*dim + j];
      }
    }
  }
}

void CompressedMat::MatMultiplyScalar(int scalar) {
  double *values = (double *)(&values_buffer_[0]);
  for (int i = 0; i < num_double_values_; i++) {
    values[i] = values[i] * scalar;
  }
}

void CompressedMat::MatSquare() {
  double *values = (double *)(&values_buffer_[0]);
  for (int i = 0; i < num_double_values_; i++) {
    values[i] = values[i] * values[i];
  }
}

double CompressedMat::MatSum() const {
  double sum = 0;

  DictNode *dict_nodes = (DictNode *)(buffer_);
  double *values_buffer = (double *)(values_buffer_.data());
  LightDecompression();

  // First populates the <dict_nodes> for init_data.
  for (int i = 0; i < num_init_pairs_; i++) {
    dict_nodes[i].col_value = values_buffer[init_value_indexes_[i]];
    dict_nodes[i].value = dict_nodes[i].col_value;
  }

  int run_index = 0;
  uint32_t seq_num = num_init_pairs_;
  for (int i = 0; i < num_rows_; i++) {
    uint32_t code_num = codes_[run_index++];
    uint32_t last_code_num = code_num;
    sum += dict_nodes[code_num].value;
    for (int j = 1; j < row_sizes_[i]; j++) {
      code_num = codes_[run_index++];

      // Inserts the new node by appending the last node.
      dict_nodes[seq_num].col_value = dict_nodes[last_code_num].col_value;
      dict_nodes[seq_num++].value =
          dict_nodes[last_code_num].value + dict_nodes[code_num].col_value;

      sum += dict_nodes[code_num].value;

      last_code_num = code_num;
    }
  }
  return sum;
}

void CompressedMat::LightDecompression() const {
  if (light_decompressed_) return;

  // auto start = std::chrono::system_clock::now();

  physical_decoding();

  // First populates dict_nodes for init_data.
  for (size_t i = 0; i < num_init_pairs_; i++) {
    dict_nodes_[i].first_idx = i;
  }

  uint32_t seq_num = num_init_pairs_;
  int code_idx = 0;
  for (size_t i = 0; i < num_rows_; i++) {
    uint32_t last_code_num = codes_[code_idx++];
    for (size_t j = 1; j < row_sizes_[i]; ++j) {
      const uint32_t code_num = codes_[code_idx++];

      // Inserts the new node by appending the last node.
      dict_nodes_[seq_num].first_idx = dict_nodes_[last_code_num].first_idx;
      dict_nodes_[seq_num].last_idx = dict_nodes_[code_num].first_idx;
      dict_nodes_[seq_num].parent_idx = last_code_num;
      ++seq_num;
      last_code_num = code_num;
    }
  }

  if (FLAGS_preprocess_optimization) {
    memset(reinterpret_cast<char*>(preprocessed_init_pairs_), 1,
        sizeof(bool) * num_init_pairs_);
    memset(reinterpret_cast<char*>(init_pairs_row_counts_), 0,
        sizeof(uint32_t) * num_init_pairs_);
    for (int i = 0; i < num_rows_; i++) {
      for (int j = 0; j < row_sizes_[i]; j++) {
        const uint32_t code = codes_[row_start_indexes_[i] + j];
        if (code >= num_init_pairs_) {
          const DictNode3& dict_node = dict_nodes_[code];
          preprocessed_init_pairs_[dict_node.last_idx] = false;
          if (dict_node.parent_idx < num_init_pairs_) {
            preprocessed_init_pairs_[dict_node.parent_idx] = false;
          }
        } else {
          if (init_pairs_row_counts_[code] == INIT_PAIRS_REF_THRESHOLD) {
            // Reach the limit.
            preprocessed_init_pairs_[code] = false;
          } else {
            init_pairs_row_indexes_[code * INIT_PAIRS_REF_THRESHOLD +
              init_pairs_row_counts_[code]] = i;
            init_pairs_row_counts_[code] ++;
          }
        }
      }
    }
  } else {
    memset(reinterpret_cast<char*>(preprocessed_init_pairs_), 0,
        sizeof(bool) * num_init_pairs_);
  }



  // LOG(INFO) << "TOC, light_decompression: "
  //           << std::chrono::duration_cast<std::chrono::nanoseconds>(
  //                  std::chrono::system_clock::now() - start)
  //                      .count() /
  //                  (1000.0 * 1000.0 * 1000.0)
  //           << " secs";

  light_decompressed_ = true;
}

std::vector<std::vector<io::sparse_pair>> CompressedMat::Decompression() {
  LightDecompression();

  double *values_buffer = (double *)(values_buffer_.data());
  std::vector<io::sparse_pair> init_data;
  init_data.resize(num_init_pairs_);
  for (int i = 0; i < num_init_pairs_; i++) {
    init_data[i] = std::make_pair(init_column_indexes_[i],
                       values_buffer[init_value_indexes_[i]]);
  }
  std::vector<std::vector<int>> codes;
  codes.resize(num_rows_);
  int max_code = -1;
  int run_index = 0;
  for (int i = 0; i < codes.size(); i++) {
    const int row_size = row_sizes_[i];
    codes[i].resize(row_size);
    for (int j = 0; j < row_size; j++) {
      codes[i][j] = codes_[run_index++];
      max_code = codes[i][j] > max_code ? codes[i][j] : max_code;
    }
  }
  compress::LZWDecompresser<io::sparse_pair, io::pair_hash<int, double>>
      decoder(codes, init_data, num_cols_, max_code + 1);
  CHECK(decoder.decompress_data());
  return decoder.get_data();
}

/////////////////////////////////////////////////////////////////////////////////////////
// Mat Code
/////////////////////////////////////////////////////////////////////////////////////////
double *Mat::DATA_BUFFER = new double[1024 * 1024];
double *Mat::OTHER_MAT_BUFFER = new double[1024 * 1024];

void Mat::MatMultiplyScalar(int scalar) {
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < num_cols_; j++) {
      data_[i][j] = data_[i][j] * scalar;
    }
  }
}

void Mat::MatSquare() {
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < num_cols_; j++) {
      data_[i][j] = data_[i][j] * data_[i][j];
    }
  }
}

double Mat::MatSum() const {
  double sum = 0;
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < num_cols_; j++) {
      sum += data_[i][j];
    }
  }
  return sum;
}

std::vector<double> Mat::MatMultiplyVec(const std::vector<double> &vec) const {
  std::vector<double> results(num_rows_, 0);
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < num_cols_; j++) {
      results[i] += data_[i][j] * vec[j];
    }
  }
  return std::move(results);
}

std::vector<double> Mat::VecMultiplyMat(const std::vector<double> &vec) const {
  std::vector<double> results(num_cols_, 0);
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < num_cols_; j++) {
      results[j] += data_[i][j] * vec[i];
    }
  }
  return std::move(results);
}

std::vector<std::vector<double>> Mat::MatMultiplyOtherMat(
    const std::vector<std::vector<double>> &other_mat) const {
  int dim = other_mat[0].size();
  std::vector<std::vector<double>> results(num_rows_);
  for (int i = 0; i < num_rows_; i++) {
    results[i].resize(dim);
  }

  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < num_cols_; k++) {
        results[i][j] += data_[i][k] * other_mat[k][j];
      }
    }
  }
  return std::move(results);
}

void Mat::MatMultiplyOtherMatInPlace(const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {
  // auto start = std::chrono::system_clock::now();
  const int dim = other_mat[0].size();
  #pragma omp parallel
  {
    // Transpose 'other_mat'.
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_cols_; i++) {
      for (int j = 0; j < dim; j++) {
        OTHER_MAT_BUFFER[j*num_cols_ + i] = other_mat[i][j];
      }
    }
    #pragma omp barrier

    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_rows_; i++) {
      const double* x = &data_[i][0];
      for (int j = 0; j < dim; j++) {
        double& result = (*results)[i][j];
        result = 0;
        const double* y = OTHER_MAT_BUFFER + j * num_cols_;
        #pragma omp simd reduction(+: result)
        for (int k = 0; k < num_cols_; k++) {
          result += x[k] * y[k];
        }
      }
    }
  }

  // LOG(INFO) << "Mat, mat times other mat: "
  //           << std::chrono::duration_cast<std::chrono::nanoseconds>(
  //                  std::chrono::system_clock::now() - start)
  //                      .count() /
  //                  (1000.0 * 1000.0 * 1000.0)
  //           << " secs";
}

std::vector<std::vector<double>> Mat::OtherMatMultiplyMat(
    const std::vector<std::vector<double>>& other_mat) const {
  const int dim = other_mat[0].size();
  std::vector<std::vector<double>> results(num_cols_);
  for (int i = 0; i < num_cols_; i++) {
    results[i].resize(dim);
  }

  // for (int k=0; k < num_rows_; k++) {
  //   for (int i=0; i < num_cols_; i++) {
  //     for (int j =0; j < dim; j++) {
  //       results[i][j] += other_mat[k][j] * data_[k][i];
  //     }
  //   }
  // }
  for (int i = 0; i < num_cols_; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < num_rows_; k++) {
        results[i][j] += other_mat[k][j] * data_[k][i];
      }
    }
  }
  return std::move(results);
}

void Mat::OtherMatMultiplyMatInPlace(const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {
  // auto start = std::chrono::system_clock::now();
  const int dim = other_mat[0].size();

  #pragma omp parallel
  {
    // Transpose 'other_mat'.
    #pragma omp for schedule (static) nowait
    for (int k = 0; k < num_rows_; k++) {
      for (int i = 0; i < dim; i++) {
        OTHER_MAT_BUFFER[i*num_rows_ + k] = other_mat[k][i];
      }
    }
    #pragma omp for schedule (static) nowait
    // Transpose 'data_'.
    for (int k = 0; k < num_rows_; k++) {
      for (int j = 0; j < num_cols_; j++) {
        DATA_BUFFER[j*num_rows_ + k] = data_[k][j];
      }
    }
    #pragma omp barrier

    #pragma omp for schedule (static) nowait
    for (int j = 0; j < num_cols_; j++) {
      const double* y = DATA_BUFFER + j * num_rows_;
      for (int i = 0; i < dim; i++) {
        const double* x = OTHER_MAT_BUFFER + i * num_rows_;
        double& result = (*results)[j][i];
        result = 0;
        #pragma omp simd reduction(+: result)
        for (int k = 0; k < num_rows_; k++) {
          result += x[k] * y[k];
        }
      }
    }
  }

  //LOG(INFO) << "Mat Other mat times mat: "
  //          << std::chrono::duration_cast<std::chrono::nanoseconds>(
  //                 std::chrono::system_clock::now() - start)
  //                     .count() /
  //                 (1000.0 * 1000.0 * 1000.0)
  //          << " secs";
}

Mat Mat::CreateMat(std::vector<std::vector<double>> data) {
  Mat mat;
  mat.num_rows_ = data.size();
  mat.num_cols_ = data[0].size();
  mat.data_ = std::move(data);
  return std::move(mat);
}

Mat Mat::CreateFromString(const std::string &data) {
  Mat mat;
  int64_t index = 0;
  memcpy(&mat.num_rows_, &data[index], sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&mat.num_cols_, &data[index], sizeof(uint32_t));
  index += sizeof(uint32_t);
  for (int i = 0; i < mat.num_rows_; i++) {
    mat.data_.emplace_back();
    double val;
    for (int j = 0; j < mat.num_cols_; j++) {
      memcpy(&val, &data[index], sizeof(double));
      mat.data_[i].push_back(val);
      index += sizeof(double);
    }
  }
  return std::move(mat);
}

int64_t Mat::size() {
  int64_t ret_size = sizeof(uint32_t) + sizeof(uint32_t);
  ret_size += static_cast<int64_t>(num_rows_) * static_cast<int64_t>(num_cols_) *
              sizeof(double);
  return ret_size;
}

std::string Mat::serialize_as_string() {
  std::string ret_string;
  ret_string.resize(size());
  int64_t index = 0;
  memcpy(&ret_string[index], &num_rows_, sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&ret_string[index], &num_cols_, sizeof(uint32_t));
  index += sizeof(uint32_t);
  for (const auto &row : data_) {
    for (double val : row) {
      memcpy(&ret_string[index], &val, sizeof(double));
      index += sizeof(double);
    }
  }
  return std::move(ret_string);
}

bool Mat::operator==(const Mat &right) const {
  if (data_ != right.data_)
    return false;
  if (num_rows_ != right.num_rows_)
    return false;
  if (num_cols_ != right.num_cols_)
    return false;
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////
// DviMat Code
/////////////////////////////////////////////////////////////////////////////////////////
double *DviMat::DATA_BUFFER = new double[10 * 1024 * 1024];
double *DviMat::OTHER_MAT_BUFFER = new double[10 * 1024 * 1024];
DviMat DviMat::CreateDviMat(const std::vector<std::vector<double>> &data) {
  DviMat mat;
  mat.num_rows_ = data.size();
  mat.num_cols_ = data[0].size();

  std::vector<double> doubles;
  std::unordered_map<double, uint32_t> distinct_values =
      two_d_distinct_values(data);
  mat.index_num_bytes_ = determine_num_bytes(distinct_values.size());
  mat.indexes_buffer_.resize(mat.num_rows_ * mat.num_cols_ *
                             mat.index_num_bytes_);
  mat.values_buffer_.resize(distinct_values.size() * sizeof(double));
  mat.num_values_ = distinct_values.size();

  // Populates the indexes
  int run_index = 0;
  for (const auto &row : data) {
    for (double value : row) {
      int index = distinct_values[value];
      memcpy(&mat.indexes_buffer_[(run_index++) * mat.index_num_bytes_],
             (char *)&index, mat.index_num_bytes_);
    }
  }

  // Populates the values
  for (const auto &pair : distinct_values) {
    double value = pair.first;
    int index = pair.second;
    memcpy(&mat.values_buffer_[index * sizeof(double)], (char *)&value,
           sizeof(double));
  }

  return mat;
}

bool DviMat::operator==(const DviMat &right) const {
  if (num_rows_ != right.num_rows_)
    return false;
  if (num_cols_ != right.num_cols_)
    return false;
  if (num_values_ != right.num_values_)
    return false;
  if (index_num_bytes_ != right.index_num_bytes_)
    return false;
  if (indexes_buffer_ != right.indexes_buffer_)
    return false;
  if (values_buffer_ != right.values_buffer_)
    return false;
  return true;
}

std::string DviMat::serialize_as_string() {
  std::string ret_string;
  ret_string.resize(size());
  int64_t index = 0;
  memcpy(&ret_string[index], &num_rows_, sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&ret_string[index], &num_cols_, sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&ret_string[index], &num_values_, sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&ret_string[index], &index_num_bytes_, sizeof(uint8_t));
  index += sizeof(uint8_t);
  memcpy(&ret_string[index], &indexes_buffer_[0], indexes_buffer_.size());
  index += indexes_buffer_.size();
  memcpy(&ret_string[index], &values_buffer_[0], values_buffer_.size());
  index += values_buffer_.size();
  CHECK(index == size());

  return std::move(ret_string);
}

DviMat DviMat::CreateFromString(const std::string &data) {
  DviMat mat;
  int64_t index = 0;
  memcpy(&mat.num_rows_, &data[index], sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&mat.num_cols_, &data[index], sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&mat.num_values_, &data[index], sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&mat.index_num_bytes_, &data[index], sizeof(uint8_t));
  index += sizeof(uint8_t);
  mat.indexes_buffer_.resize(mat.num_rows_ * mat.num_cols_ *
                             mat.index_num_bytes_);
  memcpy(&mat.indexes_buffer_[0], &data[index], mat.indexes_buffer_.size());
  index += mat.indexes_buffer_.size();
  mat.values_buffer_.resize(mat.num_values_ * sizeof(double));
  memcpy(&mat.values_buffer_[0], &data[index], mat.values_buffer_.size());
  index += mat.values_buffer_.size();

  return std::move(mat);
}

int64_t DviMat::size() {
  int64_t mat_size = 3 * sizeof(uint32_t) + 1 * sizeof(uint8_t);
  mat_size += indexes_buffer_.size();
  mat_size += values_buffer_.size();
  return mat_size;
}

std::vector<double>
DviMat::MatMultiplyVec(const std::vector<double> &vec) const {
  std::vector<double> results(num_rows_);
  double *values = (double *)(&values_buffer_[0]);
  IntegerIterator iter(indexes_buffer_.data(), index_num_bytes_);
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < num_cols_; j++) {
      results[i] += values[iter.NextInt()] * vec[j];
    }
  }
  return results;
}

std::vector<double>
DviMat::VecMultiplyMat(const std::vector<double> &vec) const {
  std::vector<double> results(num_cols_);
  double *values = (double *)(&values_buffer_[0]);
  IntegerIterator iter(indexes_buffer_.data(), index_num_bytes_);
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < num_cols_; j++) {
      results[j] += vec[i] * values[iter.NextInt()];
    }
  }
  return results;
}

std::vector<std::vector<double>> DviMat::MatMultiplyOtherMat(
    const std::vector<std::vector<double>> &other_mat) const {
  int dim = other_mat[0].size();
  std::vector<std::vector<double>> results(num_rows_);
  for (int i = 0; i < num_rows_; i++) {
    results[i].resize(dim);
  }
  double *values = (double *)(&values_buffer_[0]);
  IntegerIterator iter(indexes_buffer_.data(), index_num_bytes_);
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < num_cols_; j++) {
      double value = values[iter.NextInt()];
      for (int k = 0; k < dim; k++) {
        results[i][k] += value * other_mat[j][k];
      }
    }
  }
  return std::move(results);
}

void DviMat::MatMultiplyOtherMatInPlace(const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {
  // TODO(fenganli): Fill in the implementation here.
  const int dim = other_mat[0].size();
  #pragma omp parallel
  {
    // Transpose 'other_mat'.
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_cols_; i++) {
      for (int j = 0; j < dim; j++) {
        OTHER_MAT_BUFFER[j*num_cols_ + i] = other_mat[i][j];
      }
    }
    // Decompress 'mat'
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_rows_; i++) {
      for (int j = 0; j < num_cols_; j++) {
        DATA_BUFFER[i*num_cols_ + j] = seek_by_indexes(i, j);
      }
    }
    #pragma omp barrier

    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_rows_; i++) {
      const double* x = DATA_BUFFER + i * num_cols_;
      for (int k = 0; k < dim; k++) {
        double& result = (*results)[i][k];
        result = 0;
        const double* y = OTHER_MAT_BUFFER + k * num_cols_;
        #pragma omp simd reduction(+: result)
        for (int j = 0; j < num_cols_; j++) {
          result += x[j] * y[j];
        }
      }
    }

  }
}

std::vector<std::vector<double>> DviMat::OtherMatMultiplyMat(
    const std::vector<std::vector<double>> &other_mat) const {
  const int dim = other_mat[0].size();
  std::vector<std::vector<double>> results(num_cols_);
  for (int i = 0; i < num_cols_; ++i) {
    results[i].resize(dim);
  }
  double *values = (double *)(&values_buffer_[0]);
  IntegerIterator iter(indexes_buffer_.data(), index_num_bytes_);
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < num_cols_; j++) {
      double value = values[iter.NextInt()];
      for (int k = 0; k < dim; k++) {
        results[j][k] += other_mat[i][k] * value;
      }
    }
  }
  return std::move(results);
}

void DviMat::OtherMatMultiplyMatInPlace(const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {
  // TODO(fenganli): Fill in the implementation here.

  const int dim = other_mat[0].size();

  #pragma omp parallel
  {
    // Transpose 'other_mat'.
    #pragma omp for schedule (static) nowait
    for (int k = 0; k < num_rows_; k++) {
      for (int i = 0; i < dim; i++) {
        OTHER_MAT_BUFFER[i*num_rows_ + k] = other_mat[k][i];
      }
    }
    #pragma omp for schedule (static) nowait
    // Transpose 'data_'.
    for (int k = 0; k < num_rows_; k++) {
      for (int j = 0; j < num_cols_; j++) {
        DATA_BUFFER[j*num_rows_ + k] = seek_by_indexes(k, j);
      }
    }
    #pragma omp barrier

    #pragma omp for schedule (static) nowait
    for (int j = 0; j < num_cols_; j++) {
      const double* y = DATA_BUFFER + j * num_rows_;
      for (int i = 0; i < dim; i++) {
        const double* x = OTHER_MAT_BUFFER + i * num_rows_;
        double& result = (*results)[j][i];
        result = 0;
        #pragma omp simd reduction(+: result)
        for (int k = 0; k < num_rows_; k++) {
          result += x[k] * y[k];
        }
      }
    }
  }
}

void DviMat::MatMultiplyScalar(int scalar) {
  double *values = (double *)(&values_buffer_[0]);
  for (int i = 0; i < num_values_; i++) {
    values[i] = values[i] * scalar;
  }
}

void DviMat::MatSquare() {
  double *values = (double *)(&values_buffer_[0]);
  for (int i = 0; i < num_values_; i++) {
    values[i] = values[i] * values[i];
  }
}

double DviMat::MatSum() const {
  double sum = 0;
  double *values = (double *)(&values_buffer_[0]);
  IntegerIterator iter(indexes_buffer_.data(), index_num_bytes_);
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < num_cols_; j++) {
      sum += values[iter.NextInt()];
    }
  }
  return sum;
}

/////////////////////////////////////////////////////////////////////////////////////////
// CsrMat Code
/////////////////////////////////////////////////////////////////////////////////////////
CsrMat CsrMat::CreateCsrMat(
    const std::vector<std::vector<io::sparse_pair>> &data, int dim) {
  CsrMat csr_mat;
  csr_mat.num_rows_ = data.size();
  csr_mat.num_cols_ = dim;
  int nnz = 0;
  for (const auto &row : data) {
    csr_mat.row_start_indexes_.push_back(nnz);
    nnz += row.size();
    csr_mat.row_sizes_.push_back(row.size());
    for (const auto &pair : row) {
      csr_mat.col_indexes_.push_back(pair.first);
      csr_mat.values_.push_back(pair.second);
    }
  }
  return std::move(csr_mat);
}

CsrMat CsrMat::CreateFromString(const std::string &data) {
  CsrMat csr_mat;
  int64_t index = 0;
  memcpy(&csr_mat.num_rows_, &data[index], sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&csr_mat.num_cols_, &data[index], sizeof(uint32_t));
  index += sizeof(uint32_t);
  uint32_t nnz = 0;
  csr_mat.row_sizes_.resize(csr_mat.num_rows_);
  csr_mat.row_start_indexes_.resize(csr_mat.num_rows_);
  memcpy(&csr_mat.row_sizes_[0], &data[index], sizeof(uint32_t) * csr_mat.num_rows_);
  for (int i = 0; i < csr_mat.num_rows_; i++) {
    csr_mat.row_start_indexes_[i] = nnz;
    nnz += csr_mat.row_sizes_[i];
  }
  index += sizeof(uint32_t) * csr_mat.num_rows_;
  csr_mat.col_indexes_.resize(nnz);
  memcpy(&csr_mat.col_indexes_[0], &data[index], sizeof(uint32_t) * nnz);
  index += sizeof(uint32_t) * nnz;
  csr_mat.values_.resize(nnz);
  memcpy(&csr_mat.values_[0], &data[index], sizeof(double) * nnz);
  return std::move(csr_mat);
}

int64_t CsrMat::size() {
  int64_t ret_size = sizeof(uint32_t) + sizeof(uint32_t);
  ret_size += sizeof(uint32_t) * row_sizes_.size();
  ret_size += sizeof(uint32_t) * col_indexes_.size();
  ret_size += sizeof(double) * values_.size();
  return ret_size;
}

std::string CsrMat::serialize_as_string() {
  std::string ret_string;
  ret_string.resize(size());
  int64_t index = 0;
  memcpy(&ret_string[index], &num_rows_, sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&ret_string[index], &num_cols_, sizeof(uint32_t));
  index += sizeof(uint32_t);
  for (uint32_t row_size : row_sizes_) {
    memcpy(&ret_string[index], &row_size, sizeof(uint32_t));
    index += sizeof(uint32_t);
  }
  for (uint32_t col_index : col_indexes_) {
    memcpy(&ret_string[index], &col_index, sizeof(uint32_t));
    index += sizeof(uint32_t);
  }
  for (double value : values_) {
    memcpy(&ret_string[index], &value, sizeof(double));
    index += sizeof(double);
  }
  return std::move(ret_string);
}

bool CsrMat::operator==(const CsrMat &right) const {
  if (num_rows_ != right.num_rows_)
    return false;
  if (num_cols_ != right.num_cols_)
    return false;
  if (row_sizes_ != right.row_sizes_)
    return false;
  if (col_indexes_ != right.col_indexes_)
    return false;
  if (values_ != right.values_)
    return false;
  return true;
}

std::vector<double>
CsrMat::MatMultiplyVec(const std::vector<double> &vec) const {
  std::vector<double> results(num_rows_);
  int run_index = 0;
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < row_sizes_[i]; j++) {
      results[i] += values_[run_index] * vec[col_indexes_[run_index]];
      run_index++;
    }
  }
  return results;
}

std::vector<double>
CsrMat::VecMultiplyMat(const std::vector<double> &vec) const {
  std::vector<double> results(num_cols_);
  int run_index = 0;
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < row_sizes_[i]; j++) {
      results[col_indexes_[run_index]] += vec[i] * values_[run_index];
      run_index++;
    }
  }
  return results;
}

std::vector<std::vector<double>> CsrMat::MatMultiplyOtherMat(
    const std::vector<std::vector<double>> &other_mat) const {
  int dim = other_mat[0].size();
  std::vector<std::vector<double>> results(num_rows_);
  for (int i = 0; i < num_rows_; i++) {
    results[i].resize(dim);
  }
  // int run_index = 0;
  // for (int i = 0; i < num_rows_; i++) {
  //   for (int j=0; j<row_sizes_[i]; j++) {
  //     int col_index = col_indexes_[run_index];
  //     double value = values_[run_index++];
  //     for (int k = 0; k < dim; k++) {
  //       results[i][k] += value * other_mat[col_index][k];
  //     }
  //   }
  // }
  for (int k = 0; k < dim; k++) {
    int run_index = 0;
    for (int i = 0; i < num_rows_; i++) {
      for (int j = 0; j < row_sizes_[i]; j++) {
        int col_index = col_indexes_[run_index];
        double value = values_[run_index++];
        results[i][k] += value * other_mat[col_index][k];
      }
    }
  }
  return std::move(results);
}

void CsrMat::MatMultiplyOtherMatInPlace(const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {

  // auto start = std::chrono::system_clock::now();
  const int dim = other_mat[0].size();
  // #pragma omp parallel
  // {
  //   #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_rows_; i++) {
      for (int j = 0; j < dim; j++) {
        (*results)[i][j] = 0;
      }
    }
  //  #pragma omp barrier
  //  #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_rows_; i++) {
      int run_index = row_start_indexes_[i];
      for (int j = 0; j < row_sizes_[i]; j++) {
        const int col_index = col_indexes_[run_index];
        const double value = values_[run_index++];
        for (int k = 0; k < dim; k++) {
          (*results)[i][k] += value * other_mat[col_index][k];
        }
      }
    }
  // }

  // LOG(INFO) << "Csr Mat_times_other_mat_time: "
  //           << std::chrono::duration_cast<std::chrono::nanoseconds>(
  //                  std::chrono::system_clock::now() - start)
  //                      .count() /
  //                  (1000.0 * 1000.0 * 1000.0)
  //           << " secs";
}


std::vector<std::vector<double>> CsrMat::OtherMatMultiplyMat(
    const std::vector<std::vector<double>> &other_mat) const {
  const int dim = other_mat[0].size();
  std::vector<std::vector<double>> results(num_cols_);
  for (int i = 0; i < num_cols_; i++) {
    results[i].resize(dim);
  }
  // int run_index = 0;
  // for (int i=0; i<num_rows_; i++) {
  //   for (int j=0; j<row_sizes_[i]; j++) {
  //     int col_index = col_indexes_[run_index];
  //     double value = values_[run_index++];
  //     for (int k = 0; k < dim; k++) {
  //       results[col_index][k] += other_mat[i][k] * value;
  //     }
  //   }
  // }
  for (int k = 0; k < dim; k++) {
    int run_index = 0;
    for (int i = 0; i < num_rows_; i++) {
      for (int j = 0; j < row_sizes_[i]; j++) {
        int col_index = col_indexes_[run_index];
        double value = values_[run_index++];
        results[col_index][k] += other_mat[i][k] * value;
      }
    }
  }
  return std::move(results);
}

void CsrMat::OtherMatMultiplyMatInPlace(const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {
  // auto start = std::chrono::system_clock::now();
  const int dim = other_mat[0].size();
  #pragma omp parallel 
  {
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_cols_; i++) {
      for (int j = 0; j < dim; j++) {
        (*results)[i][j] = 0;
      }
    }
    #pragma omp barrier

    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_rows_; i++) {
      int run_index = row_start_indexes_[i];
      for (int j = 0; j < row_sizes_[i]; j++) {
        const int col_index = col_indexes_[run_index];
        const double value = values_[run_index++];
        for (int k = 0; k < dim; k++) {
          (*results)[col_index][k] += other_mat[i][k] * value;
        }
      }
    }
  }

  // LOG(INFO) << "Csr Other mat times mat: "
  //           << std::chrono::duration_cast<std::chrono::nanoseconds>(
  //                  std::chrono::system_clock::now() - start)
  //                      .count() / (1000.0 * 1000.0 * 1000.0) << " secs";
}

void CsrMat::MatMultiplyScalar(int scalar) {
  int run_index = 0;
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < row_sizes_[i]; j++) {
      values_[run_index] = values_[run_index] * scalar;
      run_index++;
    }
  }
}

void CsrMat::MatSquare() {
  int run_index = 0;
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < row_sizes_[i]; j++) {
      values_[run_index] = values_[run_index] * values_[run_index];
      run_index++;
    }
  }
}

double CsrMat::MatSum() const {
  double sum = 0;
  int run_index = 0;
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < row_sizes_[i]; j++) {
      sum += values_[run_index++];
    }
  }
  return sum;
}

/////////////////////////////////////////////////////////////////////////////////////////
// CsrViMat code
/////////////////////////////////////////////////////////////////////////////////////////

CsrViMat
CsrViMat::CreateCsrViMat(const std::vector<std::vector<io::sparse_pair>> &data,
                         int dim) {
  CsrViMat mat;
  mat.num_rows_ = data.size();
  mat.num_cols_ = dim;
  int nnz = 0;
  std::vector<double> values;
  mat.row_start_indexes_.resize(mat.num_rows_);
  for (int i = 0; i < mat.num_rows_; i++) {
    mat.row_start_indexes_[i] = nnz;
    nnz += data[i].size();
    mat.row_sizes_.push_back(data[i].size());
    for (const auto &pair : data[i]) {
      mat.col_indexes_.push_back(pair.first);
      values.push_back(pair.second);
    }
  }
  // Gets the distinct doubles
  std::unordered_map<double, uint32_t> distinct_doubles =
      distinct_values(values);
  mat.value_index_num_bytes_ = determine_num_bytes(distinct_doubles.size());
  mat.value_indexes_buffer_.resize(mat.value_index_num_bytes_ *
                                   mat.col_indexes_.size());
  mat.values_buffer_.resize(distinct_doubles.size() * sizeof(double));
  mat.num_values_ = distinct_doubles.size();
  // Populates mat.value_indexes_buffer_
  int run_index = 0;
  for (int i = 0; i < mat.num_rows_; i++) {
    for (int j = 0; j < mat.row_sizes_[i]; j++) {
      double value = values[run_index];
      int index = distinct_doubles[value];
      memcpy(&mat.value_indexes_buffer_[run_index * mat.value_index_num_bytes_],
             (char *)&index, mat.value_index_num_bytes_);
      run_index++;
    }
  }
  // Populates mat.values_buffer_
  for (const auto &pair : distinct_doubles) {
    double value = pair.first;
    int index = pair.second;
    memcpy(&mat.values_buffer_[index * sizeof(double)], (char *)&value,
           sizeof(double));
  }
  return mat;
}

bool CsrViMat::operator==(const CsrViMat &right) const {
  if (num_rows_ != right.num_rows_)
    return false;
  if (num_cols_ != right.num_cols_)
    return false;
  if (num_values_ != right.num_values_)
    return false;
  if (value_index_num_bytes_ != right.value_index_num_bytes_)
    return false;
  if (row_sizes_ != right.row_sizes_)
    return false;
  if (col_indexes_ != right.col_indexes_)
    return false;
  if (value_indexes_buffer_ != right.value_indexes_buffer_)
    return false;
  if (values_buffer_ != right.values_buffer_)
    return false;
  return true;
}

std::string CsrViMat::serialize_as_string() {
  std::string ret_string;
  ret_string.resize(size());
  int64_t index = 0;
  memcpy(&ret_string[index], &num_rows_, sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&ret_string[index], &num_cols_, sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&ret_string[index], &num_values_, sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&ret_string[index], &value_index_num_bytes_, sizeof(uint8_t));
  index += sizeof(uint8_t);
  for (int i = 0; i < row_sizes_.size(); i++) {
    memcpy(&ret_string[index], &row_sizes_[i], sizeof(uint32_t));
    index += sizeof(uint32_t);
  }
  for (int i = 0; i < col_indexes_.size(); i++) {
    memcpy(&ret_string[index], &col_indexes_[i], sizeof(uint32_t));
    index += sizeof(uint32_t);
  }
  memcpy(&ret_string[index], &value_indexes_buffer_[0],
         value_indexes_buffer_.size());
  index += value_indexes_buffer_.size();
  memcpy(&ret_string[index], &values_buffer_[0], values_buffer_.size());
  index += values_buffer_.size();
  CHECK(index == size());
  return std::move(ret_string);
}

CsrViMat CsrViMat::CreateFromString(const std::string &data) {
  CsrViMat mat;
  int64_t index = 0;
  memcpy(&mat.num_rows_, &data[index], sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&mat.num_cols_, &data[index], sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&mat.num_values_, &data[index], sizeof(uint32_t));
  index += sizeof(uint32_t);
  memcpy(&mat.value_index_num_bytes_, &data[index], sizeof(uint8_t));
  index += sizeof(uint8_t);
  int nnz = 0;
  mat.row_start_indexes_.resize(mat.num_rows_);
  for (int i = 0; i < mat.num_rows_; i++) {
    mat.row_start_indexes_[i] = nnz;
    int row_size;
    memcpy(&row_size, &data[index], sizeof(uint32_t));
    index += sizeof(uint32_t);
    nnz += row_size;
    mat.row_sizes_.push_back(row_size);
  }
  for (int i = 0; i < nnz; i++) {
    int col_index;
    memcpy(&col_index, &data[index], sizeof(uint32_t));
    index += sizeof(uint32_t);
    mat.col_indexes_.push_back(col_index);
  }

  mat.value_indexes_buffer_.resize(nnz * mat.value_index_num_bytes_);
  memcpy(&mat.value_indexes_buffer_[0], &data[index],
         mat.value_indexes_buffer_.size());
  index += mat.value_indexes_buffer_.size();

  mat.values_buffer_.resize(mat.num_values_ * sizeof(double));
  memcpy(&mat.values_buffer_[0], &data[index], mat.values_buffer_.size());
  index += mat.values_buffer_.size();

  CHECK(index == data.size());

  return std::move(mat);
}

int64_t CsrViMat::size() {
  int64_t mat_size = 3 * sizeof(uint32_t) + 1 * sizeof(uint8_t);
  mat_size += (row_sizes_.size() + col_indexes_.size()) * sizeof(uint32_t) +
              value_indexes_buffer_.size() + values_buffer_.size();
  return mat_size;
}

std::vector<double>
CsrViMat::MatMultiplyVec(const std::vector<double> &vec) const {
  IntegerIterator iter(value_indexes_buffer_.data(), value_index_num_bytes_);
  double *values = (double *)(&values_buffer_[0]);
  std::vector<double> results(num_rows_);
  int run_index = 0;
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < row_sizes_[i]; j++) {
      results[i] += vec[col_indexes_[run_index++]] * values[iter.NextInt()];
    }
  }
  return results;
}

std::vector<double>
CsrViMat::VecMultiplyMat(const std::vector<double> &vec) const {
  IntegerIterator iter(value_indexes_buffer_.data(), value_index_num_bytes_);
  double *values = (double *)(&values_buffer_[0]);
  std::vector<double> results(num_cols_);
  int run_index = 0;
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < row_sizes_[i]; j++) {
      results[col_indexes_[run_index++]] += vec[i] * values[iter.NextInt()];
    }
  }
  return results;
}

std::vector<std::vector<double>> CsrViMat::MatMultiplyOtherMat(
    const std::vector<std::vector<double>> &other_mat) const {
  int dim = other_mat[0].size();
  std::vector<std::vector<double>> results(num_rows_);
  for (int i = 0; i < num_rows_; i++) {
    results[i].resize(dim);
  }
  for (int k = 0; k < dim; k++) {
    IntegerIterator iter(value_indexes_buffer_.data(), value_index_num_bytes_);
    double *values = (double *)(&values_buffer_[0]);
    int run_index = 0;
    for (int i = 0; i < num_rows_; i++) {
      for (int j = 0; j < row_sizes_[i]; j++) {
        int col_index = col_indexes_[run_index++];
        double value = values[iter.NextInt()];
        results[i][k] += value * other_mat[col_index][k];
      }
    }
  }
  return std::move(results);
}

void CsrViMat::MatMultiplyOtherMatInPlace(const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {
  // TODO(fenganli): Fill out the implementation here.

  const int dim = other_mat[0].size();
  // #pragma omp parallel
  // {
  //   #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_rows_; i++) {
      for (int j = 0; j < dim; j++) {
        (*results)[i][j] = 0;
      }
    }
  //  #pragma omp barrier
  //  #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_rows_; i++) {
      int run_index = row_start_indexes_[i];
      for (int j = 0; j < row_sizes_[i]; j++) {
        const int col_index = col_indexes_[run_index];
        const double value = seek_by_index(run_index++);
        for (int k = 0; k < dim; k++) {
          (*results)[i][k] += value * other_mat[col_index][k];
        }
      }
    }
  // }
}

std::vector<std::vector<double>> CsrViMat::OtherMatMultiplyMat(
    const std::vector<std::vector<double>> &other_mat) const {
  int dim = other_mat[0].size();
  std::vector<std::vector<double>> results(num_cols_);
  for (int i = 0; i < num_cols_; i++) {
    results[i].resize(dim);
  }
  for (int k = 0; k < dim; k++) {
    IntegerIterator iter(value_indexes_buffer_.data(), value_index_num_bytes_);
    double *values = (double *)(&values_buffer_[0]);
    int run_index = 0;
    for (int i = 0; i < num_rows_; i++) {
      for (int j = 0; j < row_sizes_[i]; j++) {
        int col_index = col_indexes_[run_index++];
        double value = values[iter.NextInt()];
        results[col_index][k] += other_mat[i][k] * value;
      }
    }
  }
  return std::move(results);
}

void CsrViMat::OtherMatMultiplyMatInPlace(const std::vector<std::vector<double>>& other_mat,
    std::vector<std::vector<double>>* results) const {
  // TODO(fenganli): Fill out the implementation here.

  const int dim = other_mat[0].size();
  #pragma omp parallel 
  {
    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_cols_; i++) {
      for (int j = 0; j < dim; j++) {
        (*results)[i][j] = 0;
      }
    }
    #pragma omp barrier

    #pragma omp for schedule (static) nowait
    for (int i = 0; i < num_rows_; i++) {
      int run_index = row_start_indexes_[i];
      for (int j = 0; j < row_sizes_[i]; j++) {
        const int col_index = col_indexes_[run_index];
        const double value = seek_by_index(run_index++);
        for (int k = 0; k < dim; k++) {
          (*results)[col_index][k] += other_mat[i][k] * value;
        }
      }
    }
  }
}

void CsrViMat::MatMultiplyScalar(int scalar) {
  double *values = (double *)(&values_buffer_[0]);
  for (int i = 0; i < num_values_; i++) {
    values[i] = values[i] * scalar;
  }
}

void CsrViMat::MatSquare() {
  double *values = (double *)(&values_buffer_[0]);
  for (int i = 0; i < num_values_; i++) {
    values[i] = values[i] * values[i];
  }
}

double CsrViMat::MatSum() const {
  double sum = 0;
  IntegerIterator iter(value_indexes_buffer_.data(), value_index_num_bytes_);
  double *values = (double *)(&values_buffer_[0]);
  for (int i = 0; i < num_rows_; i++) {
    for (int j = 0; j < row_sizes_[i]; j++) {
      sum += values[iter.NextInt()];
    }
  }
  return sum;
}

} // namespace core
