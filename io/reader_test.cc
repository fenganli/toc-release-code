#include "csv_reader.h"

#include <iostream>
#include <string>

using namespace std; // using std

DECLARE_double(reader_num_columns);

namespace {

// A helper for making sure that two dense matrixes are the same.
void CheckDenseMatrixEqual(const std::vector<std::vector<double>> &lhs,
                           const std::vector<std::vector<double>> &rhs) {
  CHECK_EQ(lhs.size(), rhs.size());
  for (int i = 0; i < lhs.size(); i++) {
    CHECK_EQ(lhs[i].size(), rhs[i].size()) << "i: " << i;
    for (int j = 0; j < lhs[i].size(); j++) {
      CHECK_EQ(lhs[i][j], rhs[i][j]) << "i: " << i << " j: " << j;
    }
  }
}

// A helper for making sure that two sparse matrixes are the same.
void CheckSparseMatrixEqual(
    const std::vector<std::vector<io::sparse_pair>> &lhs,
    const std::vector<std::vector<io::sparse_pair>> &rhs) {
  CHECK_EQ(lhs.size(), rhs.size());
  for (int i = 0; i < lhs.size(); i++) {
    CHECK_EQ(lhs[i].size(), rhs[i].size()) << "i: " << i;
    for (int j = 0; j < lhs[i].size(); j++) {
      CHECK_EQ(lhs[i][j].first, rhs[i][j].first) << "i: " << i << " j: " << j;
      CHECK_EQ(lhs[i][j].second, rhs[i][j].second) << "i: " << i << " j: " << j;
    }
  }
}

// A helper for making sure that two vectors are the same.
void CheckVectorEqual(const std::vector<int> &lhs,
                      const std::vector<int> &rhs) {
  CHECK_EQ(lhs.size(), rhs.size());
  for (int i = 0; i < lhs.size(); i++) {
    CHECK_EQ(lhs[i], rhs[i]) << "i: " << i;
  }
}

} // namespace

void test_helper(string file_path, int label_index, int mini_batch_size,
                 int test_num);
void Libsvm_test_helper(string file_path, int mini_batch_size);

int main(int argc, char **argv) {
  // test_helper("./test_data/csv.txt", /*label_index=*/-1,
  // /*mini_batch_size=*/2, /*test_num=*/1);
  // test_helper("./test_data/csv.txt", /*label_index=*/1,
  // /*mini_batch_size=*/2, /*test_num=*/2);
  Libsvm_test_helper("./test_data/test_libsvm.txt", /*mini_batch_size=*/2);
  return 0;
}

void Libsvm_test_helper(string file_path, int mini_batch_size) {
  io::LibsvmReader libsvm_reader(file_path);
  // Reduce the number of columns in the reader for testing.
  FLAGS_reader_num_columns = 5;
  CHECK(libsvm_reader.read());

  const vector<vector<double>> expected_dense_mat = {
      {0, 1, 2, 0, 0}, {0, 0, 2, 3, 0}, {0, 0, 0, 3, 4}};
  const vector<vector<double>> dense_mat = *libsvm_reader.get_dense_mat();
  CheckDenseMatrixEqual(expected_dense_mat, dense_mat);

  const vector<vector<io::sparse_pair>> expected_sparse_mat = {
      {io::sparse_pair(1, 1), io::sparse_pair(2, 2)},
      {io::sparse_pair(2, 2), io::sparse_pair(3, 3)},
      {io::sparse_pair(3, 3), io::sparse_pair(4, 4)}};
  const vector<vector<io::sparse_pair>> sparse_mat =
      *libsvm_reader.get_sparse_mat();
  CheckSparseMatrixEqual(expected_sparse_mat, sparse_mat);

  const std::vector<int> labels = *libsvm_reader.get_labels();
  const std::vector<int> expected_labels = {1, 2, 3};
  CheckVectorEqual(labels, expected_labels);

  libsvm_reader.sample_mini_batch(mini_batch_size, /*mini_batch_id=*/0);
  const vector<vector<double>> expected_dense_mini_batch = {{0, 1, 2, 0, 0},
                                                            {0, 0, 2, 3, 0}};
  const vector<vector<double>> dense_mini_batch =
      *libsvm_reader.get_dense_mini_batch();
  CheckDenseMatrixEqual(expected_dense_mini_batch, dense_mini_batch);

  const vector<vector<io::sparse_pair>> sparse_mini_batch =
      *libsvm_reader.get_sparse_mini_batch();
  const vector<vector<io::sparse_pair>> expected_sparse_mini_batch = {
      {io::sparse_pair(1, 1), io::sparse_pair(2, 2)},
      {io::sparse_pair(2, 2), io::sparse_pair(3, 3)}};
  CheckSparseMatrixEqual(expected_sparse_mini_batch, sparse_mini_batch);

  const vector<int> mini_batch_labels = *libsvm_reader.get_mini_batch_labels();
  const vector<int> expected_mini_batch_labels = {1, 2};
  CheckVectorEqual(expected_mini_batch_labels, mini_batch_labels);

  LOG(INFO) << "Tests for file: " << file_path << " passed!";
}

void test_helper(string file_path, int label_index, int mini_batch_size,
                 int test_num) {
  cout << "**************** Test " << test_num << " *********************"
       << endl;
  cout << "should be able to read " << file_path << " successfully!" << endl;
  io::CsvReader csv_reader(file_path, label_index);
  CHECK(csv_reader.read());

  cout << "dense matrix data: " << endl;
  vector<vector<double>> dense_mat = *csv_reader.get_dense_mat();
  for (int i = 0; i < dense_mat.size(); i++) {
    for (int j = 0; j < dense_mat[i].size(); j++) {
      cout << dense_mat[i][j] << " ";
    }
    cout << endl;
  }

  cout << "sparse matrix data: " << endl;
  auto sparse_mat = *csv_reader.get_sparse_mat();
  for (int i = 0; i < sparse_mat.size(); i++) {
    for (int j = 0; j < sparse_mat[i].size(); j++) {
      cout << sparse_mat[i][j].first << ": " << sparse_mat[i][j].second << ", ";
    }
    cout << endl;
  }

  cout << "label_index: " << label_index << endl;
  cout << "labels: " << endl;
  auto labels = *csv_reader.get_labels();
  for (int i = 0; i < labels.size(); i++) {
    cout << labels[i] << endl;
  }

  cout << "sample mini batch size: " << mini_batch_size << endl;
  csv_reader.sample_mini_batch(mini_batch_size);
  cout << "dense mini batch: " << endl;
  vector<vector<double>> dense_mini_batch = *csv_reader.get_dense_mini_batch();
  for (int i = 0; i < dense_mini_batch.size(); i++) {
    for (int j = 0; j < dense_mini_batch[i].size(); j++) {
      cout << dense_mini_batch[i][j] << " ";
    }
    cout << endl;
  }

  cout << "sparse mini batch: " << endl;
  auto sparse_mini_batch = *csv_reader.get_sparse_mini_batch();
  for (int i = 0; i < sparse_mini_batch.size(); i++) {
    for (int j = 0; j < sparse_mini_batch[i].size(); j++) {
      cout << sparse_mini_batch[i][j].first << ": "
           << sparse_mini_batch[i][j].second << ", ";
    }
    cout << endl;
  }

  cout << "mini batch labels: " << endl;
  auto mini_batch_labels = *csv_reader.get_mini_batch_labels();
  for (int i = 0; i < mini_batch_labels.size(); i++) {
    cout << mini_batch_labels[i] << endl;
  }
}
