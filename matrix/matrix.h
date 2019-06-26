#ifndef _MATRIX_MATRIX_H_
#define _MATRIX_MATRIX_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../compress/compress_data.h"
#include "../compress/decompress_data.h"
#include "../io/reader.h"

#define likely(x) __builtin_expect(!!(x), 1)

namespace core {

// A class that can iterate through the encoded integers. Caller must ensure
// that they never
// iterate out of the bound.
class IntegerIterator {
public:
  IntegerIterator(const char *buffer, int8_t int_byte_num)
      : buffer_(buffer), original_buffer_(buffer), int_byte_num_(int_byte_num) {
  }

  // TODO(fenganl): Consider make a bulk loading API of this.
  inline uint32_t NextInt() {
    uint32_t ret_int = 0;
    switch (int_byte_num_) {
    case 1:
      ret_int = *((uint8_t *)(buffer_));
      break;
    case 2:
      ret_int = *((uint16_t *)(buffer_));
      break;
    case 3:
      ret_int = *((uint32_t *)(buffer_)) & 0xFFFFFF;
      break;
    case 4:
      ret_int = *((uint32_t *)(buffer_));
    }
    buffer_ += int_byte_num_;
    return ret_int;
  }

  void GetInts(uint32_t *int_array, int num_ints) {
    int index = 0;
    if (int_byte_num_ == 1) {
      for (int i = 0; i < num_ints; i++) {
        int_array[i] = *((uint8_t *)(buffer_));
        buffer_++;
      }
    } else if (int_byte_num_ == 2) {
      for (int i = 0; i < num_ints; i++) {
        int_array[i] = *((uint16_t *)(buffer_));
        buffer_ += 2;
      }
    } else if (int_byte_num_ == 3) {
      for (int i = 0; i < num_ints; i++) {
        int_array[i] = *((uint32_t *)(buffer_)) & 0xFFFFFF;
        buffer_ += 3;
      }
    } else if (int_byte_num_ == 4) {
      for (int i = 0; i < num_ints; i++) {
        int_array[i] = *((uint32_t *)(buffer_));
        buffer_ += 4;
      }
    } else {
      CHECK(false) << "int_byte_num_ should be from 1 to 4";
    }
  }

  void Reset() { buffer_ = original_buffer_; }

private:
  // pointer to the buffer. Not owned.
  const char *buffer_;
  // pointer to the original buffer. Not owned.
  const char *original_buffer_;
  // # of bytes used to encode the integer.
  int8_t int_byte_num_;
};

// Used for matrix times vec operation
struct DictNode {
  uint32_t col_idx;
  double col_value;
  double value;
} __attribute__((packed));

// Used for mat_times_other_mat and other_mat_times_mat operation.
struct DictNode3 {
  // index of the first double value
  int first_idx;
  // index of the last double value
  int last_idx;
  // index of the parent dict node
  int parent_idx;
};

// Compresses matrix using only sparse encoding and logical encoding. Physical
// encoding is not used.
class LogicalCompressedMat {
public:
  // Creates a logical compressed matrix from the <sparse_mat> with column size
  // <dim>. If <init_pairs> is not empty, uses it to initialize the compressor.
  static LogicalCompressedMat CreateLogicalCompressedMat(
      const std::vector<std::vector<io::sparse_pair>> &sparse_matrix,
      const std::vector<io::sparse_pair> &init_pairs, int dim);

  // Creates a logical compressed matrix from the serialized <data>.
  static LogicalCompressedMat CreateFromString(const std::string &data);

  // Returns the number of bytes used to store the logical compressed matrix.
  int64_t size();

  bool operator==(const LogicalCompressedMat &right) const;

  void LightDecompression() const;

  std::string serialize_as_string();

  // Mat multiplies vec.
  std::vector<double> MatMultiplyVec(const std::vector<double> &vec) const;
  // Vec multiplies mat.
  std::vector<double> VecMultiplyMat(const std::vector<double> &vec) const;
  // Current mat multiplies other mat.
  std::vector<std::vector<double>> MatMultiplyOtherMat(
      const std::vector<std::vector<double>> &other_mat) const;
  // Logically speaking, result = other_mat * mat. However, other_mat and
  // result are both stored in a columnar way (i.e., transposed).
  std::vector<std::vector<double>>
  OtherMatMultiplyMat(const std::vector<std::vector<double>> &other_mat) const;

  // Current mat multiplies other mat.
  void MatMultiplyOtherMatInPlace(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // Logically speaking, result = other_mat * mat. However, other_mat and
  // result are both stored in a columnar way (i.e., transposed).
  void OtherMatMultiplyMatInPlace(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // Multiplies all the elements in the matrix with a scalar.
  void MatMultiplyScalar(int scalar);
  // Square all the elements in the matrix.
  void MatSquare();
  // Computes the sum of all the elements in the matrix.
  double MatSum() const;

  int get_num_rows() const { return num_rows_; }

  int get_num_cols() const { return num_cols_; }

private:
  mutable bool light_decompressed_ = false;

  void MatMultiplyOtherMatImpl(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  void OtherMatMultiplyMatImpl(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // The number of init pairs in the compressed matrix.
  int num_init_pairs_ = 0;
  // The number of rows in the matrix.
  int num_rows_ = 0;
  // The number of cols in the matrix.
  int num_cols_ = 0;
  // The number of dictionary nodes.
  int num_dict_nodes_ = 0;
  // Column indexes in the initial pairs.
  std::vector<int> init_column_indexes_;
  // Values in the initial pairs.
  std::vector<double> init_values_;
  // Row sizes.
  std::vector<uint32_t> row_sizes_;
  // Codes.
  std::vector<uint32_t> codes_;

  // The start indexes for each row.
  static uint32_t* row_start_indexes_;
  // The dict nodes stuff.
  static DictNode3* dict_nodes_;
  // The cached values of dict nodes.
  static double* dict_nodes_values_;
  // This is the general used buffer.
  static char * buffer_;
  static double* results_buffer_;
  static double* other_mat_buffer_;
  static bool* preprocessed_init_pairs_;
  static uint32_t* init_pairs_row_counts_;
  static uint32_t* init_pairs_row_indexes_;
};

// A class that represents the compressed matrix. We now only support double
// type. This is the TOC matrix object.
class CompressedMat {
public:
  // Creates a compressed matrix from the <sparse_mat> with column size <dim>.
  // If <init_pairs> is not empty, uses it to initialize the compressor.
  static CompressedMat CreateCompressedMat(
      const std::vector<std::vector<io::sparse_pair>> &sparse_matrix,
      const std::vector<io::sparse_pair> &init_pairs, int dim);

  static CompressedMat CreateFromString(const std::string &data);

  // Returns the number of bytes used to store the compressed mat.
  int64_t size();

  bool operator==(const CompressedMat &right) const;

  std::string serialize_as_string();

  // Mat multiplies vec
  std::vector<double> MatMultiplyVec(const std::vector<double> &vec) const;
  // Vec multiplies mat
  std::vector<double> VecMultiplyMat(const std::vector<double> &vec) const;
  // Current mat multiplies other mat.
  std::vector<std::vector<double>>
  MatMultiplyOtherMat(const std::vector<std::vector<double>> &other_mat) const;
  // Logically speaking, result = other_mat * mat. However, other_mat and
  // result are both stored in a columnar way.
  std::vector<std::vector<double>>
  OtherMatMultiplyMat(const std::vector<std::vector<double>> &other_mat) const;

  // Current mat multiplies other mat.
  void MatMultiplyOtherMatInPlace(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // Logically speaking, result = other_mat * mat. However, other_mat and
  // result are both stored in a columnar way (i.e., transposed).
  void OtherMatMultiplyMatInPlace(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // Multiply all the elements in the matrix with a scalar.
  void MatMultiplyScalar(int scalar);
  // Square all the elements in the matrix.
  void MatSquare();
  // Computes the sum of all the elements in the matrix.
  double MatSum() const;

  // Decompress the full data.
  std::vector<std::vector<io::sparse_pair>> Decompression();

  // Light decompression for matrix operations.
  void LightDecompression() const;

  int get_num_rows() const { return num_rows_; }

  int get_num_cols() const { return num_cols_; }

private:
  // Physical decoding the values from buffers and store them.
  void physical_decoding() const;

  void MatMultiplyOtherMatImpl(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  void OtherMatMultiplyMatImpl(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  int8_t code_num_byte_;
  int8_t column_index_num_byte_;
  int8_t value_index_num_byte_;
  mutable bool light_decompressed_ = false;
  // The number of init pairs in the compressed matrix.
  int num_init_pairs_ = 0;
  // The number of double values.
  int num_double_values_ = 0;
  // The number of rows in the matrix.
  int num_rows_ = 0;
  // The number of cols in the matrix.
  int num_cols_ = 0;
  // The number of dictionary nodes.
  int num_dict_nodes_ = 0;
  // The buffer that stores the column indexes in the initial pairs.
  std::string init_column_indexes_buffer_;
  // The buffer that stores the indexes pointing to the double values in the
  // initial pairs.
  std::string init_value_indexes_buffer_;
  // The buffer that stores the double values.
  std::string values_buffer_;
  // The buffer that stores the # of codes for each row in the matrix.
  std::string row_sizes_buffer_;
  // The buffer that stores the codes.
  std::string codes_buffer_;

  // Below are just static allocated memory and variables. We do this because
  // we want to save the time we spend on allocating memories.

  // The init column indexes.
  static uint32_t* init_column_indexes_;
  // The init value indexes_;
  static uint32_t* init_value_indexes_;
  // How many codes in each row.
  static uint32_t* row_sizes_;
  // The start indexes for each row.
  static uint32_t* row_start_indexes_;
  // The codes.
  static uint32_t* codes_;
  // The dict nodes stuff.
  static DictNode3* dict_nodes_;
  // The cached values of dict nodes.
  static double* dict_nodes_values_;
  // This is the general used buffer.
  static char * buffer_;
  static double* results_buffer_;
  static double* other_mat_buffer_;
  static bool* preprocessed_init_pairs_;
  static uint32_t* init_pairs_row_counts_;
  static uint32_t* init_pairs_row_indexes_;
};

// A class that represents the dense matrix. We now only support double type.
class Mat {
public:
  // Creates the mat from the 2d std::vector.
  static Mat CreateMat(std::vector<std::vector<double>> data);

  // Creates the mat from the serialized string.
  static Mat CreateFromString(const std::string &data);

  int64_t size();
  std::string serialize_as_string();
  bool operator==(const Mat &right) const;

  // Mat multiplies vec operation
  std::vector<double> MatMultiplyVec(const std::vector<double> &vec) const;
  // Vec multiplies mat operation
  std::vector<double> VecMultiplyMat(const std::vector<double> &vec) const;
  // Current mat multiplies other mat.
  std::vector<std::vector<double>>
  MatMultiplyOtherMat(const std::vector<std::vector<double>> &other_mat) const;
  // Logically speaking, result = other_mat * mat. However, other_mat and
  // result are both stored in a columnar way.
  std::vector<std::vector<double>>
  OtherMatMultiplyMat(const std::vector<std::vector<double>> &other_mat) const;

  // Current mat multiplies other mat.
  void MatMultiplyOtherMatInPlace(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // Logically speaking, result = other_mat * mat. However, other_mat and
  // result are both stored in a columnar way.
  void OtherMatMultiplyMatInPlace(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // Multiply all the elements in the matrix with a scalar.
  void MatMultiplyScalar(int scalar);
  // Square all the elements in the matrix.
  void MatSquare();
  // Returns the sum of all the elements in the matrix
  double MatSum() const;

  int get_num_rows() const { return num_rows_; }

  int get_num_cols() const { return num_cols_; }

  // Gets the underlying data.
  const std::vector<std::vector<double>> *get_data() const { return &data_; }

  // Gets the mutable underlying data.
  std::vector<std::vector<double>>* get_mutable_data() {
    return &data_;
  }

  void resize(int num_row) {
    num_rows_ = num_row;
    data_.resize(num_rows_);
    for (int i = 0; i < num_rows_; i++) {
      data_[i].resize(num_cols_);
    }
  }

private:
  std::vector<std::vector<double>> data_;
  uint32_t num_rows_;
  uint32_t num_cols_;

  static double * DATA_BUFFER;
  static double * OTHER_MAT_BUFFER;
};

// Uses value indexing to compress the matrix.
class DviMat {
public:
  static DviMat CreateDviMat(const std::vector<std::vector<double>> &data);
  static DviMat CreateFromString(const std::string &data);

  int64_t size();

  std::string serialize_as_string();

  bool operator==(const DviMat &right) const;

  // Matrix multiplies vector
  std::vector<double> MatMultiplyVec(const std::vector<double> &vec) const;
  // Vector multiplies matrix
  std::vector<double> VecMultiplyMat(const std::vector<double> &vec) const;
  // Current mat multiplies other mat.
  std::vector<std::vector<double>>
  MatMultiplyOtherMat(const std::vector<std::vector<double>> &other_mat) const;
  // Logically speaking, result = other_mat * mat. However, other_mat and
  // result are both stored in a columnar way.
  std::vector<std::vector<double>>
  OtherMatMultiplyMat(const std::vector<std::vector<double>> &other_mat) const;

  // Current mat multiplies other mat.
  void MatMultiplyOtherMatInPlace(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // Logically speaking, result = other_mat * mat. However, other_mat and
  // result are both stored in a columnar way.
  void OtherMatMultiplyMatInPlace(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // Multiply all the elements in the matrix with a scalar.
  void MatMultiplyScalar(int scalar);
  // Square all the elements in the matrix.
  void MatSquare();
  // Sum of all the elements in the matrix.
  double MatSum() const;

  int get_num_rows() const { return num_rows_; }

  int get_num_cols() const { return num_cols_; }

  double seek_by_indexes(int row, int col) const {
    uint32_t ret_index = 0;
    switch (index_num_bytes_) {
    case 1:
      ret_index = *((uint8_t *)(indexes_buffer_.data() + row * num_cols_ + col));
      break;
    case 2:
      ret_index = *((uint16_t *)(indexes_buffer_.data() + 2 * (row * num_cols_ + col)));
      break;
    case 3:
      ret_index = *((uint32_t *)(indexes_buffer_.data() + 3 * (row * num_cols_ + col))) & 0xFFFFFF;
      break;
    case 4:
      ret_index = *((uint32_t *)(indexes_buffer_.data() + 4 * (row * num_cols_ + col)));
    }

    return *((double *)(values_buffer_.data() + sizeof(double) * ret_index));
  }

private:
  uint32_t num_rows_;
  uint32_t num_cols_;
  uint32_t num_values_;
  uint8_t index_num_bytes_;
  std::string indexes_buffer_;
  std::string values_buffer_;

  static double * DATA_BUFFER;
  static double * OTHER_MAT_BUFFER;
};

// A class that represents the sparse matrix in CSR format. We now only support
// double type.
class CsrMat {
public:
  // Creates a CsrMat from <data>. <dim> specifies the num_cols_ in the matrix.
  static CsrMat
  CreateCsrMat(const std::vector<std::vector<io::sparse_pair>> &data, int dim);
  // Creates a CsrMat from serialized <data>.
  static CsrMat CreateFromString(const std::string &data);

  int64_t size();
  std::string serialize_as_string();
  bool operator==(const CsrMat &right) const;

  // Mat multiplies vector
  std::vector<double> MatMultiplyVec(const std::vector<double> &vec) const;
  // Vector multiplies mat
  std::vector<double> VecMultiplyMat(const std::vector<double> &vec) const;
  // Matrix multiplies other mat
  std::vector<std::vector<double>>
  MatMultiplyOtherMat(const std::vector<std::vector<double>> &other_mat) const;
  // Logically speaking, result = other_mat * mat. However, other_mat and
  // result are both stored in a columnar way.
  std::vector<std::vector<double>>
  OtherMatMultiplyMat(const std::vector<std::vector<double>> &other_mat) const;

  // Matrix multiplies other mat
  void MatMultiplyOtherMatInPlace(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // Logically speaking, result = other_mat * mat. However, other_mat and
  // result are both stored in a columnar way.
  void OtherMatMultiplyMatInPlace(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // Multiply all the elements in the matrix with a scalar.
  void MatMultiplyScalar(int scalar);
  // Square all the elements in the matrix.
  void MatSquare();
  // The sum of the elements.
  double MatSum() const;

  int get_num_rows() const { return num_rows_; }

  int get_num_cols() const { return num_cols_; }

private:
  uint32_t num_rows_;
  uint32_t num_cols_;
  std::vector<uint32_t> row_sizes_;
  std::vector<uint32_t> row_start_indexes_;
  std::vector<uint32_t> col_indexes_;
  std::vector<double> values_;
};

// Uses value indexing to compressed the csr mat.
class CsrViMat {
  // Creates a CsrViMat from <data>. <dim> specifies the num_cols in the matrix.
public:
  static CsrViMat
  CreateCsrViMat(const std::vector<std::vector<io::sparse_pair>> &data,
                 int dim);
  static CsrViMat CreateFromString(const std::string &data);

  bool operator==(const CsrViMat &right) const;
  std::string serialize_as_string();
  int64_t size();

  // Mat multiplies vector
  std::vector<double> MatMultiplyVec(const std::vector<double> &vec) const;
  // Vector multiplies mat
  std::vector<double> VecMultiplyMat(const std::vector<double> &vec) const;
  // Matrix multiplies other mat
  std::vector<std::vector<double>>
  MatMultiplyOtherMat(const std::vector<std::vector<double>> &other_mat) const;
  // Logically speaking, result = other_mat * mat. However, other_mat and
  // result are both stored in a columnar way.
  std::vector<std::vector<double>>
  OtherMatMultiplyMat(const std::vector<std::vector<double>> &other_mat) const;

  // Matrix multiplies other mat
  void MatMultiplyOtherMatInPlace(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // Logically speaking, result = other_mat * mat. However, other_mat and
  // result are both stored in a columnar way.
  void OtherMatMultiplyMatInPlace(const std::vector<std::vector<double>>& other_mat,
      std::vector<std::vector<double>>* results) const;

  // Multiply all the elements in the matrix with a scalar.
  void MatMultiplyScalar(int scalar);
  // Square all the elements in the matrix.
  void MatSquare();
  // The sum of all the elements in the matrix.
  double MatSum() const;

  int get_num_rows() const { return num_rows_; }

  int get_num_cols() const { return num_cols_; }

  double seek_by_index(int value_index) const {
    uint32_t ret_index = 0;
    switch (value_index_num_bytes_) {
    case 1:
      ret_index = *((uint8_t *)(value_indexes_buffer_.data() + value_index));
      break;
    case 2:
      ret_index = *((uint16_t *)(value_indexes_buffer_.data() + 2 * value_index));
      break;
    case 3:
      ret_index = *((uint32_t *)(value_indexes_buffer_.data() + 3 * value_index)) & 0xFFFFFF;
      break;
    case 4:
      ret_index = *((uint32_t *)(value_indexes_buffer_.data() + 4 * value_index));
    }

    return *((double *)(values_buffer_.data() + sizeof(double) * ret_index));
  }

private:
  uint32_t num_rows_;
  uint32_t num_cols_;
  uint32_t num_values_;
  uint8_t value_index_num_bytes_;
  std::vector<uint32_t> row_sizes_;
  std::vector<uint32_t> row_start_indexes_;
  std::vector<uint32_t> col_indexes_;
  std::string value_indexes_buffer_;
  std::string values_buffer_;
};

} // namespace core

#endif // _MATRIX_MATRIX_H_
