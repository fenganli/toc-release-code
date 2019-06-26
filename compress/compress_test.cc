#include <chrono>
#include <ctime>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <snappy.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../io/reader.h"
#include "../matrix/matrix.h"
#include "../util/gzip.h"
#include "compress_data.h"
#include "decompress_data.h"

using namespace std;
using namespace io;

DEFINE_string(file, "test_data/csv.txt", "the file path of the whole file");

int main(int argc, char **argv) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  // read and consolidate data
  CsvReader csv_reader(FLAGS_file, /*label_index=*/-1);
  csv_reader.read();
  vector<vector<sparse_pair>> sparse_mat = *csv_reader.get_sparse_mat();
  int num_cols = csv_reader.get_num_cols();
  compress::LZWCompresser<io::sparse_pair, io::pair_hash<int, double>> encoder(
      sparse_mat, num_cols);
  CHECK(encoder.compress_data());
  compress::LZWDecompresser<io::sparse_pair, io::pair_hash<int, double>>
      decoder(encoder.get_codes(), encoder.get_init_data(), num_cols,
              encoder.get_seq_num());
  CHECK(decoder.decompress_data());
  const auto encode_data = encoder.get_data();
  const auto decode_data = decoder.get_data();
  LOG(INFO) << "Encoder data: ";
  for (int i = 0; i < encode_data.size(); i++) {
    for (int j = 0; j < encode_data[i].size(); j++) {
      cout << encode_data[i][j].first << ":"
           << encode_data[i][j].second << ", ";
    }
    cout << endl;
  }
  LOG(INFO) << "Decoder data: ";

  for (int i = 0; i < decode_data.size(); i++) {
    for (int j = 0; j < decode_data[i].size(); j++) {
      cout << decode_data[i][j].first << ":"
           << decode_data[i][j].second << ", ";
    }
    cout << endl;
  }
  CHECK(encode_data == decode_data);
  return 0;
}
