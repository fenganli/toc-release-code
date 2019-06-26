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
#include "../models/mlp_model.h"
#include "../util/gzip.h"

using namespace std;
using namespace io;

DEFINE_string(file_directory, "../../../data/mnist/mini_batches",
              "the directory that contains all the files");
DEFINE_string(test_file, "../../../data/mnist/mnist_test.csv",
              "the file path to the test file");
DEFINE_int32(num_shards, 1, "the number of file shards we read in each epoch");
DEFINE_string(model, "ann", "the ml model");
DEFINE_string(format, "csr", "the format of the mat");
DEFINE_int32(num_cols, 784, "the number of columns in the dataset");
DEFINE_int32(num_outputs, 10, "the number of outputs in the model");
DEFINE_int32(label_index, 0, "the column index of the label");

DEFINE_int32(num_epoches, 10, " the number of epoches");
DEFINE_double(learning_rate, 0.1, " the learning rate");
namespace {

  void LoadStringAndLabels(fstream* file_shard, std::string* serialized_str,
      std::vector<int>* labels) {
    int num_bytes = 0;
    int num_labels = 0;
    file_shard->read((char*)&num_bytes, sizeof(int));
    serialized_str->resize(num_bytes);
    file_shard->read((char*)&(*serialized_str)[0], num_bytes);
    file_shard->read((char*)&num_labels, sizeof(int));
    labels->resize(num_labels);
    file_shard->read((char*)&(*labels)[0], sizeof(int) * num_labels);
  }

}  // namespace

enum CompressionMethod { NONE = 0, GZIP = 1, SNAPPY = 2 };

template<typename T>
void CreateMatAndLabelsFromFileShard(const string& filename, CompressionMethod compression_method,
    std::vector<T>* mats, std::vector<std::vector<int>>* mats_labels) {
  fstream file_shard(filename, ios::in | ios::binary);
  CHECK(file_shard.is_open());
  int num_mini_batches;
  file_shard.read((char *)&num_mini_batches, sizeof(int));
  string serialized_str;
  string uncompressed_str;
  for (int i = 0; i < num_mini_batches; i++) {
    std::vector<int> labels;
    LoadStringAndLabels(&file_shard, &serialized_str, &labels);
    if (compression_method == NONE) {
      mats->push_back(T::CreateFromString(serialized_str));
    } else if (compression_method == GZIP) {
      mats->push_back(T::CreateFromString(Gzip::decompress(serialized_str)));
    } else if (compression_method == SNAPPY) {
      snappy::Uncompress(serialized_str.data(), serialized_str.size(), &uncompressed_str);
      mats->push_back(T::CreateFromString(uncompressed_str));
    }
    mats_labels->push_back(std::move(labels));
  }
}

template <typename T1, typename T2>
void TrainModel(const std::vector<T1>& mats,
    const std::vector<std::vector<int>>& labels, T2 *model) {
  const int num_mini_batches = mats.size();
  for (int i = 0; i < num_mini_batches; i++) {
    model->TrainModel(mats[i], labels[i]);
  }
}

template <typename T1, typename T2>
void TestModel(const T1 &mat, const T2 &model, const vector<int> &labels) {
  LOG(INFO) << "loss: " << model.ComputeLoss(mat, labels);
  LOG(INFO) << "accuracy: " << model.ComputeAccuracy(mat, labels);
}

int main(int argc, char **argv) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  model::MlpModel mlp_model({FLAGS_num_cols, 200, 50, FLAGS_num_outputs},
                            FLAGS_learning_rate, /*mini_batch_size=*/250);

  for (int epoch = 0; epoch < FLAGS_num_epoches; epoch++) {
    auto epoch_start = chrono::system_clock::now();
    for (int shard_id = 0; shard_id < FLAGS_num_shards; shard_id++) {
      const string file_name = FLAGS_file_directory + "/file-" +
                               to_string(shard_id) + "." + FLAGS_format;
      int64_t read_nano_secs = 0;
      int64_t train_nano_secs = 0;
      std::vector<std::vector<int>> mats_labels;
      if (FLAGS_format == "csr") {
        auto read_start = std::chrono::system_clock::now();
        std::vector<core::CsrMat> mats;
        CreateMatAndLabelsFromFileShard(file_name, NONE, &mats, &mats_labels);
        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();
        TrainModel(mats, mats_labels, &mlp_model);
        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();
      } else if (FLAGS_format == "csrvi") {
        auto read_start = std::chrono::system_clock::now();
        std::vector<core::CsrViMat> mats;
        CreateMatAndLabelsFromFileShard(file_name, NONE, &mats, &mats_labels);

        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();
        TrainModel(mats, mats_labels, &mlp_model);

        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();
      } else if (FLAGS_format == "dvi") {
        auto read_start = std::chrono::system_clock::now();
        std::vector<core::DviMat> mats;
        CreateMatAndLabelsFromFileShard(file_name, NONE, &mats, &mats_labels);

        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();

        TrainModel(mats, mats_labels, &mlp_model);

        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();
      } else if (FLAGS_format == "gzip") {
        auto read_start = std::chrono::system_clock::now();

        std::vector<core::Mat> mats;
        CreateMatAndLabelsFromFileShard(file_name, GZIP, &mats, &mats_labels);

        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();

        TrainModel(mats, mats_labels, &mlp_model);

        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();
      } else if (FLAGS_format == "snappy") {
        auto read_start = std::chrono::system_clock::now();
        std::vector<core::Mat> mats;
        CreateMatAndLabelsFromFileShard(file_name, SNAPPY, &mats, &mats_labels);

        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();

        TrainModel(mats, mats_labels, &mlp_model);

        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();
      } else if (FLAGS_format == "mat") {
        auto read_start = std::chrono::system_clock::now();
        std::vector<core::Mat> mats;
        CreateMatAndLabelsFromFileShard(file_name, NONE, &mats, &mats_labels);

        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();

        TrainModel(mats, mats_labels, &mlp_model);

        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();
      } else if (FLAGS_format == "toc") {
        auto read_start = std::chrono::system_clock::now();

        std::vector<core::CompressedMat> mats;
        CreateMatAndLabelsFromFileShard(file_name, NONE, &mats, &mats_labels);

        read_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - read_start).count();
        auto train_start = std::chrono::system_clock::now();

        TrainModel(mats, mats_labels, &mlp_model);

        train_nano_secs = chrono::duration_cast<chrono::nanoseconds>(
            chrono::system_clock::now() - train_start).count();
      }
      LOG(INFO) << "epoch_id: " << epoch;
      LOG(INFO) << "shard_id: " << shard_id;
      LOG(INFO) << "reading_time: " << read_nano_secs / (1000.0 * 1000.0 * 1000.0) << " secs";
      LOG(INFO) << "training_time: " << train_nano_secs / (1000.0 * 1000.0 * 1000.0) << " secs";
      LOG(INFO) << "total_time: " << (read_nano_secs + train_nano_secs) / (1000.0 * 1000.0 * 1000.0) << " secs";
      if (FLAGS_test_file != "") {
        // Read the test data
        CsvReader reader(FLAGS_test_file, FLAGS_label_index);
        reader.read();
        core::Mat mat = core::Mat::CreateMat(*reader.get_dense_mat());
        std::vector<int> labels = *reader.get_labels();
        TestModel(mat, mlp_model, labels);
      }
    }
  }
  return 0;
}
