#ifndef __MODEL_MLP_MODEL__
#define __MODEL_MLP_MODEL__

#include "../matrix/matrix.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <numeric>
#include <random>
#include <stdlib.h>
#include <vector>

namespace model {

// Stores all the information about the connection between two layers.
// Say there are m nodes in the left layer and n nodes in the right layer.
// Then dimensions of 'weigths' will be (m, n). The dimension of 'offsets'
// will be (n).
class LayerConnection {
public:
  LayerConnection(int left_layer_size, int right_layer_size, int mini_batch_size)
      : left_layer_size_(left_layer_size), right_layer_size_(right_layer_size),
        mini_batch_size_(mini_batch_size) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0, 1.0);
    // Init the weights.
    weights_.resize(left_layer_size);
    weights_updates_.resize(left_layer_size);
    for (int i = 0; i < weights_.size(); i++) {
      weights_[i].resize(right_layer_size);
      weights_updates_[i].resize(right_layer_size);
      for (int j = 0; j < weights_[i].size(); j++) {
        weights_[i][j] = distribution(generator);
      }
    }
    offsets_.resize(right_layer_size);
    offsets_updates_.resize(right_layer_size);
    for (int i = 0; i < right_layer_size; i++) {
      offsets_[i] = distribution(generator);
    }

    // Preallocate the memory for outputs and errors. This is just for the
    // memory management.
    std::vector<std::vector<double>> outputs(mini_batch_size);
    std::vector<std::vector<double>> errors(mini_batch_size);
    for (int i = 0; i < mini_batch_size; i++) {
      outputs[i].resize(right_layer_size);
      errors[i].resize(right_layer_size);
    }
    outputs_ = core::Mat::CreateMat(std::move(outputs));
    right_layer_input_errors_ = core::Mat::CreateMat(std::move(errors));
  }

  // Computes the right layer output from the input layer.
  // If 'softmax' is true, use softmax as the activation function. Otherwise,
  // use sigmoid as the activation function.
  template <typename T> void ComputeOutputs(const T &mat, bool softmax) const {
    if (outputs_.get_num_rows() != mat.get_num_rows()) {
      outputs_.resize(mat.get_num_rows());
    }
    std::vector<std::vector<double>>* outputs = outputs_.get_mutable_data();
    mat.MatMultiplyOtherMatInPlace(weights_, outputs);
    if (softmax) {
      for (int i = 0; i < outputs->size(); i++) {
        double sum = 0;
        for (int j = 0; j < (*outputs)[i].size(); j++) {
          (*outputs)[i][j] = exp(offsets_[j] + (*outputs)[i][j]);
          sum += (*outputs)[i][j];
        }
        double sum_inverse = 1.0/sum;
        for (int j = 0; j < (*outputs)[i].size(); j++) {
          (*outputs)[i][j] = (*outputs)[i][j] * sum_inverse;
        }
      }
      return;
    }

    for (int i = 0; i < outputs->size(); i++) {
      for (int j = 0; j < (*outputs)[i].size(); j++) {
        (*outputs)[i][j] = sigmoid(offsets_[j] + (*outputs)[i][j]);
      }
    }
  }

  const core::Mat *GetOutputs() { return &outputs_; }

  std::vector<std::vector<double>>* GetMutableInputErrors() {
    return right_layer_input_errors_.get_mutable_data();
  }

  // Computes the gradients of 'weights_' and 'offsets_' using
  // 'left_layer_output'. Also updates 'weights_' and 'offset_' using the
  // computed gradients and 'learning_rate'.
  //
  // This function must be called after ComputeOutputs. Otherwise the
  // behaviour is undefined.
  template <typename T>
  void UpdateWeights(const T &left_layer_output, double learning_rate) {
    const int num_rows = mini_batch_size_;
    const std::vector<std::vector<double>> *right_layer_outputs = outputs_.get_data();
    const std::vector<std::vector<double>> *right_layer_input_errors =
      right_layer_input_errors_.get_data();

    // Update the weights.
    const double num_rows_inverse = 1.0 / num_rows;
    // Compute the updates to the weights.
    left_layer_output.OtherMatMultiplyMatInPlace(*right_layer_input_errors, &weights_updates_);

    for (int i = 0; i < offsets_updates_.size(); i++) {
      offsets_updates_[i] = 0;
    }

    #pragma omp parallel
    {
      #pragma omp for nowait
      for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < offsets_updates_.size(); j++) {
          offsets_updates_[j] += (*right_layer_input_errors)[i][j];
        }
      }
      #pragma omp for nowait
      for (int i = 0; i < weights_updates_.size(); i++) {
        for (int j = 0; j < weights_updates_[i].size(); j++) {
          weights_[i][j] -= weights_updates_[i][j] * learning_rate * num_rows_inverse;
        }
      }
    }
    // Update the offsets.
    for (int i = 0; i < offsets_.size(); i++) {
      offsets_[i] -= offsets_updates_[i] * learning_rate * num_rows_inverse;
    }
  }

  // Updates the 'left_layer_input_errors'.
  void BackPropogateErrors(const core::Mat& left_layer_output,
      std::vector<std::vector<double>>* left_layer_input_errors) {
    const int num_rows = mini_batch_size_;
    const std::vector<std::vector<double>>* left_layer_output_vectors =
      left_layer_output.get_data();
    const std::vector<std::vector<double>> *right_layer_input_errors =
      right_layer_input_errors_.get_data();
    #pragma omp parallel
    {     
      // Finally, we need to compute the left layer input errors.
      // This part could have been represented by a matrix operation. But that
      // requires transposing right_layer_input_errors, thus we just compute this
      // using some for loops.
      #pragma omp for schedule (static) nowait
      for (int i = 0; i < num_rows; i++) {
        const double *x = &(*right_layer_input_errors)[i][0];
        for (int j = 0; j < left_layer_size_; j++) {
          const double *y = &weights_[j][0];
          double& value = (*left_layer_input_errors)[i][j];
          value = 0;
          #pragma omp simd reduction(+: value)
          for (int k = 0; k < right_layer_size_; ++k) {
            value += x[k] * y[k];
          }
          const double output = (*left_layer_output_vectors)[i][j];
          value = value * output * (1 - output);
        }
      }
    }
  }

private:
  inline static double sigmoid(double value) { return 1.0 / (1 + exp(-value)); }

  int left_layer_size_;
  int right_layer_size_;
  // This is the mini_batch size. This is recorded here just for better memory
  // management.
  int mini_batch_size_;
  // Note the weights are transposed.
  std::vector<std::vector<double>> weights_;
  std::vector<std::vector<double>> weights_updates_;
  std::vector<double> offsets_;
  std::vector<double> offsets_updates_;
  // The outputs of the right layer.
  mutable core::Mat outputs_;
  // The errors of the right layer input.
  core::Mat right_layer_input_errors_;
};

// Ann model
class MlpModel {
public:
  MlpModel(std::vector<int> layer_sizes, double learning_rate, int mini_batch_size)
      : layer_sizes_(std::move(layer_sizes)), learning_rate_(learning_rate),
      mini_batch_size_(mini_batch_size) {
    // Initialize the layer_connections.
    for (int i = 0; i < layer_sizes_.size() - 1; i++) {
      layer_connections_.push_back(std::make_unique<LayerConnection>(
          layer_sizes_[i], layer_sizes_[i + 1], mini_batch_size));
    }
  }

  // Update the models using <mat> and <labels>. <labels> are supposed to be 0
  // or 1.
  template <typename T>
  void TrainModel(const T &mat, const std::vector<int> &labels) {
    // auto train_start = std::chrono::system_clock::now();

    const int num_rows = mat.get_num_rows();
    const std::vector<std::vector<double>> *predict_outputs =
        ForwardComputing(mat);

    const int output_layer_size = layer_sizes_[layer_sizes_.size() - 1];

    // Computes the output layer input errors.
    std::vector<std::vector<double>>* output_layer_input_errors = 
      layer_connections_[layer_connections_.size()-1]->GetMutableInputErrors();
    if (output_layer_size == 1) {
      for (int i = 0; i < num_rows; i++) {
        (*output_layer_input_errors)[i][0] = (*predict_outputs)[i][0] - labels[i];
      }
    } else {
      for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < output_layer_size; j++) {
          (*output_layer_input_errors)[i][j] = (*predict_outputs)[i][j] -
            (labels[i] == j ? 1 : 0);
        }
      }
    }

    // Actually do the back propogation.
    for (int i = layer_connections_.size() - 1; i >= 0; i--) {
      if (i != 0) {
        layer_connections_[i]->UpdateWeights(
            *layer_connections_[i-1]->GetOutputs(), learning_rate_);
        layer_connections_[i]->BackPropogateErrors(
            *layer_connections_[i-1]->GetOutputs(),
            layer_connections_[i-1]->GetMutableInputErrors());
      } else {
        layer_connections_[i]->UpdateWeights(mat, learning_rate_);
      }
    }
  }

  // Computes the average loss using <mat> and <weights_>.
  template <typename T>
  double ComputeLoss(const T &mat, const std::vector<int> &labels) const {
    double loss = 0;
    const int num_rows = mat.get_num_rows();
    const std::vector<std::vector<double>> *predict_outputs =
        ForwardComputing(mat);

    const int output_layer_size = layer_sizes_[layer_sizes_.size() - 1];

    if (output_layer_size == 1) {
      for (int i = 0; i < num_rows; i++) {
        const double diff = labels[i] - (*predict_outputs)[i][0];
        loss += 0.5 * diff * diff;
      }
    } else {
      for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < output_layer_size; j++) {
          const double diff = (labels[i] == j ? 1 : 0) - (*predict_outputs)[i][j];
          loss += 0.5 * diff * diff;
        }
      }
    }
    return loss / labels.size() / output_layer_size;
  }

  // Computes the average loss using <mat> and <weights_>.
  template <typename T>
  double ComputeAccuracy(const T &mat, const std::vector<int> &labels) const {
    int correct_count = 0;
    const int num_rows = mat.get_num_rows();
    const std::vector<std::vector<double>> *predict_outputs =
        ForwardComputing(mat);

    const int output_layer_size = layer_sizes_[layer_sizes_.size() - 1];

    if (output_layer_size == 1) {
      for (int i = 0; i < num_rows; i++) {
        if ((labels[i] == 0 && (*predict_outputs)[i][0] < 0.5) ||
            (labels[i] == 1 && (*predict_outputs)[i][0] > 0.5)) {
          correct_count += 1;
        }
      }
    } else {
      for (int i = 0; i < num_rows; i++) {
        // Find out the prediction.
        int prediction = -1;
        double confidence = 0;
        for (int j = 0; j < output_layer_size; j++) {
          if (prediction == -1 || (*predict_outputs)[i][j] > confidence) {
            prediction = j;
            confidence = (*predict_outputs)[i][j];
          }
        }
        if (labels[i] == prediction)
          correct_count++;
      }
    }
    return correct_count * 1.0 / labels.size();
  }

private:
  // Forward computing. Returns the outputs of the output layers.
  template <typename T>
  const std::vector<std::vector<double>> *ForwardComputing(const T &mat) const {
    const int num_rows = mat.get_num_rows();
    layer_connections_[0]->ComputeOutputs(mat, /*soft_max=*/false);
    for (int i = 1; i < layer_connections_.size(); i++) {
      // For the output layer, if multiple outputs, use softmax instead of
      // sigmoid as the activation function.
      if (i == layer_connections_.size() - 1 &&
          layer_sizes_[layer_sizes_.size() - 1] > 1) {
        layer_connections_[i]->ComputeOutputs(
            *layer_connections_[i - 1]->GetOutputs(), /*soft_max=*/true);
      } else {
        layer_connections_[i]->ComputeOutputs(
            *layer_connections_[i - 1]->GetOutputs(), /*soft_max=*/false);
      }
    }
    return layer_connections_[layer_connections_.size() - 1]
        ->GetOutputs()->get_data();
  }

  // The connections between layers.
  std::vector<std::unique_ptr<LayerConnection>> layer_connections_;

  // The sizes of layers from input layers to output layers.
  std::vector<int> layer_sizes_;

  double learning_rate_;

  // The mini batch size. This is recorded here for better memory management.
  int mini_batch_size_;
};

} // namespace model

#endif
