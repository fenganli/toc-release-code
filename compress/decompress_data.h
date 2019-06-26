#ifndef _COMPRESS_DECOMPRESS_DATA_H
#define _COMPRESS_DECOMPRESS_DATA_H

#include <algorithm>
#include <iomanip>
#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace compress {

template <typename T, typename F> class LZWDecompresser {
public:
  LZWDecompresser(const std::vector<std::vector<int>> codes,
                  const std::vector<T> &init_data, int dim,
                  int num_dict_nodes) {
    codes_ = codes;
    init_data_ = init_data;
    tree_nodes_.resize(num_dict_nodes);
  }

  ~LZWDecompresser() {
    if (root_ != nullptr)
      delete root_;
  }

  bool decompress_data();

  std::vector<std::vector<T>> get_data() { return data_; }

  int get_seq_num() { return seq_num_; }

private:
  // Initializes the root.
  void InitRoot() {
    if (root_ != nullptr)
      delete root_;

    root_ = new TreeNode<T, F>(ROOT_SEQ_NUM, T());
    seq_num_ = 0;
  }

  // Initializes the tree.
  void InitTree() {
    // Scans <init_data_> to add all the distinct column-idx:value
    // pairs to the tree.
    for (const auto &data_point : init_data_) {
      TreeNode<T, F> *node = &tree_nodes_[seq_num_];
      node->set_value(data_point);
      node->set_seq_num(seq_num_);
      node->insert_parent(root_);
      seq_num_++;
    }
  }

  TreeNode<T, F> *root_ = nullptr;
  int dim_;
  std::vector<std::vector<T>> data_;
  std::vector<std::vector<int>> codes_;
  std::vector<T> init_data_;
  // Stores the tree nodes. The index of the vector is also
  // the sequence number of the tree node.
  std::vector<TreeNode<T, F>> tree_nodes_;
  int seq_num_;
};

template <typename T, typename F>
bool LZWDecompresser<T, F>::decompress_data() {
  InitRoot();
  InitTree();

  const int num_rows = codes_.size();
  for (int i = 0; i < num_rows; i++) {
    std::vector<T> row_vec;
    const int num_codes = codes_[i].size();
    int last_code = -1;
    for (int j = 0; j < num_codes; j++) {
      const int code = codes_[i][j];
      TreeNode<T, F> *tree_node = &tree_nodes_[code];
      const int row_vec_size = row_vec.size();
      // Do a backtracking to get all the T values.
      while (tree_node->get_seq_num() != ROOT_SEQ_NUM) {
        row_vec.push_back(tree_node->get_value());
        tree_node = tree_node->get_parent();
      }
      // Appends the tree
      if (last_code != -1 && seq_num_ < tree_nodes_.size()) {
        TreeNode<T, F> *node = &tree_nodes_[seq_num_];
        node->set_value(row_vec.back());
        node->set_seq_num(seq_num_++);
        node->insert_parent(&tree_nodes_[last_code]);
      }
      std::reverse(row_vec.begin() + row_vec_size, row_vec.end());
      last_code = code;
    }
    data_.push_back(std::move(row_vec));
  }
  return true;
}

} // namespace compress

#endif // compress/decompress_data.h
