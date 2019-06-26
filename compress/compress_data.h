#ifndef _COMPRESS_COMPRESS_DATA_H
#define _COMPRESS_COMPRESS_DATA_H

#include <algorithm>
#include <iomanip>
#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace compress {

// some constants used for LZW comression
const int ROOT_SEQ_NUM = -1;

// This class represents a tree node in the prefix tree in LZW algo.
// T is the value type. F is the hash comparision function for type T.
template <typename T, typename F> class TreeNode {
public:
  TreeNode(int seq_num, const T &val) : seq_num_(seq_num), val_(val) {}

  // Empty constructor
  TreeNode() {}

  void insert_parent(TreeNode<T, F> *parent) { parent_ = parent; }
  TreeNode<T, F> *get_parent() { return parent_; }

  // Tells whether a new is inserted.
  bool insert_child(const T &key, TreeNode<T, F> *child) {
    auto result = children_.insert(std::make_pair(key, child));
    return result.second;
  }

  TreeNode<T, F> *get_child(const T &key) {
    if (terminal())
      return nullptr;
    auto iter = children_.find(key);
    if (iter == children_.end())
      return nullptr;
    else
      return iter->second;
  }

  bool terminal() { return children_.empty(); }

  int get_seq_num() { return seq_num_; }

  void set_seq_num(int seq_num) { seq_num_ = seq_num; }

  T get_value() { return val_; }

  void set_value(const T &val) { val_ = val; }

private:
  std::unordered_map<T, TreeNode<T, F> *, F> children_;
  TreeNode<T, F> *parent_;
  int seq_num_; // seq number of this tree node
  T val_;
};

template <typename T, typename F> class LZWCompresser {
public:
  LZWCompresser(const std::vector<std::vector<T>> &data, int dim) {
    data_ = data;
    dim_ = dim;
   }

  ~LZWCompresser() {
    if (root_ != nullptr)
      delete root_;
    for (auto *p : tree_nodes_) {
      delete p;
    }
  }

  bool compress_data();

  std::vector<std::vector<T>> get_data() { return data_; }

  std::vector<std::vector<int>> get_codes() { return codes_; }

  std::vector<T> get_init_data() { return init_data_; }

  int get_seq_num() { return seq_num_; }

  int get_code_size() {
    int code_size = 0;
    for (int i = 0; i < codes_.size(); i++)
      code_size += codes_[i].size();
    return code_size;
  }

private:
  // Initializes the root.
  void InitRoot() {
    if (root_ != nullptr)
      delete root_;

    root_ = new TreeNode<T, F>(ROOT_SEQ_NUM, T());
    seq_num_ = 0;
    tree_nodes_.push_back(new TreeNode<T, F>());
  }

  // Initializes the tree.
  void InitTree() {
    // Scans the whole dataset to add all the distinct column-idx:value
    // pairs to the tree and <_init_data>.
    for (const auto &data_vec : data_) {
      for (const auto &data_point : data_vec) {
        if (root_->insert_child(data_point, tree_nodes_[seq_num_])) {
          tree_nodes_[seq_num_]->set_seq_num(seq_num_);
          seq_num_++;
          tree_nodes_.push_back(new TreeNode<T, F>());
          init_data_.push_back(data_point);
        }
      }
    }
  }

  TreeNode<T, F> *root_ = nullptr;
  int dim_;
  std::vector<std::vector<T>> data_;
  std::vector<std::vector<int>> codes_;
  std::vector<T> init_data_;
  std::vector<TreeNode<T, F> *> tree_nodes_;
  int seq_num_;
};

template <typename T, typename F> bool LZWCompresser<T, F>::compress_data() {
  InitRoot();
  InitTree();

  const int num_rows = data_.size();
  for (int i = 0; i < num_rows; i++) {
    std::vector<int> code_vec;
    int idx = 0;
    const int row_size = data_[i].size();
    while (idx < row_size) {
      int len = 0;
      TreeNode<T, F> *pointer = root_;
      T *value = &data_[i][idx];
      TreeNode<T, F> *p;
      while (p = pointer->get_child(*value), p != nullptr) {
        len++;
        pointer = p;
        if (idx + len == row_size) {
          break; // get the end of the row
        }
        value = &data_[i][idx + len];
      }
      code_vec.push_back(pointer->get_seq_num());
      // append the unmatched character
      if (idx + len < row_size) {
        TreeNode<T, F> *child = tree_nodes_[seq_num_];
        pointer->insert_child(*value, child);
        child->set_seq_num(seq_num_++);
        tree_nodes_.push_back(new TreeNode<T, F>());
      }
      idx += len;
    }
    codes_.push_back(std::move(code_vec));
  }
  return true;
}

} // namespace compress

#endif // compress/compress_data.h
