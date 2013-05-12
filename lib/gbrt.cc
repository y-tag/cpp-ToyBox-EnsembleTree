#include "gbrt.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <functional>
#include <map>
#include <vector>
#include <utility>

namespace {


} // namespace

namespace toybox {
namespace ensemble {

GBRT::GBRT()
  : tree_num_(1), learning_rate_(0.1),
    leaf_num_(10), min_leaf_instance_rate_(1e-4),
    data_sampling_rate_(1.0),
    feature_sampling_rate_(1.0), is_feature_sampling_randomized_(false),
    base_predict_(0.0) {
}

GBRT::GBRT(
    int tree_num, double learning_rate,
    int leaf_num, double min_leaf_instance_rate,
    double data_sampling_rate, double feature_sampling_rate,
    bool is_feature_sampling_randomized)
  : tree_num_(tree_num), learning_rate_(learning_rate),
    leaf_num_(leaf_num), min_leaf_instance_rate_(min_leaf_instance_rate),
    data_sampling_rate_(data_sampling_rate),
    feature_sampling_rate_(feature_sampling_rate),
    is_feature_sampling_randomized_(is_feature_sampling_randomized),
    base_predict_(0.0) {
}

GBRT::~GBRT() {
}

int GBRT::Train(
    const data &x_vec,
    const std::vector<double> &y_vec,
    const std::vector<int> &target_datum_vec,
    const std::vector<int> &target_feature_vec) {
  if (leaf_num_ < 2) {
    return -1;
  }

  tree_vec_.clear();

  std::vector<int> datum_vec(target_datum_vec);
  std::sort(datum_vec.begin(), datum_vec.end());

  size_t sampled_num = datum_vec.size() * data_sampling_rate_;

  std::vector<double> diff_y_vec(y_vec);
  std::vector<double> current_y_vec(y_vec.size(), 0.0);

  base_predict_ = 0.0;
  for (size_t i = 0; i < datum_vec.size(); ++i) {
    int idx = datum_vec[i];
    base_predict_ += y_vec[idx];
  }
  base_predict_ /= datum_vec.size();

  for (size_t i = 0; i < datum_vec.size(); ++i) {
    int idx = datum_vec[i];
    current_y_vec[idx] += base_predict_;
    diff_y_vec[idx] = y_vec[idx] - current_y_vec[idx];
  }

  for (int i = 0; i < tree_num_; ++i) {
    RegressionTree tree(
        leaf_num_, min_leaf_instance_rate_,
        feature_sampling_rate_, is_feature_sampling_randomized_
    );

    std::vector<int> sampled_datum_vec;
    if (sampled_num == datum_vec.size()) {
      sampled_datum_vec.assign(datum_vec.begin(), datum_vec.end());
    } else {
      for (size_t j = 0; j < sampled_num; ++j) {
        int idx = rand() % datum_vec.size();
        sampled_datum_vec.push_back(datum_vec[idx]);
      }
      std::sort(sampled_datum_vec.begin(), sampled_datum_vec.end());
    }

    tree.Train(x_vec, diff_y_vec, sampled_datum_vec, target_feature_vec);

    for (int node = 0; node < leaf_num_; ++node) {
      double v = tree.GetLeafNodeValue(node);
      tree.SetLeafNodeValue(node, learning_rate_ * v);
    }

    for (size_t i = 0; i < datum_vec.size(); ++i) {
      int idx = datum_vec[i];
      current_y_vec[idx] += tree.Predict(x_vec[idx]);
      diff_y_vec[idx] = y_vec[idx] - current_y_vec[idx];
    }

    tree_vec_.push_back(tree);
  }

  return 1;
}

double GBRT::Predict(const datum &x) const {
  double predicted_value = base_predict_;
  for (size_t i = 0; i < tree_vec_.size(); ++i) {
    predicted_value += tree_vec_[i].Predict(x);
  }

  return predicted_value;
}

bool GBRT::IsInitialized() const {
  return true;
}


} // namespace ensemble
} // namespace toybox
