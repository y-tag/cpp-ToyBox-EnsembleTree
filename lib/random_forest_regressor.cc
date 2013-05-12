#include "random_forest_regressor.h"

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

RandomForestRegressor::RandomForestRegressor()
  : tree_num_(10), leaf_num_(10), min_leaf_instance_rate_(1e-4),
     data_sampling_rate_(0.5),feature_sampling_rate_(0.5) {
}

RandomForestRegressor::RandomForestRegressor(
    int tree_num, int leaf_num,
    double min_leaf_instance_rate,
    double data_sampling_rate,
    double feature_sampling_rate)
  : tree_num_(tree_num), leaf_num_(leaf_num),
    min_leaf_instance_rate_(min_leaf_instance_rate),
    data_sampling_rate_(data_sampling_rate),
    feature_sampling_rate_(feature_sampling_rate) {
}

RandomForestRegressor::~RandomForestRegressor() {
}

int RandomForestRegressor::Train(
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

  for (int i = 0; i < tree_num_; ++i) {
    RegressionTree tree(
        leaf_num_, min_leaf_instance_rate_,
        feature_sampling_rate_, true
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

    tree.Train(x_vec, y_vec, sampled_datum_vec, target_feature_vec);
    tree_vec_.push_back(tree);
  }

  return 1;
}

double RandomForestRegressor::Predict(const datum &x) const {
  double predicted_value = 0.0;
  for (size_t i = 0; i < tree_vec_.size(); ++i) {
    predicted_value += tree_vec_[i].Predict(x);
  }
  predicted_value /= tree_vec_.size();

  return predicted_value;
}

bool RandomForestRegressor::IsInitialized() const {
  return true;
}


} // namespace ensemble
} // namespace toybox
