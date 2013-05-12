#include "regression_tree.h"

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

class SortByFeatureValue {
  public:
    SortByFeatureValue(const toybox::ensemble::data &x_vec, int feature)
      : x_vec_(x_vec), feature_(feature) {};
    bool operator()(const int a, const int b) {
      return x_vec_[a][feature_] < x_vec_[b][feature_];
    }

  private:
    const toybox::ensemble::data &x_vec_;
    int feature_;
    SortByFeatureValue();
};

int find_best_split(
    const toybox::ensemble::data &x_vec,
    const std::vector<double> &y_vec,
    const std::vector<int> &feature_vec,
    size_t fnum_to_use,
    int min_leaf_instance,
    int start_index, int end_index,
    std::vector<int> *datum_vec,
    int *feature, double *thresh,
    int *split_index,
    double *gain,
    double *left_mean, double *right_mean
    ) {

  int best_feature   = -1; *feature = -1;
  double best_thresh = 0.0; *thresh = 0.0;
  double least_loss  = DBL_MAX; *gain = 0.0;
  double best_left_mean  = 0.0; *left_mean = 0.0;
  double best_right_mean = 0.0; *right_mean = 0.0;

  //fprintf(stderr, "start_index:%d, end_index:%d\n", start_index, end_index);
  if (end_index - start_index < std::max(2, min_leaf_instance) ||
      end_index > static_cast<int>(datum_vec->size())) {
    return -1;
  }

  double sum = 0.0;
  for (int j = start_index; j < end_index; ++j) {
    int idx = (*datum_vec)[j];
    sum += y_vec[idx];
  }
  //fprintf(stderr, "sum: %f\n", sum);

  double old_loss = 0.0;
  double old_mean = sum / (end_index - start_index);
  for (int j = start_index; j < end_index; ++j) {
    int idx = (*datum_vec)[j];
    double tmp = y_vec[idx];
    old_loss += (tmp - old_mean) * (tmp - old_mean);
  }
  //fprintf(stderr, "old_mean: %f\n", old_mean);
  //fprintf(stderr, "old_loss: %f\n", old_loss);
  
  for (size_t i = 0; i < fnum_to_use; ++i) {
    int f = feature_vec[i];
    std::sort(
        datum_vec->begin() + start_index,
        datum_vec->begin() + end_index,
        SortByFeatureValue(x_vec, f)
    );

    /*
    fprintf(stderr, "f: %d\n", f);
    for (int j = start_index; j < end_index; ++j) {
      int idx = (*datum_vec)[j];
      fprintf(stderr, "label:%f", y_vec[idx]);
      for (size_t j = 0; j < x_vec[idx].size(); ++j) {
        fprintf(stderr, " %lu:%f", j, x_vec[idx][j]);
      }
      fprintf(stderr, "\n");
    }
    */

    double left_sum = 0.0;
    int left_num = 0;

    for (int j = start_index; j < end_index - 1; ++j) {
      int idx = (*datum_vec)[j];
      double t = x_vec[idx][f];

      left_sum += y_vec[idx];
      left_num += 1;

      int next_index = (*datum_vec)[j + 1];
      if (t == x_vec[next_index][f]) {
        continue;
      }

      double right_sum = sum - left_sum;
      int right_num = (end_index - start_index) - left_num;

      if (left_num  < min_leaf_instance) { continue; }
      if (right_num < min_leaf_instance) { break; }

      double lmean = left_sum / left_num;
      double rmean = right_sum / right_num;

      double l = 0.0;
      for (int k = start_index; k <= j; ++k) {
        int kidx = (*datum_vec)[k];
        double tmp = y_vec[kidx];
        l += (tmp - lmean) * (tmp - lmean);
      }
      for (int k = j + 1; k < end_index; ++k) {
        int kidx = (*datum_vec)[k];
        double tmp = y_vec[kidx];
        l += (tmp - rmean) * (tmp - rmean);
      }

      //fprintf(stderr, "f: %d, t: %f, ls: %f, ln: %d, lm: %f, rs: %f, rn: %d, rm: %f, l: %f\n", f, t, left_sum, left_num, lmean, right_sum, right_num, rmean, l);

      if (l < least_loss) {
        best_feature = f; best_thresh  = t;
        least_loss   = l;
        best_left_mean  = lmean; best_right_mean = rmean;
      }
    }
  }

  if (best_feature < 0) {
    return -1;
  }

  std::sort(
      datum_vec->begin() + start_index,
      datum_vec->begin() + end_index,
      SortByFeatureValue(x_vec, best_feature)
      );

  int i = start_index;
  while (i < end_index) {
    int idx = (*datum_vec)[i];
    if (x_vec[idx][best_feature] > best_thresh) {
      break;
    }
    ++i;
  }

  *feature = best_feature;
  *thresh  = best_thresh;
  *split_index = i;
  *gain = old_loss - least_loss;
  *left_mean  = best_left_mean;
  *right_mean = best_right_mean;

  return 1;
}

struct LeafInfo {
  LeafInfo(int leaf_node, int feature, double thresh, double gain,
           int start_index, int split_index, int end_index,
           double left_mean, double right_mean) 
    : leaf_node_(leaf_node), feature_(feature), thresh_(thresh), gain_(gain), 
      start_index_(start_index), split_index_(split_index), end_index_(end_index),
      left_mean_(left_mean), right_mean_(right_mean) {};
  int leaf_node_;
  int feature_;
  double thresh_;
  double gain_;
  int start_index_;
  int split_index_;
  int end_index_;
  double left_mean_;
  double right_mean_;
};

bool gain_less_than(const LeafInfo &a, const LeafInfo &b) {
  return a.gain_ < b.gain_;
}


} // namespace

namespace toybox {
namespace ensemble {

RegressionTree::RegressionTree()
  : leaf_num_(10), min_leaf_instance_rate_(1e-4),
    feature_sampling_rate_(1.0), is_feature_sampling_randomized_(false),
    feature_vec_(1, 0), thresh_vec_(1, DBL_MAX),
    left_vec_(1, ~0), right_vec_(1, ~0),
    predict_vec_(1, 0.0) {
}

RegressionTree::RegressionTree(
    int leaf_num, double min_leaf_instance_rate,
    double feature_sampling_rate, bool is_feature_sampling_randomized)
  : leaf_num_(leaf_num), min_leaf_instance_rate_(min_leaf_instance_rate),
    feature_sampling_rate_(feature_sampling_rate),
    is_feature_sampling_randomized_(is_feature_sampling_randomized),
    feature_vec_(1, 0), thresh_vec_(1, DBL_MAX),
    left_vec_(1, ~0), right_vec_(1, ~0),
    predict_vec_(1, 0.0) {
}

RegressionTree::~RegressionTree() {
}

int RegressionTree::Train(
    const data &x_vec,
    const std::vector<double> &y_vec,
    const std::vector<int> &target_datum_vec,
    const std::vector<int> &target_feature_vec) {
  if (leaf_num_ < 2) {
    return -1;
  }

  feature_vec_.assign(leaf_num_ - 1, 0);
  thresh_vec_.assign(leaf_num_ - 1, 0.0);
  left_vec_.assign(leaf_num_ - 1, 0);
  right_vec_.assign(leaf_num_ - 1, 0);
  predict_vec_.assign(leaf_num_, 0.0);

  std::vector<int> datum_vec(target_datum_vec);
  std::sort(datum_vec.begin(), datum_vec.end());
  std::vector<int> feature_vec(target_feature_vec);

  int min_leaf_instance = datum_vec.size() * min_leaf_instance_rate_;
  size_t fnum_to_use = feature_vec.size() * feature_sampling_rate_;
  if (fnum_to_use < feature_vec.size()) {
    std::random_shuffle(feature_vec.begin(), feature_vec.end());
  } else {
    std::sort(feature_vec.begin(), feature_vec.end());
  }

  int inter_node = 0; int leaf_node = 0;
  int feature = -1; double thresh = 0.0;
  int start_index = 0; int split_index = 0; int end_index = datum_vec.size();
  double gain = 0.0;
  double left_mean = 0.0; double right_mean = 0.0;

  // set default leaf node
  double sum = 0.0;
  for (int j = start_index; j < end_index; ++j) {
    sum += y_vec[datum_vec[j]];
  }
  feature_vec_[0] = 0; thresh_vec_[0] = DBL_MAX;
  left_vec_[0] = ~0; right_vec_[0] = ~0;
  predict_vec_[0] = sum / (end_index - start_index);

  std::vector<LeafInfo> candidate_heap;
  std::vector<int> leaf_node_heap;

  for (int i = 0; i < leaf_num_; ++i) {
    leaf_node_heap.push_back(i);
  }
  std::make_heap(leaf_node_heap.begin(), leaf_node_heap.end(), std::greater<int>());

  if (is_feature_sampling_randomized_) {
    std::random_shuffle(feature_vec.begin(), feature_vec.end());
  }
  find_best_split(
      x_vec, y_vec, feature_vec, fnum_to_use, min_leaf_instance,
      start_index, end_index,
      &datum_vec, &feature, &thresh, &split_index, &gain,
      &left_mean, &right_mean
  );

  if (feature < 0) { // cannot split at all
    return 0;
  }

  // get leaf node id
  leaf_node = leaf_node_heap.front();
  std::pop_heap(leaf_node_heap.begin(), leaf_node_heap.end(), std::greater<int>());
  leaf_node_heap.pop_back();
  // push split candidate info
  LeafInfo info(leaf_node, feature, thresh, gain, start_index, split_index, end_index, left_mean, right_mean);
  candidate_heap.push_back(info);
  std::push_heap(candidate_heap.begin(), candidate_heap.end(), gain_less_than);

  // grow tree
  while (candidate_heap.size() > 0 && leaf_node_heap.size() > 0) {
    //pop the candidate that has largest gain
    LeafInfo info = candidate_heap.front();
    std::pop_heap(candidate_heap.begin(), candidate_heap.end(), gain_less_than);
    candidate_heap.pop_back();

    // change leaf node to internal node
    std::vector<int>::iterator itr;
    itr = std::find(left_vec_.begin(), left_vec_.end(), ~(info.leaf_node_));
    if (itr != left_vec_.end())  { *itr = inter_node; }
    itr = std::find(right_vec_.begin(), right_vec_.end(), ~(info.leaf_node_));
    if (itr != right_vec_.end()) { *itr = inter_node; }

    // release leaf node id
    leaf_node_heap.push_back(info.leaf_node_);
    std::push_heap(leaf_node_heap.begin(), leaf_node_heap.end(), std::greater<int>());

    // get leaf node ids for children
    int left_leaf_node  = leaf_node_heap.front();
    std::pop_heap(leaf_node_heap.begin(), leaf_node_heap.end(), std::greater<int>());
    leaf_node_heap.pop_back();
    int right_leaf_node = leaf_node_heap.front();
    std::pop_heap(leaf_node_heap.begin(), leaf_node_heap.end(), std::greater<int>());
    leaf_node_heap.pop_back();

    // set internal node info
    feature_vec_[inter_node] = info.feature_;
    thresh_vec_[inter_node]  = info.thresh_;
    left_vec_[inter_node]    = ~left_leaf_node;
    right_vec_[inter_node]   = ~right_leaf_node;
    inter_node++;

    // set leaf node info
    predict_vec_[left_leaf_node]  = info.left_mean_;
    predict_vec_[right_leaf_node] = info.right_mean_;

    // for left child
    if (is_feature_sampling_randomized_) {
      std::random_shuffle(feature_vec.begin(), feature_vec.end());
    }
    find_best_split(
        x_vec, y_vec, feature_vec, fnum_to_use, min_leaf_instance,
        info.start_index_, info.split_index_,
        &datum_vec, &feature, &thresh, &split_index, &gain,
        &left_mean, &right_mean
    );
    if (feature >= 0) { // splitable
      // push split candidate info
      LeafInfo linfo(left_leaf_node, feature, thresh, gain, info.start_index_, split_index, info.split_index_, left_mean, right_mean);
      candidate_heap.push_back(linfo);
      std::push_heap(candidate_heap.begin(), candidate_heap.end(), gain_less_than);
    }

    // for right child
    if (is_feature_sampling_randomized_) {
      std::random_shuffle(feature_vec.begin(), feature_vec.end());
    }
    find_best_split(
        x_vec, y_vec, feature_vec, fnum_to_use, min_leaf_instance,
        info.split_index_, info.end_index_,
        &datum_vec, &feature, &thresh, &split_index, &gain,
        &left_mean, &right_mean
    );
    if (feature >= 0) { // splitable
      // push split candidate info
      LeafInfo rinfo(right_leaf_node, feature, thresh, gain, info.split_index_, split_index, info.end_index_, left_mean, right_mean);
      candidate_heap.push_back(rinfo);
      std::push_heap(candidate_heap.begin(), candidate_heap.end(), gain_less_than);
    }

  }

  /*
  for (size_t i = 0; i < feature_vec_.size(); ++i) {
    fprintf(stderr, "%lu %d %f %d %d\n", i, feature_vec_[i], thresh_vec_[i], left_vec_[i], right_vec_[i]);
  }
  fprintf(stderr, "\n");
  for (size_t i = 0; i < predict_vec_.size(); ++i) {
    fprintf(stderr, "%lu %f\n", i, predict_vec_[i]);
  }
  fprintf(stderr, "\n");
  */

  return 1;
}

double RegressionTree::Predict(const datum &x) const {
  int node = GetCorrespondingLeafNode(x);
  if (node < 0 || node >= static_cast<int>(predict_vec_.size())) {
    return 0;
  }
  return predict_vec_[node];
}

int RegressionTree::GetCorrespondingLeafNode(const datum &x) const {
  if (predict_vec_.size() == 0) {
    return -1;
  }

  int x_size = x.size();
  int node = 0;
  for (size_t i = 0; i < predict_vec_.size(); ++i) {
    int feature   = feature_vec_[node];
    double thresh = thresh_vec_[node];

    double val = (feature < x_size) ? x[feature] : 0.0;
    node = (val <= thresh) ? left_vec_[node] : right_vec_[node];

    if (node < 0) { return ~node; }
  }

  return 0;
}

double RegressionTree::GetLeafNodeValue(int node) {
  if (node < 0 || node >= static_cast<int>(predict_vec_.size())) {
    return 0.0;
  }
  return predict_vec_[node];
}

int RegressionTree::SetLeafNodeValue(int node, double value) {
  if (node < 0 || node >= static_cast<int>(predict_vec_.size())) {
    return -1;
  }
  predict_vec_[node] = value;
  return 1;
}

bool RegressionTree::IsInitialized() const {
  return true;
}


} // namespace ensemble
} // namespace toybox
