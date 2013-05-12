#include "regression_tree.h"
#include "random_forest_regressor.h"
#include "gbrt.h"
#include "svmlight_reader.h"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <vector>
#include <map>

int main(int argc, char **argv) {

  if (argc < 3) {
    fprintf(stderr, "Usage: %s train_file test_file\n", argv[0]);
    return 1;
  }

  const char *train_file = argv[1];
  const char *test_file  = argv[2];

  srand(1000);

  toybox::ensemble::SVMLightReader train_reader(train_file);
  toybox::ensemble::SVMLightReader test_reader(test_file);
  if (! train_reader.IsInitialized()) {
    fprintf(stderr, "Fail to open train_file: %s\n", train_file);
    return 1;
  }
  if (! test_reader.IsInitialized()) {
    fprintf(stderr, "Fail to open test_file: %s\n", test_file);
    return 1;
  }

  int ret = 0;
  std::vector<std::pair<int, double> > x;
  double y = 0;

  std::vector<int> datum_vec;
  std::vector<int> feature_vec;
  int max_fid = 0;

  std::vector<std::vector<std::pair<int, double> > > tmp_x_vec;
  std::vector<double> y_vec;
  while (train_reader.Read(&x, &y) == 1) {
    for (size_t j = 0; j < x.size(); ++j) {
      int k = x[j].first;
      if (k > max_fid) {
        max_fid = k;
      }
    }
    tmp_x_vec.push_back(x);
    y_vec.push_back(y);
  }

  toybox::ensemble::data x_vec;
  for (size_t i = 0; i < tmp_x_vec.size(); ++i) {
    toybox::ensemble::datum dx(max_fid);
    for (size_t j = 0; j < tmp_x_vec[i].size(); ++j) {
      int    k = tmp_x_vec[i][j].first - 1;
      double v = tmp_x_vec[i][j].second;
      if (k < max_fid) {
        dx[k] = v;
      }
    }

    x_vec.push_back(dx);
  }

  for (size_t i = 0; i < x_vec.size(); ++i) {
    datum_vec.push_back(i);
  }

  for (int fid = 0; fid < max_fid; ++fid) {
    feature_vec.push_back(fid);
  }

  int tree_num = 10;
  double learning_rate = 1.0;
  int leaf_num = 10;
  double min_leaf_rate = 0.25 * 1e-2;
  double drate = 1.0;
  double frate = 1.0;
  bool is_randomized = false;

  //toybox::ensemble::RegressionTree tree(leaf_num, min_leaf_rate, frate, is_randomized);
  //toybox::ensemble::RandomForestRegressor tree(tree_num, leaf_num, min_leaf_rate, drate, frate);
  toybox::ensemble::GBRT tree(tree_num, learning_rate, leaf_num, min_leaf_rate, drate, frate, is_randomized);

  tree.Train(x_vec, y_vec, datum_vec, feature_vec);

  double squared_error = 0.0;
  while (test_reader.Read(&x, &y) == 1) {
    std::vector<double> dx(max_fid);
    for (size_t j = 0; j < x.size(); ++j) {
      int    k = x[j].first - 1;
      double v = x[j].second;
      if (k < max_fid) {
        dx[k] = v;
      }
    }

    double predicted_value = tree.Predict(dx);
    fprintf(stdout, "%f\n", predicted_value);

    squared_error += (predicted_value - y) * (predicted_value - y);
  }
  fprintf(stderr, "squared_error: %f\n", squared_error);

  return 0;
}
