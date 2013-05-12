#ifndef TOYBOX_ENSEMBLE_RANDOMFORESTREGRESSOR_H
#define TOYBOX_ENSEMBLE_RANDOMFORESTREGRESSOR_H

#include <vector>
#include <utility>

#include "tree.h"
#include "regression_tree.h"

namespace toybox {
namespace ensemble {

class RandomForestRegressor {
  public:
    RandomForestRegressor();
    RandomForestRegressor(
        int tree_num,
        int leaf_num,
        double min_leaf_instance_rate,
        double data_sampling_rate,
        double feature_sampling_rate);
    ~RandomForestRegressor();
    int Train(
        const data &x_vec,
        const std::vector<double> &y_vec,
        const std::vector<int> &target_datum_vec,
        const std::vector<int> &target_feature_vec);
    double Predict(const datum &x) const;
    bool IsInitialized() const;

  private:
    int tree_num_;
    double learning_rate_;
    int leaf_num_;
    double min_leaf_instance_rate_;
    double data_sampling_rate_;
    double feature_sampling_rate_;
    std::vector<RegressionTree> tree_vec_;
};


} // namespace ensemble
} // namespace toybox

#endif // TOYBOX_ENSEMBLE_RANDOMFORESTREGRESSOR_H

