// Copyright Taylor Pospisil 2018.
// Distributed under MIT License (http://opensource.org/licenses/MIT)
#include <Rcpp.h>

#include "Tree.h"
#include "Forest.h"

using namespace Rcpp;

//' @name ForestRcpp
//' @title Fit a random forest for CDE
//'
//' @description Provides a wrapper to the C++ RFCDE implementation.
//'
//' @param x_train a matrix of training covariates.
//' @param z_basis a matrix of basis functions evaluations for training responses.
//' @param n_trees the number of trees in the forest.
//' @param mtry the number of candidate variables to try for each split.
//' @param node_size the minimum number of observations in a leaf node.
//'
//' @export ForestRcpp
// [[Rcpp:export]]
class ForestRcpp {
private:
  Forest obj;
public:
  void train(NumericMatrix x_train, NumericMatrix z_basis, IntegerVector lens, int n_trees,
             int mtry, int node_size, double min_loss_delta, double flambda, bool fit_oob) {
    int n_train = x_train.nrow();
    int n_var = x_train.ncol();
    int n_basis = z_basis.ncol();
    
    obj.train(&x_train(0,0), &z_basis(0,0), &lens(0), n_train, n_var, n_basis,
              n_trees, mtry, node_size, min_loss_delta, flambda, fit_oob);
  };

  void fill_weights(Rcpp::NumericVector x_test, Rcpp::IntegerVector weights) {
    obj.fill_weights(&x_test(0), &weights(0));
  };

  void fill_oob_weights(Rcpp::IntegerMatrix weights) {
    obj.fill_oob_weights(&weights(0,0));
  };

  void fill_loss_importance(Rcpp::NumericVector scores) {
    obj.fill_loss_importance(&scores(0));
  };
  
  void set_leaves_id() {
    obj.forest_set_leaf_ids(); 
  };

  void fill_count_importance(Rcpp::NumericVector scores) {
    obj.fill_count_importance(&scores(0));
  }; 
  
  //NumericVector fill_count_importance(Rcpp::NumericVector x_test) {
  //  return obj.traverse_forest(&x_test(0));
  //};
  
  NumericVector traverse_forest(Rcpp::NumericVector x_test) {
    return obj.traverse_forest(&x_test(0)); // &x_test(0,0)
  };
  
  //NumericMatrix traverse_forestMatrix(Rcpp::NumericMatrix x_test) {
     
  //};//
};

RCPP_MODULE(RFCDEModule) {
  class_<ForestRcpp>("ForestRcpp")
    .constructor()
    .method("set_leaves_id", &ForestRcpp::set_leaves_id)
    .method("traverse_forest", &ForestRcpp::traverse_forest)
    //.method("traverse_forestMatrix", &ForestRcpp::traverse_forestMatrix)
    .method("train", &ForestRcpp::train)
    .method("fill_weights", &ForestRcpp::fill_weights)
    .method("fill_oob_weights", &ForestRcpp::fill_oob_weights)
    .method("fill_loss_importance", &ForestRcpp::fill_loss_importance)
    .method("fill_count_importance", &ForestRcpp::fill_count_importance)
    ;
}
