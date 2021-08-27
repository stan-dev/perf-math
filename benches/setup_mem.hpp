#ifndef CALLBACK_SETUP_MEM_HPP
#define CALLBACK_SETUP_MEM_HPP

#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>

/**
 * Setup memory onto Stan's memory arena. When Stan is executing after the
 * first run it should never allocate memory and just take memory from it's
 * stack allocator. In order to replicate this in benchmarks we first call this
 * function which places a bunch of memory on the stack allocator.
 * and then free it for the next use.
 */
template <int Size>
static void setup_mem() {
  using stan::math::var;
  Eigen::Matrix<double, -1, -1> x_vals = Eigen::MatrixXd::Random(Size, Size);
  Eigen::Matrix<double, -1, -1> y_vals = Eigen::MatrixXd::Random(Size, Size);
  using stan::math::sum;
  Eigen::Matrix<var, -1, -1> x = x_vals;
  Eigen::Matrix<var, -1, -1> y = y_vals;
  var lp = 0;
  for (Eigen::Index i = 0; i < x_vals.size(); ++i) {
    lp += x.coeffRef(i) * y.coeffRef(i) + x.coeffRef(i);
  }
  benchmark::DoNotOptimize(lp.vi_);
  lp.grad();
  stan::math::recover_memory();
}

#endif
