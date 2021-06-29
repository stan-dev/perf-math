#ifndef CALLBACK_TOSS_ME_HPP
#define CALLBACK_TOSS_ME_HPP

#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>

static bool did_already = false;
// Just to fill up the stack allocator
template <int Size>
static void toss_me(benchmark::State& state) {
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
  for (auto _ : state) {
    if (!did_already) {
      did_already = true;
      lp.grad();
      stan::math::set_zero_all_adjoints();
    }
  }
  stan::math::recover_memory();
}

#endif
