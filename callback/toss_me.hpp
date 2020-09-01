#ifndef CALLBACK_TOSS_ME_HPP
#define CALLBACK_TOSS_ME_HPP

#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>

// Just to fill up the stack allocator
static void toss_me(benchmark::State& state) {
  using stan::math::var;
  Eigen::Matrix<double, -1, -1> x_vals = Eigen::MatrixXd::Random(5000, 5000);
  using stan::math::var;
  using stan::math::sum;
  Eigen::Matrix<var, -1, -1> x = x_vals;
  auto blah = sum(x);
  benchmark::DoNotOptimize(blah.vi_);
  for (auto _ : state) {
  }
  stan::math::recover_memory();
}

#endif
