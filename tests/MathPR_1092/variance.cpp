#include <benchmark/benchmark.h>
#include <stan/math/prim/mat/fun/mean.hpp>
#include <Eigen/Dense>

static void Variance_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::VectorXd;
  using stan::math::mean;

  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    double mn = mean(v_d);
    double sum_sq_diff = 0.0;
    for (int i = 0; i < v_d.size(); ++i) {
      double diff = v_d(i) - mn;
      sum_sq_diff += diff * diff;
    }
  }
}
BENCHMARK(Variance_Old);

static void Variance_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::VectorXd;

  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    double out = (v_d.array() - v_d.mean()).matrix().squaredNorm();
  }
}
BENCHMARK(Variance_New);

BENCHMARK_MAIN();
