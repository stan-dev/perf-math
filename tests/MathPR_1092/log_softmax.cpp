#include <benchmark/benchmark.h>
#include <stan/math/prim/mat/fun/log_sum_exp.hpp>
#include <Eigen/Dense>

static void LogSoftmx_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::VectorXd;
  using stan::math::log_sum_exp;

  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    VectorXd out(1000);
    double z = log_sum_exp(v_d);
    for (int i = 0; i < v_d.size(); ++i)
      out(i) = v_d(i) - z;
  }
}
BENCHMARK(LogSoftmx_Old);

static void LogSoftmx_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::VectorXd;
  using stan::math::log_sum_exp;

  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    VectorXd out = v_d.array() - log_sum_exp(v_d);
  }
}
BENCHMARK(LogSoftmx_New);

BENCHMARK_MAIN();
