#include <benchmark/benchmark.h>
#include <stan/math/prim/mat/fun/log_sum_exp.hpp>
#include <Eigen/Dense>

static void Softmax_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::VectorXd;

  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    VectorXd theta(1000);
    VectorXd out(1000);
    double sum = 0.0;
    double max = v_d.maxCoeff();
  for (int i = 0; i < v_d.size(); ++i) {
    theta(i) = std::exp(v_d(i) - max);  // extra work for (v[i] == max_v)
    sum += theta(i);               // extra work vs. sum() w. auto-diff
  }
  for (int i = 0; i < v_d.size(); ++i)
    out(i) = theta(i) / sum;
  }
}
BENCHMARK(Softmax_Old);

static void Softmax_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::VectorXd;

  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    VectorXd theta = (v_d.array() - v_d.maxCoeff()).exp();
    VectorXd out = theta.array() / theta.sum();
  }
}
BENCHMARK(Softmax_New);

BENCHMARK_MAIN();
