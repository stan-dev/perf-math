#include <benchmark/benchmark.h>
#include <vector>
#include <numeric>
#include <functional>
#include <Eigen/Dense>

static void CumSum_Old(benchmark::State& state) {
  using Eigen::VectorXd;

  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    VectorXd out(1000);
    out(0) = v_d(0);

    for (int i = 1; i < out.size(); ++i)
      out(i) = v_d(i) + out(i - 1);
  }
}
BENCHMARK(CumSum_Old);

static void CumSum_New(benchmark::State& state) {
  using Eigen::VectorXd;

  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    VectorXd out(1000);

    std::partial_sum(v_d.data(), v_d.data() + v_d.size(), out.data(),
                 std::plus<double>());
  }
}
BENCHMARK(CumSum_New);

BENCHMARK_MAIN();
