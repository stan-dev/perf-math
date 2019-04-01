#include <benchmark/benchmark.h>
#include <Eigen/Dense>

static void DivideCols_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  std::vector<VectorXd> v_d_array(1000);
  VectorXd v_d = VectorXd::Random(1000);

  for (int i = 0; i < 1000; ++i) 
    v_d_array[i] = VectorXd::Random(1000);

  for (auto _ : state) {
    std::vector<VectorXd> out(1000);

    for (size_t n = 0; n < 1000; ++n) {
      out[n].resize(1000);
      for (size_t d = 0; d < 1000; ++d) {
        out[n][d] = v_d_array[n][d] / v_d[d];
      }
    }
  }
}
BENCHMARK(DivideCols_Old);

static void DivideCols_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  std::vector<VectorXd> v_d_array(1000);
  VectorXd v_d = VectorXd::Random(1000);

  for (int i = 0; i < 1000; ++i) 
    v_d_array[i] = VectorXd::Random(1000);

  for (auto _ : state) {
    std::vector<VectorXd> out(1000);

    for (size_t n = 0; n < 1000; ++n) {
      out[n].resize(1000);
      out[n] = v_d_array[n].array() / v_d.array();
    }
  }
}
BENCHMARK(DivideCols_New);

BENCHMARK_MAIN();
