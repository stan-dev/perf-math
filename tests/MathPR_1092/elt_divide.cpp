#include <benchmark/benchmark.h>
#include <Eigen/Dense>



static void Div_Mat_Mat_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  MatrixXd m_d2 = MatrixXd::Random(1000, 1000);

  for (auto _ : state) {
    MatrixXd out(1000,1000);

    for (int i = 0; i < out.size(); ++i)
      out(i) = m_d(i) / m_d2(i);
  }
}
BENCHMARK(Div_Mat_Mat_Old);

static void Div_Mat_Mat_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  MatrixXd m_d2 = MatrixXd::Random(1000, 1000);

  for (auto _ : state) {
    MatrixXd out = m_d.array() / m_d2.array();
  }
}
BENCHMARK(Div_Mat_Mat_New);

BENCHMARK_MAIN();
