#include <benchmark/benchmark.h>
#include <Eigen/Dense>

static void ColDotProduct_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::VectorXd;
  using Eigen::MatrixXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  MatrixXd m_d2 = MatrixXd::Random(1000, 1000);

  for (auto _ : state) {
    VectorXd out(1000);

    for (int i = 0; i < 1000; ++i)
      out(i) = m_d.col(i).dot(m_d2.col(i));
  }
}
BENCHMARK(ColDotProduct_Old);

static void ColDotProduct_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::VectorXd;
  using Eigen::MatrixXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  MatrixXd m_d2 = MatrixXd::Random(1000, 1000);

  for (auto _ : state) {
    VectorXd out = (m_d.transpose() * m_d2).diagonal();
  }
}
BENCHMARK(ColDotProduct_New);

BENCHMARK_MAIN();
