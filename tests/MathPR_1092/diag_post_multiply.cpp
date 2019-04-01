#include <benchmark/benchmark.h>
#include <Eigen/Dense>

static void DiagPostMult_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    MatrixXd out(1000,1000);

    for (int j = 0; j < out.cols(); ++j)
      for (int i = 0; i < out.rows(); ++i)
      out(i,j) = v_d(j) * m_d(i,j);
  }
}
BENCHMARK(DiagPostMult_Old);

static void DiagPostMult_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    MatrixXd out = m_d * v_d.asDiagonal();
  }
}
BENCHMARK(DiagPostMult_New);

BENCHMARK_MAIN();
