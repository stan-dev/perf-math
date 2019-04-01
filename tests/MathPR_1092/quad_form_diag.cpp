#include <benchmark/benchmark.h>
#include <Eigen/Dense>


static void QuadFormDiag_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    MatrixXd result(1000, 1000);
    for (int i = 0; i < 1000; i++) {
      result(i, i) = v_d(i) * v_d(i) * m_d(i, i);
      for (int j = i + 1; j < 1000; ++j) {
        double temp = v_d(i) * v_d(j);
        result(j, i) = temp * m_d(j, i);
        result(i, j) = temp * m_d(i, j);
      }
    }
  }
}
BENCHMARK(QuadFormDiag_Old);

static void QuadFormDiag_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    MatrixXd result = v_d.asDiagonal() * m_d * v_d.asDiagonal();
  }
}
BENCHMARK(QuadFormDiag_New);

BENCHMARK_MAIN();
