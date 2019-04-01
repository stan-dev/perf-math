#include <benchmark/benchmark.h>
#include <Eigen/Dense>

static void Add_Mat_Dbl_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  double d = m_d(1,1);

  for (auto _ : state) {
    MatrixXd out(1000,1000);

    for (int i = 0; i < out.size(); ++i)
      out(i) = m_d(i) + d;
  }
}
BENCHMARK(Add_Mat_Dbl_Old);

static void Add_Mat_Dbl_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  double d = m_d(1,1);

  for (auto _ : state) {
    MatrixXd out = m_d.array() + d;
  }
}
BENCHMARK(Add_Mat_Dbl_New);

static void Add_Mat_Mat_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  MatrixXd m_d2 = MatrixXd::Random(1000, 1000);

  for (auto _ : state) {
    MatrixXd out(1000,1000);

    for (int i = 0; i < out.size(); ++i)
      out(i) = m_d(i) + m_d2(i);
  }
}
BENCHMARK(Add_Mat_Mat_Old);

static void Add_Mat_Mat_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  MatrixXd m_d2 = MatrixXd::Random(1000, 1000);

  for (auto _ : state) {
    MatrixXd out = m_d + m_d2;
  }
}
BENCHMARK(Add_Mat_Mat_New);

BENCHMARK_MAIN();
