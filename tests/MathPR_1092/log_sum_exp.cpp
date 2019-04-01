#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <limits>

static void LogSumExp_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;

  MatrixXd m_d = MatrixXd::Random(1000,1000);

  for (auto _ : state) {
    double max = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < m_d.size(); i++)
      if (m_d(i) > max)
        max = m_d(i);

    double sum = 0.0;
    for (int i = 0; i < m_d.size(); i++)
      if (m_d(i) != -std::numeric_limits<double>::infinity())
        sum += std::exp(m_d(i) - max);

    double out = max + std::log(sum);
  }
}
BENCHMARK(LogSumExp_Old);

static void LogSumExp_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;

  MatrixXd m_d = MatrixXd::Random(1000,1000);

  for (auto _ : state) {
  double max = m_d.maxCoeff();
  double out = max + std::log((m_d.array() - max).exp().sum());
  }
}
BENCHMARK(LogSumExp_New);

BENCHMARK_MAIN();
