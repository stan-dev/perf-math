#include <benchmark/benchmark.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;

static void BM_EigenElementAccess(benchmark::State& state) {
  auto m_d = MatrixXd::Random(500, 500).eval();

  for (auto _ : state) {
    benchmark::DoNotOptimize(m_d.data());
    m_d(400);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_EigenElementAccess);

static void BM_EigenCoeff(benchmark::State& state) {
  auto m_d = MatrixXd::Random(500, 500).eval();

  for (auto _ : state) {
    // Actual task
    benchmark::DoNotOptimize(m_d.data());
    m_d.coeff(400);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_EigenCoeff);

static void BM_VectorElementAccess(benchmark::State& state) {
  auto m_d = MatrixXd::Random(500, 500).eval();
  std::vector<double> v_d;
  for (int i = 0; i < m_d.size(); i++) {
    v_d.push_back(m_d(i));
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(v_d.data());
    v_d[400];
    benchmark::ClobberMemory();
  }

}
BENCHMARK(BM_VectorElementAccess);

BENCHMARK_MAIN();
