#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <stan/math/rev/core.hpp>

const int R = -1;
const int C = -1;

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void clobber() {
  asm volatile("" : : : "memory");
}

static void BM_EigenElementAccess(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using namespace stan::math;
  auto m_d = MatrixXd::Random(500, 500).eval();

  for (auto _ : state) {
    // Actual task
    escape(m_d.data());
    m_d(400);
    clobber();
  }
}
BENCHMARK(BM_EigenElementAccess);

static void BM_VectorElementAccess(benchmark::State& state) {
  using std::vector;
  using Eigen::MatrixXd;
  using namespace stan::math;

  auto m_d = MatrixXd::Random(500, 500).eval();
  std::vector<double> v_d;
  for (int i = 0; i < m_d.size(); i++) {
    v_d.push_back(m_d(i));
  }

  for (auto _ : state) {
    escape(v_d.data());
    v_d[400];
    clobber();
  }

}
BENCHMARK(BM_VectorElementAccess);

BENCHMARK_MAIN();
