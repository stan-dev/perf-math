#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <stan/math/rev/core.hpp>

const int R = -1;
const int C = -1;

static void BM_LogOrig(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using namespace stan::math;
  auto m_d = MatrixXd::Random(500, 500);

  Eigen::FullPivHouseholderQR<Matrix<double, R, C> > hh
      = m_d.fullPivHouseholderQr();

  for (auto _ : state) {
    double* gradients
        = ChainableStack::instance().memalloc_.alloc_array<double>(m_d.size());
    Eigen::Map<Matrix<double, R, C>> m_inv_transpose(gradients, hh.rows(), hh.cols());
    m_inv_transpose = hh.inverse().transpose();
  }
}
BENCHMARK(BM_LogOrig);

static void BM_LogProposed(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using namespace stan::math;

  auto m_d = MatrixXd::Random(500, 500);
  Eigen::FullPivHouseholderQR<Matrix<double, R, C> > hh
      = m_d.fullPivHouseholderQr();

  for (auto _ : state) {
    Matrix<double, R, C> m_inv_transpose = hh.inverse().transpose();
    double* gradients
        = ChainableStack::instance().memalloc_.alloc_array<double>(m_d.size());
    for (int i = 0; i < m_d.size(); ++i)
      gradients[i] = m_inv_transpose(i);
  }
}
BENCHMARK(BM_LogProposed);

BENCHMARK_MAIN();
