#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <stan/math/prim/scal/meta/scalar_seq_view.hpp>

const int R = -1;
const int C = -1;

static void AddDiag_Mat_Dbl_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using namespace stan::math;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  double d = m_d(1,1);

  for (auto _ : state) {
    MatrixXd out = m_d;
    stan::scalar_seq_view<double> to_add_vec(d);
    size_t length_diag = std::min(m_d.rows(), m_d.cols());

    for (size_t i = 0; i < length_diag; ++i)
      out(i, i) += to_add_vec[i];
  }
}
BENCHMARK(AddDiag_Mat_Dbl_Old);

static void AddDiag_Mat_Dbl_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using namespace stan::math;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  double d = m_d(1,1);

  for (auto _ : state) {
    MatrixXd out = m_d;
    out.diagonal().array() += d;
  }
}
BENCHMARK(AddDiag_Mat_Dbl_New);

static void AddDiag_Mat_Mat_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using namespace stan::math;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    MatrixXd out = m_d;
    stan::scalar_seq_view<VectorXd> to_add_vec(v_d);
    size_t length_diag = std::min(m_d.rows(), m_d.cols());

    for (size_t i = 0; i < length_diag; ++i)
      out(i, i) += to_add_vec[0][i];
  }
}
BENCHMARK(AddDiag_Mat_Mat_Old);

static void AddDiag_Mat_Mat_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using namespace stan::math;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    MatrixXd out = m_d;
    out.diagonal() += v_d;
  }
}
BENCHMARK(AddDiag_Mat_Mat_New);

BENCHMARK_MAIN();
