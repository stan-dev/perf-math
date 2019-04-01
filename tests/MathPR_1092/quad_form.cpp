#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <stan/math/prim/mat/fun/dot_product.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/transpose.hpp>


static void QuadForm_Mat_Vec_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using stan::math::dot_product;
  using stan::math::multiply;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    double out = dot_product(v_d, multiply(m_d, v_d));
  }
}
BENCHMARK(QuadForm_Mat_Vec_Old);

static void QuadForm_Mat_Vec_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  VectorXd v_d = VectorXd::Random(1000);

  for (auto _ : state) {
    double out = v_d.dot(m_d * v_d);
  }
}
BENCHMARK(QuadForm_Mat_Vec_New);

static void QuadForm_Mat_Mat_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using stan::math::multiply;
  using stan::math::transpose;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  MatrixXd m_d2 = MatrixXd::Random(1000, 1000);

  for (auto _ : state) {
    MatrixXd out = multiply(transpose(m_d2), multiply(m_d, m_d2));
  }
}
BENCHMARK(QuadForm_Mat_Mat_Old);

static void QuadForm_Mat_Mat_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;

  MatrixXd m_d = MatrixXd::Random(1000, 1000);
  MatrixXd m_d2 = MatrixXd::Random(1000, 1000);

  for (auto _ : state) {
    MatrixXd out = m_d2.transpose() * m_d * m_d2;
  }
}
BENCHMARK(QuadForm_Mat_Mat_New);

BENCHMARK_MAIN();
