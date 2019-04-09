#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <stan/math/rev/core.hpp>

using Eigen::MatrixXd;
using namespace stan::math;

MatrixXd inplace(MatrixXd& L_A) {
  Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>, Eigen::Lower> L_factor(L_A);
  return L_A;
}

MatrixXd returncopy(const MatrixXd& m) {
  Eigen::LLT<MatrixXd> llt(m.rows());
  llt.compute(m);
  return llt.matrixL();
}

static void ReturnCopy(benchmark::State& state) {
  auto m_d = MatrixXd::Random(500, 500).eval();
  MatrixXd result(500, 500);

  for (auto _ : state) {
    result = returncopy(m_d);
  }
}
BENCHMARK(ReturnCopy);

static void Inplace(benchmark::State& state) {
  auto m_d = MatrixXd::Random(500, 500).eval();
  MatrixXd result(500, 500);

  for (auto _ : state) {
    result = inplace(m_d);
  }
}
BENCHMARK(Inplace);

BENCHMARK_MAIN();
