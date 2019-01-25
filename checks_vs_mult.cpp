#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <stan/math/rev/core.hpp>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void clobber() {
  asm volatile("" : : : "memory");
}

template <typename T, typename S, typename M>
static void BM_Check(benchmark::State& state) {
  const int N = state.range(0);
  auto sigma = Eigen::Matrix<S, -1, -1>::Identity(N, N).eval();
  auto mu = Eigen::Matrix<M, -1, -1>::Zero(N, 1).eval();
  auto x = Eigen::Matrix<T, -1, -1>::Random(N, 1).eval();

  for (auto _ : state) {
    escape(sigma.data());
    escape(mu.data());
    escape(x.data());
    // Actual task
    if (sigma == Eigen::Matrix<S, -1, -1>::Identity(N, N)) {
      if (mu == Eigen::Matrix<M, -1, 1>::Zero(N, 1))
        escape(x.data());
      else {
        auto res = (mu + x).eval();
        escape(res.data());
      }
    } else {
      auto res = mu + sigma * x;
      auto res_eval = res.eval();
      escape(res_eval.data());
    }
    clobber();
  }
}

template <typename T>
inline bool is_identity(Eigen::Matrix<T, -1, -1> x) {
  for (int j = 0; j < x.rows(); ++j) {
    for (int i = 0; i < j; ++i) {
      if (x(i, j) != 0) return false;
    }
    if (x(j, j) != 1) return false;
    for (int i = j + 1; i < x.rows(); ++i) {
      if (x(i, j) != 0) return false;
    }
  }
  return true;
}

template <typename T, int R, int C>
inline bool is_zero(Eigen::Matrix<T, R, C> x) {
  for (int i = 0; i < x.size(); ++i) {
    if (x.coeff(i) != 0) return false;
  }
  return true;
}

template <typename T, typename S, typename M>
static void BM_BobCheck(benchmark::State& state) {
  const int N = state.range(0);
  auto sigma = Eigen::Matrix<S, -1, -1>::Identity(N, N).eval();
  auto mu = Eigen::Matrix<M, -1, -1>::Zero(N, 1).eval();
  auto x = Eigen::Matrix<T, -1, -1>::Random(N, 1).eval();

  for (auto _ : state) {
    escape(sigma.data());
    escape(mu.data());
    escape(x.data());
    // Actual task
    if (is_identity(sigma)) {
      if (is_zero(mu))
        escape(x.data());
      else {
        auto res = (mu + x).eval();
        escape(res.data());
      }
    } else {
      auto res = mu + sigma * x;
      auto res_eval = res.eval();
      escape(res_eval.data());
    }
    clobber();
  }
}

template <typename T, typename S, typename M>
static void BM_Mult(benchmark::State& state) {
  const int N = state.range(0);
  auto sigma = Eigen::Matrix<S, -1, -1>::Identity(N, N).eval();
  auto mu = Eigen::Matrix<M, -1, -1>::Zero(N, 1).eval();
  auto x = Eigen::Matrix<T, -1, -1>::Random(N, 1).eval();

  for (auto _ : state) {
    escape(sigma.data());
    escape(mu.data());
    escape(x.data());
    // Actual task
    auto res = mu + sigma * x;
    auto res_eval = res.eval();
    escape(res_eval.data());
    clobber();
  }
}
BENCHMARK_TEMPLATE(BM_Check, double, double, double)->Arg(25);
BENCHMARK_TEMPLATE(BM_BobCheck, double, double, double)->Arg(25);
BENCHMARK_TEMPLATE(BM_Mult, double, double, double)->Arg(25);
BENCHMARK_TEMPLATE(BM_Check, double, double, double)->Arg(500);
BENCHMARK_TEMPLATE(BM_BobCheck, double, double, double)->Arg(500);
BENCHMARK_TEMPLATE(BM_Mult, double, double, double)->Arg(500);
BENCHMARK_TEMPLATE(BM_Check, stan::math::var, stan::math::var, stan::math::var)->Arg(25);
BENCHMARK_TEMPLATE(BM_BobCheck, stan::math::var, stan::math::var, stan::math::var)->Arg(25);
BENCHMARK_TEMPLATE(BM_Mult, stan::math::var, stan::math::var, stan::math::var)->Arg(25);
BENCHMARK_TEMPLATE(BM_Check, stan::math::var, stan::math::var, stan::math::var)->Arg(500);
BENCHMARK_TEMPLATE(BM_BobCheck, stan::math::var, stan::math::var, stan::math::var)->Arg(500);
BENCHMARK_TEMPLATE(BM_Mult, stan::math::var, stan::math::var, stan::math::var)->Arg(500);


BENCHMARK_MAIN();
