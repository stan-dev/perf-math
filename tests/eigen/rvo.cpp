#include <benchmark/benchmark.h>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core.hpp>

using Eigen::MatrixXd;
using matrix_v = Eigen::Matrix<stan::math::var, -1, -1>;

template <typename T>
void inout_copy(T& dst, const T& src) {
  for (int i = 0; i < src.size(); i++)
    dst.coeffRef(i) = src.coeff(i);
}

template <typename T>
T rv_copy(const T& src) {
  T dst(src.rows(), src.cols());
  for (int i = 0; i < src.size(); i++)
    dst.coeffRef(i) = src.coeff(i);
  return dst;
}

template <typename T>
T mkmat(benchmark::State& state) {
  const int R = state.range(0);
  const int C = state.range(1);
  return T::Random(R, C).eval();
}

template <typename T>
static void return_value(benchmark::State& state) {
  auto m_d = mkmat<T>(state);

  for (auto _ : state) {
    T res;
    benchmark::DoNotOptimize(res.data());
    res = rv_copy(m_d);
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(return_value, MatrixXd)->Args({1000, 1000})->Args({5000, 5000})->Args({10000, 10000});
BENCHMARK_TEMPLATE(return_value, matrix_v)->Args({1000, 1000});

template <typename T>
static void inout(benchmark::State& state) {
  auto m_d = mkmat<T>(state);

  for (auto _ : state) {
    T res(state.range(0), state.range(1));
    benchmark::DoNotOptimize(res.data());
    inout_copy(res, m_d);
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(inout, MatrixXd)->Args({1000, 1000})->Args({5000, 5000})->Args({10000, 10000});
BENCHMARK_TEMPLATE(inout, matrix_v)->Args({1000, 1000});

BENCHMARK_MAIN();
