
#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <Eigen/Dense>
#include <utility>

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_inner_const(const T1& A, const T2& B) {
  asm ("");
  return A + B;
}

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_inner_pf(T1&& A, T2&& B) {
  asm ("");
  return std::forward<T1>(A) + std::forward<T2>(B);
}

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_const(const T1& A, const T2& B) {
  asm ("");
  return add_inner_const(A, B);
}


template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_pf(T1&& A, T2&& B) {
  asm ("");
  return add_inner_pf(std::forward<T1>(A), std::forward<T2>(B));
}

static void add_constref_bench(benchmark::State& state) {
  using stan::math::var;
  for (auto _ : state) {
    var result = add_pf(std::move(Eigen::Matrix<var, -1, -1>::Random(state.range(0), state.range(0))),
    std::move(Eigen::Matrix<var, -1, -1>::Random(state.range(0), state.range(0)))).sum();
    result.grad();
    stan::math::recover_memory();
  }
}

int start_val = 2;
int end_val = 2048;
BENCHMARK(add_constref_bench)->RangeMultiplier(2)->Range(start_val, end_val);

static void add_pf_bench(benchmark::State& state) {
  using stan::math::var;
  for (auto _ : state) {
     var result = add_pf(std::move(Eigen::Matrix<var, -1, -1>::Random(state.range(0), state.range(0))),
     std::move(Eigen::Matrix<var, -1, -1>::Random(state.range(0), state.range(0)))).sum();
     result.grad();
     stan::math::recover_memory();
  }
}

BENCHMARK(add_pf_bench)->RangeMultiplier(2)->Range(start_val, end_val);

BENCHMARK_MAIN();
