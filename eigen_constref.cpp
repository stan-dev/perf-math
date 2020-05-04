
#include <benchmark/benchmark.h>
#include <stan/math/mix.hpp>
#include <Eigen/Dense>
#include <utility>

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_inner_const(const T1& A, const T2& B) {
  asm ("");
  return (A + B).eval();
}

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_inner_pf(T1&& A, T2&& B) {
  asm ("");
  return (std::forward<T1>(A) + std::forward<T2>(B)).eval();
}

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_inner_copy(T1 A, T2 B) {
  asm ("");
  return (A + B).eval();
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

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_copy(T1 A, T2 B) {
  asm ("");
  return add_inner_copy(A, B);
}

static void add_constref_bench(benchmark::State& state) {
  using mat_type = Eigen::Matrix<double, -1, -1>;
  for (auto _ : state) {
    Eigen::Matrix<double, -1, -1> result = add_const(std::move(mat_type::Random(state.range(0), state.range(0))),
    std::move(mat_type::Random(state.range(0), state.range(0))));
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

static void add_pf_bench(benchmark::State& state) {
  using mat_type = Eigen::Matrix<double, -1, -1>;
  for (auto _ : state) {
    Eigen::Matrix<double, -1, -1> result = add_pf(std::move(mat_type::Random(state.range(0), state.range(0))),
    std::move(mat_type::Random(state.range(0), state.range(0))));
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

static void add_copy_bench(benchmark::State& state) {
  using mat_type = Eigen::Matrix<double, -1, -1>;
  for (auto _ : state) {
    Eigen::Matrix<double, -1, -1> result = add_copy(std::move(mat_type::Random(state.range(0), state.range(0))),
    std::move(mat_type::Random(state.range(0), state.range(0))));
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 8192;
BENCHMARK(add_constref_bench)->RangeMultiplier(2)->Range(start_val, end_val);
BENCHMARK(add_pf_bench)->RangeMultiplier(2)->Range(start_val, end_val);
BENCHMARK(add_copy_bench)->RangeMultiplier(2)->Range(start_val, end_val);
BENCHMARK_MAIN();
