
#include <benchmark/benchmark.h>
#include <stan/math/mix.hpp>
#include <Eigen/Dense>
#include <utility>

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_inner_const(const T1& A, const T2& B) {
  asm ("");
  return A + B;
}

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_const(const T1& A, const T2& B) {
  asm ("");
  return add_inner_const(A, B);
}

template <typename T>
static void add_constref_bench(benchmark::State& state) {
  using mat_type = Eigen::Matrix<T, -1, -1>;
  for (auto _ : state) {
    T result = add_const(std::move(mat_type::Random(state.range(0), state.range(0))),
    std::move(mat_type::Random(state.range(0), state.range(0)))).sum();
    stan::math::recover_memory();
  }
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 8192;
BENCHMARK_TEMPLATE(add_constref_bench, double)->RangeMultiplier(2)->Range(start_val, end_val);
BENCHMARK_TEMPLATE(add_constref_bench, stan::math::var)->RangeMultiplier(2)->Range(start_val, end_val);

BENCHMARK_MAIN();
