
#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <utility>

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_inner_const(const T1& A, const T2& B) {
  asm ("");
  return A * B;
}

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_inner_pf(T1&& A, T2&& B) {
  asm ("");
  return A * B;
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

static void add_copy_bench(benchmark::State& state) {
  for (auto _ : state) {
     Eigen::MatrixXd result = add_const(Eigen::MatrixXd::Random(state.range(0), state.range(0)),
    Eigen::MatrixXd::Random(state.range(0), state.range(0)));
  }
}

int start_val = 2;
int end_val = 8192;
BENCHMARK(add_copy_bench)->RangeMultiplier(2)->Range(start_val, end_val);

static void add_pf_bench(benchmark::State& state) {
  for (auto _ : state) {
     Eigen::MatrixXd result = add_pf(std::move(Eigen::MatrixXd::Random(state.range(0), state.range(0))),
     std::move(Eigen::MatrixXd::Random(state.range(0), state.range(0))));
  }
}

BENCHMARK(add_pf_bench)->RangeMultiplier(2)->Range(start_val, end_val);

BENCHMARK_MAIN();
