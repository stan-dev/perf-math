
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
  return A + B;
}

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_inner_copy(T1 A, T2 B) {
  asm ("");
  return A + B;
}

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_const(const T1& A, const T2& B) {
  asm ("");
  return B + add_inner_const(A, B);
}


template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_pf(T1&& A, T2&& B) {
  asm ("");
  return B + add_inner_pf(std::forward<T1>(A), std::forward<T2>(B));
}

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_copy(T1 A, T2 B) {
  asm ("");
  return B + add_inner_copy(A, B);
}

static void add_const_bench(benchmark::State& state) {
  double A = 10.0;
  stan::math::var B(10);
  for (auto _ : state) {
    auto result = add_const(A, B);
    B.grad();
    stan::math::recover_memory();
  }
}

int start_val = 2;
int end_val = 4092;
BENCHMARK(add_const_bench);

static void add_pf_bench(benchmark::State& state) {
  double A = 10.0;
  stan::math::var B(10);
  for (auto _ : state) {
     auto result = add_pf(A, B);
     B.grad();
     stan::math::recover_memory();
  }
}

BENCHMARK(add_pf_bench);

static void add_copy_bench(benchmark::State& state) {
  double A = 10.0;
  stan::math::var B(10);
  for (auto _ : state) {
     auto result = add_copy(A, B);
     B.grad();
     stan::math::recover_memory();
  }
}

BENCHMARK(add_copy_bench);

BENCHMARK_MAIN();
