
#include <benchmark/benchmark.h>
#include <stan/math/mix.hpp>
#include <utility>
#include <vector>
#include <type_traits>

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_inner_const(const T1& A, const T2& B) {
  asm ("");
  std::vector<typename std::decay_t<T1>::value_type> C(A.size());
  std::transform(A.begin(), A.end(), B.begin(), C.begin(), [](auto&& x, auto&& y) {
    return x + y;
  });
  return C;
}

template <typename T1, typename T2>
__attribute__ ((noinline, no_icf))  auto add_const(const T1& A, const T2& B) {
  asm ("");
  return add_inner_const(A, B);
}

template <typename T>
inline __attribute__((always_inline)) auto gen_rand_nums(benchmark::State& state) {
  std::vector<T> A(state.range(0));
  std::generate(A.begin(), A.end(), []() {
    return rand() % 100;
  });
  return A;
}

template <typename T>
static void add_constref_bench(benchmark::State& state) {
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<T> result = add_inner_const(gen_rand_nums<T>(state), gen_rand_nums<T>(state));
    auto end = std::chrono::high_resolution_clock::now();
    stan::math::recover_memory();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 8192;
BENCHMARK_TEMPLATE(add_constref_bench, double)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_TEMPLATE(add_constref_bench, stan::math::var)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();

BENCHMARK_MAIN();
