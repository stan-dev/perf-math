#include <benchmark/benchmark.h>

const int kSize = 1000;

static void push(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<double> v;
    benchmark::DoNotOptimize(v.data());
    for (int i = 0; i < kSize; ++i)
      v.push_back(i + 27.2);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(push);

static void reserve_push(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<double> v;
    v.reserve(kSize);
    benchmark::DoNotOptimize(v.data());
    for (int i = 0; i < kSize; ++i)
      v.push_back(i + 27.2);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(reserve_push);

static void initialize_op_assign(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<double> v(kSize);
    benchmark::DoNotOptimize(v.data());
    for (int i = 0; i < kSize; ++i)
      v[i] = i + 27.2;
    benchmark::ClobberMemory();
  }
}
BENCHMARK(initialize_op_assign);

BENCHMARK_MAIN();
