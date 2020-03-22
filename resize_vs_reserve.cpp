
#include <benchmark/benchmark.h>
#include <utility>
#include <chrono>


static void resize_bench(benchmark::State& state) {

  for (auto _ : state) {
    std::vector<double> a(0);
    a.resize(state.range(0));
    benchmark::DoNotOptimize(a.data());
    for (int i = 0; i < a.size(); i++) {
      a[i] = 10.0;
    }
    benchmark::ClobberMemory();
  }
}

static void reserve_bench(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<double> a(0);
    a.reserve(state.range(0));
    benchmark::DoNotOptimize(a.data());
    for (int i = 0; i < state.range(0); i++) {
      a[i] = 10.0;
    }
    benchmark::ClobberMemory();
  }
}

static void reserve_pushback_bench(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<double> a(0);
    a.reserve(state.range(0));
    benchmark::DoNotOptimize(a.data());
    for (int i = 0; i < state.range(0); i++) {
      a.push_back(10.0);
    }
    benchmark::ClobberMemory();
  }
}

// The start and ending sizes for the benchmark
BENCHMARK(resize_bench)->DenseRange(10, 2010, 1000);
BENCHMARK(reserve_bench)->DenseRange(10, 2010, 1000);
BENCHMARK(reserve_pushback_bench)->DenseRange(10, 2010, 1000);

BENCHMARK_MAIN();
