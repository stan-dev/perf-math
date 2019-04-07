
#include <benchmark/benchmark.h>
#include <CL/cl.hpp>

const int kSize = 2000;

static void push(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<cl::Event> v;
    benchmark::DoNotOptimize(v.data());
    cl::Event test_event;
    for (int i = 0; i < kSize; ++i) {
      v.push_back(test_event);
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK(push);


static void reserve_push(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<cl::Event> v;
    v.reserve(kSize);
    benchmark::DoNotOptimize(v.data());
    cl::Event test_event;
    for (int i = 0; i < kSize; ++i) {
      v.push_back(test_event);
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK(reserve_push);

static void emplace(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<cl::Event> v;
    benchmark::DoNotOptimize(v.data());
    cl::Event test_event;
    for (int i = 0; i < kSize; ++i) {
      v.emplace_back(test_event);
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK(emplace);

static void reserve_emplace(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<cl::Event> v;
    v.reserve(kSize);
    benchmark::DoNotOptimize(v.data());
    cl::Event test_event;
    for (int i = 0; i < kSize; ++i) {
      v.emplace_back(test_event);
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK(reserve_emplace);

static void initialize_op_assign(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<cl::Event> v(kSize);
    benchmark::DoNotOptimize(v.data());
    cl::Event test_event;
    for (int i = 0; i < kSize; ++i) {
      v[i] = test_event;
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK(initialize_op_assign);

BENCHMARK_MAIN();
