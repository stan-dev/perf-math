#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <vector>
#include <iostream>


inline auto make_var_vec(int sizer) {
    std::vector<stan::math::var> vec(sizer);
    for (int i = 0; i < sizer; i++) {
        vec.emplace_back(new stan::math::vari(static_cast<double>(i)));
    }
  return vec;
}

inline auto make_double_vec(int sizer) {
    std::vector<double> vec(sizer);
    for (int i = 0; i < sizer; i++) {
        vec.emplace_back(static_cast<double>(i));
    }
  return vec;
}

static void multi_init_var_vec(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<stan::math::var> init_vec;
    benchmark::DoNotOptimize(init_vec.data());
    init_vec = std::vector<stan::math::var>(state.range(0));
    init_vec = make_var_vec(state.range(0));
    benchmark::ClobberMemory();
    stan::math::recover_memory();
  }
}

static void single_init_var_vec(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<stan::math::var> init_vec;
    benchmark::DoNotOptimize(init_vec.data());
    init_vec = make_var_vec(state.range(0));
    benchmark::ClobberMemory();
    stan::math::recover_memory();

  }
}

static void multi_init_double_vec(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<double> init_vec;
    benchmark::DoNotOptimize(init_vec.data());
    init_vec = std::vector<double>(state.range(0));
    init_vec = make_double_vec(state.range(0));
    benchmark::ClobberMemory();
  }
}

static void single_init_double_vec(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<double> init_vec;
    benchmark::DoNotOptimize(init_vec.data());
    init_vec = make_double_vec(state.range(0));
    benchmark::ClobberMemory();
  }
}


// The start and ending sizes for the benchmark
int start_val = 128;
int end_val = 8192;
BENCHMARK(multi_init_var_vec)->RangeMultiplier(2)->Range(start_val, end_val);
BENCHMARK(single_init_var_vec)->RangeMultiplier(2)->Range(start_val, end_val);
BENCHMARK(multi_init_double_vec)->RangeMultiplier(2)->Range(start_val, end_val);
BENCHMARK(single_init_double_vec)->RangeMultiplier(2)->Range(start_val, end_val);

BENCHMARK_MAIN();
