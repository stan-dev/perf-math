#include <benchmark/benchmark.h>
#include <stan/io/reader.hpp>
#include <stan/math.hpp>
#include <iostream>

static void multi_init_var_mat(benchmark::State& state) {
  { // Force read/write of vector of vars to trigger the stack allocator
    std::vector<stan::math::var> dummy_var_(state.range(0) * state.range(0), 10);
    benchmark::DoNotOptimize(dummy_var_.data());
    benchmark::ClobberMemory();
    stan::math::recover_memory();
  }
  for (auto _ : state) {
    std::vector<stan::math::var> params_r__(state.range(0) * state.range(0), 10);
    std::vector<int> params_i__(0);
    stan::io::reader<stan::math::var> in__(params_r__, params_i__);
    benchmark::DoNotOptimize(params_r__.data());
    benchmark::ClobberMemory();
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::Matrix<stan::math::var, -1, -1> init_mat;
    benchmark::DoNotOptimize(init_mat.data());
    init_mat = Eigen::Matrix<stan::math::var, -1, -1>(state.range(0), state.range(0));
    init_mat = in__.matrix(state.range(0), state.range(0));
    benchmark::ClobberMemory();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    stan::math::recover_memory();
  }
}

static void single_init_var_mat(benchmark::State& state) {
  { // Force read/write of vector of vars to trigger the stack allocator
    std::vector<stan::math::var> dummy_var_(state.range(0) * state.range(0), 10);
    benchmark::DoNotOptimize(dummy_var_.data());
    benchmark::ClobberMemory();
    stan::math::recover_memory();
  }
  for (auto _ : state) {
    std::vector<stan::math::var> params_r__(state.range(0) * state.range(0)), 10;
    std::vector<int> params_i__(0);
    stan::io::reader<stan::math::var> in__(params_r__, params_i__);
    benchmark::DoNotOptimize(params_r__.data());
    benchmark::ClobberMemory();
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::Matrix<stan::math::var, -1, -1> init_mat;
    benchmark::DoNotOptimize(init_mat.data());
    init_mat = in__.matrix(state.range(0), state.range(0));
    benchmark::ClobberMemory();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
    stan::math::recover_memory();
  }
}

static void multi_init_double_mat(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<double> params_r__(state.range(0) * state.range(0));
    for (int i = 0; i < state.range(0) * state.range(0); ++i) {
      params_r__[i] = static_cast<double>(rand() % 100);
    }
    std::vector<int> params_i__(state.range(0) * state.range(0));
    stan::io::reader<double> in__(params_r__, params_i__);
    benchmark::DoNotOptimize(params_r__.data());
    benchmark::ClobberMemory();
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::Matrix<double, -1, -1> init_mat;
    benchmark::DoNotOptimize(init_mat.data());
    init_mat = Eigen::Matrix<double, -1, -1>(state.range(0), state.range(0));
    init_mat = in__.matrix(state.range(0), state.range(0));
    benchmark::ClobberMemory();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

static void single_init_double_mat(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<double> params_r__(state.range(0) * state.range(0));
    for (int i = 0; i < state.range(0) * state.range(0); ++i) {
      params_r__[i] = static_cast<double>(rand() % 100);
    }
    std::vector<int> params_i__(state.range(0) * state.range(0));
    stan::io::reader<double> in__(params_r__, params_i__);
    benchmark::DoNotOptimize(params_r__.data());
    benchmark::ClobberMemory();
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::Matrix<double, -1, -1> init_mat;
    benchmark::DoNotOptimize(init_mat.data());
    init_mat = in__.matrix(state.range(0), state.range(0));
    benchmark::ClobberMemory();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}


// The start and ending sizes for the benchmark
int start_val = 128;
int end_val = 4096;
BENCHMARK(single_init_var_mat)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(multi_init_var_mat)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(multi_init_double_mat)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(single_init_double_mat)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();

BENCHMARK_MAIN();
