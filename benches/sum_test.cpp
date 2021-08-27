#include <benchmark/benchmark.h>
#include <stan/math/mix.hpp>
#include <setup_mem.hpp>
#include <Eigen/Dense>
#include <utility>

int start_val = 2;
int end_val = 4096;
constexpr int end_val2 = 6000;
// Build one big matrix of doubles we will reuse for each run of the benchmark
static auto setup_mat() {
  // Use this to put mem on the stack allocator.
  // Should only have to run `setup_mem()` once.
  setup_mem<end_val2>();
  return Eigen::VectorXd::Random(end_val2);
}
Eigen::Matrix<double, -1, 1> x_vals = setup_mat();

// Just to kick off the stack allocation
static void sum_stdvec_bench(benchmark::State& state) {
  using stan::math::var;
  for (auto _ : state) {
    std::vector<var> x(x_vals.data(), x_vals.data() + state.range(0));
    auto start = std::chrono::high_resolution_clock::now();
    var lp = stan::math::sum(x);
    benchmark::DoNotOptimize(lp.vi_);
    benchmark::ClobberMemory();
    lp.grad();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
    stan::math::recover_memory();
  }
}
/*
static void sum_eigen_bench(benchmark::State& state) {
  using stan::math::var;
  for (auto _ : state) {
    Eigen::Matrix<var, -1, 1> x = x_vals.head(state.range(0));
    stan::math::accumulator<var> acc;
    auto start = std::chrono::high_resolution_clock::now();
    var lp = stan::math::sum(x);
    benchmark::DoNotOptimize(lp.vi_);
    benchmark::ClobberMemory();
    lp.grad();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
    stan::math::recover_memory();
  }
}
*/
BENCHMARK(sum_stdvec_bench)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
//BENCHMARK(sum_eigen_bench)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
