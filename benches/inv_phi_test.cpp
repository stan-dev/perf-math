#include <benchmark/benchmark.h>
#include <stan/math/mix.hpp>
#include <setup_mem.hpp>
#include <Eigen/Dense>
#include <utility>

constexpr Eigen::Index start_size = 2;
constexpr Eigen::Index end_size = 4096;
// Build one big matrix of doubles we will reuse for each run of the benchmark
static auto setup_mat() {
  // Use this to put mem on the stack allocator.
  // Should only have to run `setup_mem()` once.
  setup_mem<end_size>();
  return stan::math::inv_logit(Eigen::VectorXd::Random(end_size));
}
Eigen::Matrix<double, -1, 1> x_vals = setup_mat();

// Just to kick off the stack allocation
static void inv_phi_eigen_bench(benchmark::State& state) {
  using stan::math::var;
  for (auto _ : state) {
    Eigen::Matrix<var, -1, 1> x = Eigen::Map<Eigen::Matrix<double, -1, 1>>(x_vals.data(), state.range(0));
    // Start time here to only measure forward and reverse pass
    // importantly this excludes the time for memory cleanup.
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::Matrix<var, -1, 1> res = stan::math::inv_Phi(x);
    benchmark::DoNotOptimize(res.data());
    benchmark::ClobberMemory();
    // Here set all the adjoints to one then calls for the gradient.
    res.adj().setOnes();
    stan::math::grad();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
    stan::math::recover_memory();
  }
}

BENCHMARK(inv_phi_eigen_bench)->RangeMultiplier(2)->Range(start_size, end_size)->UseManualTime();
BENCHMARK_MAIN();
