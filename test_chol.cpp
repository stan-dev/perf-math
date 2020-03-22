
#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include <chrono>


static void chol_bench(benchmark::State& state) {
  for (auto _ : state) {
    Eigen::Matrix<stan::math::var, -1, -1> Sigma(state.range(0), state.range(0));
      for (int i = 0; i < state.range(0); ++i) {
        Sigma(i, i) = 1;
        for (int j = 0; j < i; ++j) {
          Sigma(i, j) = std::pow(0.9, fabs(i - j));
          Sigma(j, i) = Sigma(i, j);
        }
      }
    auto start = std::chrono::high_resolution_clock::now();
      auto result = stan::math::cholesky_decompose(Sigma);
      stan::math::var result2 = sum(result);
      result2.grad();
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          end - start);
      state.SetIterationTime(elapsed_seconds.count());
      stan::math::recover_memory();
  }

}

// The start and ending sizes for the benchmark
BENCHMARK(chol_bench)->DenseRange(100, 2000, 100)->UseManualTime();

BENCHMARK_MAIN();
