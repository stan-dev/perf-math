#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"

static void dot_product(benchmark::State& state) {
  using stan::math::var;
  using stan::math::promote_scalar_t;

  Eigen::VectorXd x_vals = Eigen::VectorXd::Random(state.range(0));
  Eigen::VectorXd y_vals = Eigen::VectorXd::Random(state.range(0));

  for (auto _ : state) {
    promote_scalar_t<var, Eigen::VectorXd> x = x_vals;
    promote_scalar_t<var, Eigen::VectorXd> y = y_vals;

    auto start = std::chrono::high_resolution_clock::now();
    var lp = 0;
    lp -= dot_product(x, y);
    lp.grad();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());
    stan::math::recover_memory();
    benchmark::ClobberMemory();
  }
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK(dot_product)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();

