#include <benchmark/benchmark.h>
#include <stan/math/mix.hpp>
#include <toss_me.hpp>
#include <Eigen/Dense>
#include <utility>


// Just to kick off the stack allocation
static void add_stdvecs(benchmark::State& state) {
  using stan::math::var;
  Eigen::Matrix<double, -1, 1> x_vals = Eigen::VectorXd::Random(state.range(0));
  Eigen::Matrix<double, -1, 1> y_vals = Eigen::VectorXd::Random(state.range(0));
  Eigen::Matrix<var, -1, -1> x = x_vals;
  Eigen::Matrix<var, -1, -1> y = y_vals;
  var lp = 0;
  for (Eigen::Index i = 0; i < x_vals.size(); ++i) {
    lp += x.coeffRef(i) * y.coeffRef(i) + x.coeffRef(i);
  }
  benchmark::DoNotOptimize(lp.vi_);
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    lp.grad();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
    stan::math::set_zero_all_adjoints();
    benchmark::ClobberMemory();
  }
  stan::math::recover_memory();
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 1024;
constexpr int end_val2 = 2048;
BENCHMARK_TEMPLATE(toss_me, end_val2);
BENCHMARK(add_stdvecs)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
