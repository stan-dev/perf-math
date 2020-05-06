#include <benchmark/benchmark.h>
#include <stan/math/mix.hpp>
#include <Eigen/Dense>
#include <utility>

// Just to fill up the stack allocator
static void toss_me(benchmark::State& state) {
  using stan::math::var;
  Eigen::Matrix<double, -1, -1> x_vals = Eigen::MatrixXd::Random(1300, 1300);
  Eigen::Matrix<double, -1, -1> y_vals = Eigen::MatrixXd::Random(1300, 1300);
  using stan::math::var_type;
  using stan::math::var;
  using stan::math::sum;
  Eigen::Matrix<var, -1, -1> x = x_vals;
  Eigen::Matrix<var, -1, -1> y = y_vals;
  var lp = 0;
  lp -= sum(x * y + x);
  benchmark::DoNotOptimize(lp.vi_);
  for (auto _ : state) {
    lp.grad();
    benchmark::ClobberMemory();
    stan::math::set_zero_all_adjoints();
  }
  stan::math::recover_memory();
}

// Just to kick off the stack allocation
static void static_matmul_add(benchmark::State& state) {
  using stan::math::var;
  Eigen::Matrix<double, -1, -1> x_vals = Eigen::MatrixXd::Random(state.range(0), state.range(0));
  Eigen::Matrix<double, -1, -1> y_vals = Eigen::MatrixXd::Random(state.range(0), state.range(0));
  using stan::math::var_type;
  using stan::math::var;
  using stan::math::sum;
  var_type<Eigen::Matrix<double, -1, -1>> x = x_vals;
  var_type<Eigen::Matrix<double, -1, -1>> y = y_vals;
  var lp = 0;
  lp -= sum(x * y + x);
  benchmark::DoNotOptimize(lp.vi_);
  for (auto _ : state) {
    lp.grad();
    benchmark::ClobberMemory();
    stan::math::set_zero_all_adjoints();
  }
  stan::math::recover_memory();
}

static void dynamic_matmul_add(benchmark::State& state) {
  using stan::math::var;
  Eigen::Matrix<double, -1, -1> x_vals = Eigen::MatrixXd::Random(state.range(0), state.range(0));
  Eigen::Matrix<double, -1, -1> y_vals = Eigen::MatrixXd::Random(state.range(0), state.range(0));
  using stan::math::var_type;
  using stan::math::var;
  using stan::math::sum;
  Eigen::Matrix<var, -1, -1> x = x_vals;
  Eigen::Matrix<var, -1, -1> y = y_vals;
  var lp = 0;
  lp -= sum(x * y + x);
  benchmark::DoNotOptimize(lp.vi_);
  for (auto _ : state) {
    lp.grad();
    benchmark::ClobberMemory();
    stan::math::set_zero_all_adjoints();
  }
  stan::math::recover_memory();
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK(static_matmul_add)->RangeMultiplier(2)->Range(start_val, end_val);
BENCHMARK(dynamic_matmul_add)->RangeMultiplier(2)->Range(start_val, end_val);
BENCHMARK_MAIN();
