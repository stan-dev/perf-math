#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

static void gp_periodic_cov_likely(benchmark::State& state) {
  using stan::math::var;
  using stan::math::promote_scalar;

  auto init = [](benchmark::State& state) {
    auto x_val = stan::math::to_array_1d(Eigen::VectorXd::Random(state.range(0)));
    var sigma = 1.5;
    var length_scale = 2.1;
    var p = 1.7;

    return std::make_tuple(x_val, sigma, length_scale, p);
  };

  auto run = [](const auto&... args) {
    return sum(gp_periodic_cov(args...));
  };

  callback_bench_impl(init, run, state);
}

static void gp_periodic_cov(benchmark::State& state) {
  using stan::math::var;
  using stan::math::promote_scalar;

  auto init = [](benchmark::State& state) {
    auto x_val = stan::math::to_array_1d(Eigen::VectorXd::Random(state.range(0)));
    var sigma = 1.5;
    var length_scale = 2.1;
    var p = 1.7;

    return std::make_tuple(promote_scalar<var>(x_val), sigma, length_scale, p);
  };

  auto run = [](const auto&... args) {
    return sum(gp_periodic_cov(args...));
  };

  callback_bench_impl(init, run, state);
}

static void gp_periodic_cov_vec2(benchmark::State& state) {
  using stan::math::var;
  using stan::math::promote_scalar;

  auto init = [](benchmark::State& state) {
    auto x_val = std::vector<Eigen::VectorXd>(state.range(0), Eigen::VectorXd::Random(2));
    var sigma = 1.5;
    var length_scale = 2.1;
    var p = 1.7;

    return std::make_tuple(promote_scalar<var>(x_val), sigma, length_scale, p);
  };

  auto run = [](const auto&... args) {
    return sum(gp_periodic_cov(args...));
  };

  callback_bench_impl(init, run, state);
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK(gp_periodic_cov_likely)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(gp_periodic_cov)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(gp_periodic_cov_vec2)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
