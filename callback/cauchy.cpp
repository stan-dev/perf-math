#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

auto init = [](benchmark::State& state) {
  using stan::math::var;
  using stan::math::exp;
  using stan::math::promote_scalar;

  Eigen::VectorXd y_val = Eigen::VectorXd::Random(state.range(0));
  Eigen::VectorXd mu_val = Eigen::VectorXd::Random(state.range(0));
  Eigen::VectorXd sigma_val = exp(Eigen::VectorXd::Random(state.range(0)));

  return std::make_tuple(promote_scalar<var>(y_val),
			 promote_scalar<var>(mu_val),
			 promote_scalar<var>(sigma_val));
};

auto init_data = [](benchmark::State& state) {
  using stan::math::var;
  using stan::math::exp;
  using stan::math::promote_scalar;

  Eigen::VectorXd y_val = Eigen::VectorXd::Random(state.range(0));
  Eigen::VectorXd mu_val = Eigen::VectorXd::Random(state.range(0));
  Eigen::VectorXd sigma_val = exp(Eigen::VectorXd::Random(state.range(0)));

  return std::make_tuple(y_val,
			 promote_scalar<var>(mu_val),
			 promote_scalar<var>(sigma_val));
};

static void cauchy_lpdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return cauchy_lpdf(args...);
  };

  callback_bench_impl(init, run, state);
}

static void cauchy_cdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return cauchy_cdf(args...);
  };

  callback_bench_impl(init, run, state);
}

static void cauchy_lcdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return cauchy_lcdf(args...);
  };

  callback_bench_impl(init, run, state);
}

static void cauchy_lccdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return cauchy_lccdf(args...);
  };

  callback_bench_impl(init, run, state);
}

static void cauchy_lpdf_data(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return cauchy_lpdf(args...);
  };

  callback_bench_impl(init_data, run, state);
}

static void cauchy_cdf_data(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return cauchy_cdf(args...);
  };

  callback_bench_impl(init_data, run, state);
}

static void cauchy_lcdf_data(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return cauchy_lcdf(args...);
  };

  callback_bench_impl(init_data, run, state);
}

static void cauchy_lccdf_data(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return cauchy_lccdf(args...);
  };

  callback_bench_impl(init_data, run, state);
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK(cauchy_lpdf)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(cauchy_cdf)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(cauchy_lcdf)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(cauchy_lccdf)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(cauchy_lpdf_data)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(cauchy_cdf_data)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(cauchy_lcdf_data)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(cauchy_lccdf_data)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();

