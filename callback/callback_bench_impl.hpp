#ifndef CALLBACK_BENCH_IMPL_HPP
#define CALLBACK_BENCH_IMPL_HPP

#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>

template <typename F_init, typename F_run>
static void callback_bench_impl(F_init init, F_run run, benchmark::State& state) {
  using stan::math::var;
  using stan::math::promote_scalar_t;

  Eigen::MatrixXd x_val = Eigen::MatrixXd::Random(state.range(0), state.range(0));
  Eigen::MatrixXd y_val = Eigen::MatrixXd::Random(state.range(0), state.range(0));

  for (auto _ : state) {
    auto init_tuple = init(state);

    auto start = std::chrono::high_resolution_clock::now();
    var lp = 0;
    lp += stan::math::apply(run, init_tuple);
    lp.grad();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());
    stan::math::recover_memory();
    benchmark::ClobberMemory();
  }
}

#endif

