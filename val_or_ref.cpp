#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <iostream>

using stan::math::var;

template <typename Var, stan::require_var_t<Var>* = nullptr>
inline decltype(auto) value_of_pf(Var&& v) { return std::forward<decltype(v.vi_->val_)>(v.vi_->val_); }



template <typename ComplexT, require_complex_t<ComplexT>* = nullptr, require_vt_arithmetic<ComplexT>* = nullptr>
inline std::complex<double> value_of_x(ComplexT&& x) {
  return x;
}

template <typename ComplexT, require_complex_t<ComplexT>* = nullptr, require_vt_arithmetic<ComplexT>* = nullptr>
inline decltype(auto) value_of_xx(ComplexT&& x) {
  return std::forward<ComplexT>(x);
}

static void val_toss(benchmark::State& state) {
  std::vector<var> vv(256, var(10));
  benchmark::DoNotOptimize(vv.data());
  benchmark::ClobberMemory();
  for (auto _ : state) {
    std::vector<double> v;
    v.reserve(256);
    for (int i = 0; i < vv.size(); ++i) {
      v.push_back(value_of_cp(vv[i]));
    }
    benchmark::DoNotOptimize(v.data());
    benchmark::ClobberMemory();
  }
  stan::math::recover_memory();
}

static void val_pf(benchmark::State& state) {
  std::vector<var> vv(256, var(10));
  benchmark::DoNotOptimize(vv.data());
  benchmark::ClobberMemory();
  for (auto _ : state) {
    std::vector<double> v;
    v.reserve(256);
    for (int i = 0; i < vv.size(); ++i) {
      v.push_back(value_of_cp(vv[i]));
    }
    benchmark::DoNotOptimize(v.data());
    benchmark::ClobberMemory();
  }
  stan::math::recover_memory();
}

static void val_cr(benchmark::State& state) {
  std::vector<var> vv(256, var(10));
  benchmark::DoNotOptimize(vv.data());
  benchmark::ClobberMemory();
  for (auto _ : state) {
    std::vector<double> v;
    v.reserve(256);
    for (int i = 0; i < vv.size(); ++i) {
      v.push_back(value_of_cp(vv[i]));
    }
    benchmark::DoNotOptimize(v.data());
    benchmark::ClobberMemory();
  }
  stan::math::recover_memory();
}

static void val_cp(benchmark::State& state) {
  std::vector<var> vv(256, var(10));
  benchmark::DoNotOptimize(vv.data());
  benchmark::ClobberMemory();
  for (auto _ : state) {
    std::vector<double> v;
    v.reserve(256);
    for (int i = 0; i < vv.size(); ++i) {
      v.push_back(value_of_cp(vv[i]));
    }
    benchmark::DoNotOptimize(v.data());
    benchmark::ClobberMemory();
  }
  stan::math::recover_memory();
}

BENCHMARK(val_toss);
BENCHMARK(val_pf);
BENCHMARK(val_cr);
BENCHMARK(val_cp);
BENCHMARK_MAIN();
