
#include <benchmark/benchmark.h>
#include <stan/math/mix.hpp>
#include <utility>
#include <vector>
#include <type_traits>

namespace stan {
  namespace math {
    /*
     * This is the simplest adj_jac functor in town
     */
    struct AddFunctor {
      template <std::size_t size>
      inline Eigen::VectorXd operator()(const std::array<bool, size>& needs_adj,
         const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) {
	       check_size_match("AddFunctor::operator()", "x1", x1.size(), "x2", x2.size());
	       return x1 + x2;
      }

      template <std::size_t size>
      inline auto multiply_adjoint_jacobian(const std::array<bool, size>& needs_adj,
				     const Eigen::VectorXd& adj) {
	      return std::make_tuple(adj, adj);
      }
    };

    const auto AddFunctorAutodiffed = [](auto&& x1, auto&& x2) {
      check_size_match("AddFunctorAutodiffed::operator()", "x1", x1.size(), "x2", x2.size());
      return (x1 + x2).eval();
    };
  }
}

static void add_functor_bench(benchmark::State& state) {
  for (auto _ : state) {
    Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> x1(state.range(0));
    Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> x2(state.range(0));
    stan::math::recover_memory();
    for (int i = 0; i < state.range(0); ++i) {
      x1(i) = i / static_cast<double>(state.range(0));
      x2(i) = i / static_cast<double>(state.range(0));
    }
    benchmark::DoNotOptimize(x1.data());
    benchmark::DoNotOptimize(x2.data());
    benchmark::ClobberMemory();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 500; ++i) {
      stan::math::start_nested();
      stan::math::var y = stan::math::sum(stan::math::AddFunctorAutodiffed(x1, x2));
      benchmark::DoNotOptimize(y.vi_);
      stan::math::grad(y.vi_);
      benchmark::ClobberMemory();
      stan::math::recover_memory_nested();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

static void add_adj_jac_bench(benchmark::State& state) {
  for (auto _ : state) {
    Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> x1(state.range(0));
    Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> x2(state.range(0));
    stan::math::recover_memory();
    for (int i = 0; i < state.range(0); ++i) {
      x1(i) = i / static_cast<double>(state.range(0));
      x2(i) = i / static_cast<double>(state.range(0));
    }
    benchmark::DoNotOptimize(x1.data());
    benchmark::DoNotOptimize(x2.data());
    benchmark::ClobberMemory();
    const auto add_func = [](auto&& x1, auto&& x2) { return stan::math::adj_jac_apply<stan::math::AddFunctor>(x1, x2); };
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 500; ++i) {
      stan::math::start_nested();
      stan::math::var y = stan::math::sum(add_func(x1, x2));
      benchmark::DoNotOptimize(y.vi_);
      stan::math::grad(y.vi_);
      benchmark::ClobberMemory();
      stan::math::recover_memory_nested();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

BENCHMARK(add_functor_bench)->DenseRange(128, 1024, 128)->UseManualTime();
BENCHMARK(add_adj_jac_bench)->DenseRange(128, 1024, 128)->UseManualTime();
BENCHMARK_MAIN();

/*

TEST(AgradRev, benchmarks) {
  std::cout << "method, N, time" << std::endl;
  for(int N = 100; N <= 500; N += 100) {
    for(int j = 0; j < 5; ++j) {
      std::cout << "autodiffed, " << N << ", " << run_benchmark(, N, 500) << std::endl;
      std::cout << "inefficient, " << N << ", " << run_benchmark(, N, 500) << std::endl;
      std::cout << "efficient, " << N << ", " << run_benchmark([](auto x1, auto x2) { return stan::math::adj_jac_apply<stan::math::AddFunctor>(x1, x2); }, N, 500) << std::endl;
    }
  }
}
*/
