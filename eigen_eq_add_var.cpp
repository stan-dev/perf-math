
#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include <chrono>

using stan::math::vari;

class test_adj {
  public:
    stan::math::vari** x_vis_;
    double* x_other;
    template <typename T1>
    void write_eig(T1&& x) {
      using write_map = Eigen::Map<Eigen::Matrix<stan::math::vari*, -1, -1>>;
      write_map(this->x_vis_, x.rows(), x.cols()) = x.vi();
    }
    template <typename T1>
    void write_loop(T1&& x) {
      for (int i = 0; i < x.size(); ++i) {
        x_vis_[i] = x(i).vi_;
      }
   }

   template <typename T1>
   void add_eig(T1&& x) {
     using write_map = Eigen::Map<Eigen::Matrix<double, -1, -1>>;
     write_map(this->x_other, x.rows(), x.cols()) += x;
   }
   template <typename T1>
   void add_loop(T1&& x) {
     for (int i = 0; i < x.size(); ++i) {
       x_other[i] += x(i);
     }
  }

  test_adj(size_t n) {
    this->x_vis_ = stan::math::ChainableStack::instance_->memalloc_.alloc_array<vari*>(n);
    this->x_other = stan::math::ChainableStack::instance_->memalloc_.alloc_array<double>(n);
    for (auto i = 0; i < n; i++) {
      x_other[i] = 0;
    }
  };
};

static void eq_eig_bench(benchmark::State& state) {
  for (auto _ : state) {
    test_adj A(state.range(0) * state.range(0));
    benchmark::DoNotOptimize(A.x_vis_);
    Eigen::Matrix<stan::math::var, -1, -1> B = Eigen::Matrix<stan::math::var, -1, -1>::Random(state.range(0), state.range(0));
    auto start = std::chrono::high_resolution_clock::now();
    A.write_eig(B);
    benchmark::ClobberMemory();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          end - start);
      state.SetIterationTime(elapsed_seconds.count());
      stan::math::recover_memory();
  }
}

static void eq_loop_bench(benchmark::State& state) {
  for (auto _ : state) {
    test_adj A(state.range(0) * state.range(0));
    benchmark::DoNotOptimize(A.x_vis_);
    Eigen::Matrix<stan::math::var, -1, -1> B = Eigen::Matrix<stan::math::var, -1, -1>::Random(state.range(0), state.range(0));
    auto start = std::chrono::high_resolution_clock::now();
    A.write_loop(B);
    benchmark::ClobberMemory();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          end - start);
      state.SetIterationTime(elapsed_seconds.count());
      stan::math::recover_memory();
  }
}

static void add_eig_bench(benchmark::State& state) {
  for (auto _ : state) {
    test_adj A(state.range(0) * state.range(0));
    benchmark::DoNotOptimize(A.x_other);
    Eigen::Matrix<double, -1, -1> B = Eigen::Matrix<double, -1, -1>::Random(state.range(0), state.range(0));
    auto start = std::chrono::high_resolution_clock::now();
    A.add_eig(B);
    benchmark::ClobberMemory();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          end - start);
      state.SetIterationTime(elapsed_seconds.count());
      stan::math::recover_memory();
  }
}

static void add_loop_bench(benchmark::State& state) {
  for (auto _ : state) {
    test_adj A(state.range(0) * state.range(0));
    benchmark::DoNotOptimize(A.x_other);
    Eigen::Matrix<double, -1, -1> B = Eigen::Matrix<double, -1, -1>::Random(state.range(0), state.range(0));
    auto start = std::chrono::high_resolution_clock::now();
    A.add_loop(B);
    benchmark::ClobberMemory();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          end - start);
      state.SetIterationTime(elapsed_seconds.count());
      stan::math::recover_memory();
  }
}

// The start and ending sizes for the benchmark
BENCHMARK(eq_eig_bench)->DenseRange(10, 210, 100)->UseManualTime();
BENCHMARK(eq_loop_bench)->DenseRange(10, 210, 100)->UseManualTime();
BENCHMARK(add_eig_bench)->DenseRange(10, 210, 100)->UseManualTime();
BENCHMARK(add_loop_bench)->DenseRange(10, 210, 100)->UseManualTime();

BENCHMARK_MAIN();
