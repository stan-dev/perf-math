#define STAN_OPENCL
#define OPENCL_DEVICE_ID 0
#define OPENCL_PLATFORM_ID 2
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <benchmark/benchmark.h>
#include <stan/math.hpp>

using namespace stan::math;

static void matvec_mul(benchmark::State& state) {
  matrix_cl<double> m(Eigen::Matrix<double, -1, -1>::Random(state.range(0), state.range(1)));
  matrix_cl<double> v(Eigen::Matrix<double, -1, 1>::Random(state.range(1)));
  //  matrix_cl<double> res = m * v;
  matrix_cl<double> res(state.range(0),1);// = matrix_vector_multiply(m, v);
  //res.wait_for_write_events();
  benchmark::ClobberMemory();
  for (auto _ : state) {
        res = m * v;
//    res = matrix_vector_multiply(m, v);
    res.wait_for_write_events();
  }
}
BENCHMARK(matvec_mul)->Ranges({{32, 10000}, {32, 10000}});
BENCHMARK_MAIN();
