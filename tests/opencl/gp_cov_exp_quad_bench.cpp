
#include <stan/math.hpp>
#include <stan/math/opencl/copy.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/rev/mat/fun/gp_exp_quad_cov.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <limits.h>
#include <vector>

using namespace Eigen;
using namespace std;
using namespace stan::math;

template <typename... Args>
void print_speed(const std::string test_type, const std::string test_ret, const Args... args) {
  std::string foo = "%d, %d, %d, %d \n";
  foo = test_type + ", " + test_ret + ", " + foo;
  printf(foo.c_str(), args...);
}

auto set_tuning_opts_to_use_gpu(const std::string test_type) {
  if (test_type.compare("gpu") == 0) {
    stan::math::opencl_context.tuning_opts().gp_exp_quad_cov_simple = 1;
    stan::math::opencl_context.tuning_opts().gp_exp_quad_cov_vec = 1;
  } else {
    stan::math::opencl_context.tuning_opts().gp_exp_quad_cov_simple = INT_MAX;
    stan::math::opencl_context.tuning_opts().gp_exp_quad_cov_vec = INT_MAX;
  }
}

void test_speed(const std::string test_type, int size = 3000) {
  vector<double> x(size);
  for (int i = 0; i < size; i++) {
    x[i] = i * i;
  }

  const double sigma = 1.6;
  const double length_scale = 2.7;

  auto start = std::chrono::steady_clock::now();
  MatrixXd res_cpu = gp_exp_quad_cov(x, sigma, length_scale);
  int time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  print_speed(test_type, "1", size, 0, 1, time);
}

void test_speed_2(const std::string test_type, int size_x = 10000, int size_y = 5000) {
  vector<double> x(size_x);
  for (int i = 0; i < size_x; i++) {
    x[i] = i * i;
  }
  vector<double> y(size_y);
  for (int i = 0; i < size_y; i++) {
    y[i] = i * i * 0.1;
  }
  const double sigma = 1.6;
  const double length_scale = 2.7;

  auto start = std::chrono::steady_clock::now();
  MatrixXd res_cpu = gp_exp_quad_cov(x, y, sigma, length_scale);
  int time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  print_speed(test_type, "2", size_x, size_y, 1, time);
}

void test_speed_v(const std::string test_type, int size = 10000, int size_l = 100) {
  vector<Eigen::Matrix<double, -1, 1>> x(size);
  for (int i = 0; i < size; i++) {
    x[i].resize(size_l, 1);
    for (int j = 0; j < size_l; j++) {
      x[i](j) = j * j + i;
    }
  }
  const double sigma = 1.6;
  const double length_scale = 2.5;

  auto start = std::chrono::steady_clock::now();
  MatrixXd res_cpu = gp_exp_quad_cov(x, sigma, length_scale);
  int time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  print_speed(test_type, "1v", size, 0, size_l, time);
}

void test_speed_vv(const std::string test_type, int size = 10000, int size_l = 100) {
  vector<Eigen::Matrix<double, -1, 1>> x(size);
  for (int i = 0; i < size; i++) {
    x[i].resize(size_l, 1);
    for (int j = 0; j < size_l; j++) {
      x[i](j) = j * j + i;
    }
  }
  const double sigma = 1.6;

  vector<double> length_scale(size_l);
  for (int i = 0; i < size_l; i++) {
    length_scale[i] = i * 0.1 + 1;
  }

  auto start = std::chrono::steady_clock::now();
  MatrixXd res_cpu = gp_exp_quad_cov(x, sigma, length_scale);
  int time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  print_speed(test_type, "1vv", size, 0, size_l, time);
}

void test_speed_2v(const std::string test_type, int size_x = 10000, int size_y = 5000, int size_l = 100) {
  vector<Eigen::Matrix<double, -1, 1>> x(size_x);
  for (int i = 0; i < size_x; i++) {
    x[i].resize(size_l, 1);
    for (int j = 0; j < size_l; j++) {
      x[i](j) = j * j + i;
    }
  }
  vector<Eigen::Matrix<double, -1, 1>> y(size_y);
  for (int i = 0; i < size_y; i++) {
    y[i].resize(size_l, 1);
    for (int j = 0; j < size_l; j++) {
      y[i](j) = j * j * 0.5 + i * 2;
    }
  }
  const double sigma = 1.6;
  const double length_scale = 2.5;

  auto start = std::chrono::steady_clock::now();
  MatrixXd res_cpu = gp_exp_quad_cov(x, y, sigma, length_scale);
  int time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  print_speed(test_type, "2v", size_x, size_y, size_l, time);
}

void test_speed_2vv(const std::string test_type, int size_x = 10000, int size_y = 5000, int size_l = 100) {
  vector<Eigen::Matrix<double, -1, 1>> x(size_x);
  for (int i = 0; i < size_x; i++) {
    x[i].resize(size_l, 1);
    for (int j = 0; j < size_l; j++) {
      x[i](j) = j * j + i;
    }
  }
  vector<Eigen::Matrix<double, -1, 1>> y(size_y);
  for (int i = 0; i < size_y; i++) {
    y[i].resize(size_l, 1);
    for (int j = 0; j < size_l; j++) {
      y[i](j) = j * j * 0.5 + i * 2;
    }
  }
  double sigma = 1.6;

  vector<double> length_scale(size_l);
  for (int i = 0; i < size_l; i++) {
    length_scale[i] = i * 0.1 + 1;
  }

  auto start = std::chrono::steady_clock::now();
  MatrixXd res_cpu = gp_exp_quad_cov(x, y, sigma, length_scale);
  int time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  print_speed(test_type, "2vv", size_x, size_y, size_l, time);
}

// Run the mode with two vectors and a double for length
template <typename T>
void run_simple_test(const std::string test_type, const std::vector<T>& outer_size_iters) {
  set_tuning_opts_to_use_gpu(test_type);
  for (int i = 0; i < 10; i++) {
    for(auto&& outer_size : outer_size_iters) {
        test_speed_2(test_type, outer_size, outer_size);
        test_speed(test_type, outer_size);
    }
  }
}

// Run the mode with two vectors and a vector for length
template <typename T>
void run_l_vector_test(const std::string test_type, const std::vector<T>& outer_size_iters,
   const std::vector<T>& inner_size_iters) {
  set_tuning_opts_to_use_gpu(test_type);
  for (int i = 0; i < 10; i++) {
    for(auto&& outer_size : outer_size_iters) {
      for(auto&& inner_size : inner_size_iters) {
        test_speed_2v(test_type, outer_size, outer_size, inner_size);
        test_speed_v(test_type, outer_size, inner_size);
      }
    }
  }
}

// Run the mode with two vectors of vectors with a vector for length
template <typename T>
void run_vector_vector_test(const std::string test_type, const std::vector<T>& outer_size_iters,
   const std::vector<T>& inner_size_iters) {
  set_tuning_opts_to_use_gpu(test_type);
  for (int i = 0; i < 10; i++) {
    for(auto&& outer_size : outer_size_iters) {
      for(auto&& inner_size : inner_size_iters) {
        test_speed_2vv(test_type, outer_size, outer_size, inner_size);
        test_speed_vv(test_type, outer_size, inner_size);
      }
    }
}
}

// Makes a linearly spaced vector from a to b of size N
template <typename T>
vector<T> lin_space(T min, T max, size_t N) {
    vector<T> lin_vec;
    auto step_size = (max - min) / static_cast<T>(N - 1);
    for (int i = 0; i < N; i++) {
        lin_vec.push_back(min + i * step_size);
    }
    return lin_vec;
}

// Makes a exponentially spaced vector whose elements range a to b with spacing N
template <typename T>
vector<T> exp_space(T low, T high, size_t N) {
  auto linear_vec = lin_space(0.0, 1.0, N);
  std::vector<T> exponential_vec(N);
  auto exp_min = exp(0.0);
  auto exp_max = exp(1.0);
  // This is just rescaling the unit vector to be between high and low\
  //  and applying the exp shape.
  std::transform(linear_vec.begin(), linear_vec.end(), exponential_vec.begin(),
    [&low, &high, &exp_min, &exp_max](T x) {
      return (exp(x) - exp_min)/(exp_max - exp_min) * (high - low) + low;
    });
  return exponential_vec;
}


int main() {

  auto outer_size_iters = exp_space(0.0, 20000.0, 30);
  for (const auto& i: outer_size_iters) std::cout << i << ", ";
  std::cout << "\n";
  run_simple_test("gpu", outer_size_iters);
  run_simple_test("cpu", outer_size_iters);
  auto inner_size_iters = exp_space(0.0, 500000.0, 10);
  outer_size_iters = exp_space(0.0, 3000.0, 10);
  run_l_vector_test("gpu", outer_size_iters, inner_size_iters);
  run_l_vector_test("cpu", outer_size_iters, inner_size_iters);
  outer_size_iters = exp_space(0.0, 3000.0, 10);
  inner_size_iters = exp_space(0.0, 500000.0, 10);
  run_vector_vector_test("gpu", outer_size_iters, inner_size_iters);
  run_vector_vector_test("cpu", outer_size_iters, inner_size_iters);
}
