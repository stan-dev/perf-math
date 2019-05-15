#include <iostream>


#include <stan/math.hpp>
#include <stan/math/opencl/copy.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/rev/mat/fun/gp_exp_quad_cov.hpp>
#include <chrono>
#include <vector>
#include <cstdio>


using namespace Eigen;
using namespace std;
using namespace stan::math;

auto test_speed_1(const int size = 3000) {
  vector<double> x(size);
  for (int i = 0; i < size; i++) {
    x[i] = i * i;
  }

  const double sigma = 1.6;
  const double length_scale = 2.7;

  auto start = std::chrono::steady_clock::now();
  MatrixXd res_gpu = gp_exp_quad_cov(x, sigma, length_scale);
  auto time_gpu = std::chrono::duration_cast<std::chrono::microseconds>
   (std::chrono::steady_clock::now() - start).count();

  return time_gpu;
}

auto test_speed_2(const int size_x = 10000, const int size_y = 5000) {
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
  MatrixXd res_gpu = gp_exp_quad_cov(x, y, sigma, length_scale);
  auto time_gpu = std::chrono::duration_cast<std::chrono::microseconds>
   (std::chrono::steady_clock::now() - start).count();

  return time_gpu;
}

auto test_speed_v(const int size = 10000, const int size_l = 100) {
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
  MatrixXd res_gpu = gp_exp_quad_cov(x, sigma, length_scale);
  auto time_gpu = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();

  return time_gpu;
}

float test_speed_vv(const int size = 10000, const int size_l = 100) {
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
  MatrixXd res_gpu = gp_exp_quad_cov(x, sigma, length_scale);
  auto time_gpu = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();

  return time_gpu;

}

auto test_speed_2v(const int size_x = 10000, const int size_y = 5000, const int size_l = 100) {
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
  MatrixXd res_gpu = gp_exp_quad_cov(x, y, sigma, length_scale);
  auto time_gpu = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();

  return time_gpu;
}

auto test_speed_2vv(const int size_x = 10000, const int size_y = 5000, const int size_l = 100) {
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

  vector<double> length_scale(size_l);
  for (int i = 0; i < size_l; i++) {
    length_scale[i] = i * 0.1 + 1;
  }
  auto start = std::chrono::steady_clock::now();
  MatrixXd res_gpu = gp_exp_quad_cov(x, y, sigma, length_scale);
  auto time_gpu = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();

  return time_gpu;
}

int main() {
  printf("instance, speedup, size1, size2, s\n");
  for (int size_l = 5; size_l < 10000; size_l *= 1.3) {
    auto size_y = 1000;
    for (int size_x = 1000; size_x < 10000; size_x *= 1.2) {
      auto result = test_speed_1(size_x);
      printf("2vv, %ld, %d, %d, %d\n", result, size_x, size_y, size_l);
    }
  }

}
