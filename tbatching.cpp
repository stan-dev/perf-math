#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <future>
#include <thread>
#include <iostream>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void clobber() {
  asm volatile("" : : : "memory");
}

using matrix_d = Eigen::MatrixXd;

static void BM_ChunkedThreads(benchmark::State& state) {
  using Eigen::MatrixXd;
  auto m_d = MatrixXd::Random(50, 50).eval();
  auto output = matrix_d(0, 0);

  auto execute_chunk = [&](int size) -> std::vector<matrix_d> {
    std::vector<matrix_d> chunk_f_out;
    chunk_f_out.reserve(size);
    for (int i = 0; i < size; i++) {
      chunk_f_out.push_back(m_d.transpose() * m_d);
    }
    return chunk_f_out;
  };
  const int num_chunks = state.range(0);
  const int chunk_size = state.range(1);

  for (auto _ : state) {
    escape(m_d.data());

    // Actual task
    std::vector<std::future<std::vector<matrix_d>>> futures;
    futures.reserve(num_chunks);
    for (int i = 0; i < num_chunks; i++) {
      futures.emplace_back(std::async(std::launch::async, execute_chunk, chunk_size));
    }

    int offset = 0;
    bool resized = false;
    for (auto& f : futures) {
      const std::vector<matrix_d>& chunk_result = f.get();
      if (!resized) {
        output.resize(chunk_result[0].rows(),
                      num_chunks * chunk_result[0].cols());
        resized = true;
      }
      for (const auto& job_result : chunk_result) {
        const int num_job_outputs = job_result.cols();
        if (output.cols() < offset + num_job_outputs) {
          output.conservativeResize(Eigen::NoChange,
                                    2 * (offset + num_job_outputs));
        }
        output.block(0, offset, output.rows(), num_job_outputs) = job_result;
        offset += num_job_outputs;
      }
    }

    clobber();
  }
}
BENCHMARK(BM_ChunkedThreads)->Args({10, 12})->Args({1, 1})->Args({100, 100});

static void BM_BareThreads(benchmark::State& state) {
  using Eigen::MatrixXd;
  auto m_d = MatrixXd::Random(50, 50).eval();

  auto execute_chunk = [&]() -> matrix_d {
    return m_d.transpose() * m_d;
  };

  const int num_chunks = state.range(0);
  const int chunk_size = state.range(1);
  const int num_tasks = num_chunks * chunk_size;
  matrix_d output(m_d.rows(), m_d.cols() * num_tasks);

  for (auto _ : state) {
    escape(m_d.data());

    // Actual task
    std::vector<std::future<matrix_d>> futures;
    futures.reserve(num_tasks);
    for (int i = 0; i < num_tasks; i++) {
      futures.emplace_back(std::async(std::launch::async, execute_chunk));
    }

    int offset = 0;
    for (auto& f : futures) {
      const matrix_d& job_result = f.get();
      const int num_job_outputs = job_result.cols();
      output.block(0, offset, output.rows(), num_job_outputs)
          = job_result;
      offset += num_job_outputs;
    }

    clobber();
  }
}
BENCHMARK(BM_BareThreads)->Args({10, 12})->Args({1, 1})->Args({100, 100});

BENCHMARK_MAIN();
