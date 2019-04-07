#include <benchmark/benchmark.h>
#include <CL/cl.hpp>

#include <type_traits>
#include <vector>
#include <iterator>

// vec insert
template <typename T>
inline static const std::vector<T>& vec_push_concat(const std::vector<T>& v1) {
  return v1;
}

template <typename T, typename... Args>
inline static const std::vector<T> vec_push_concat(const std::vector<T>& v1,
                                       const Args... args) {
  std::vector<T> vec = vec_push_concat(args...);
  vec.insert(vec.end(), v1.begin(), v1.end());
  return vec;
}

// Nice syntax to allow in-order expansion of parameter packs.
struct do_in_order {
    template<typename T> do_in_order(std::initializer_list<T>&&) { }
};

namespace details {
template<typename V> static void concat_helper(V& l, const V& r) {
    l.insert(l.end(), r.begin(), r.end());
}
template<class V> static void concat_helper(V& l, V&& r) {
    l.insert(l.end(), std::make_move_iterator(r.begin()),
             std::make_move_iterator(r.end()));
}
} // namespace details

template<typename... A>
static const std::vector<cl::Event> vec_move_concat(const std::vector<cl::Event>& v1, A&&... vr) {
    std::size_t s = v1.size();
    do_in_order { s += vr.size() ... };
    std::vector<cl::Event> v;
    v.reserve(s);
    do_in_order { (details::concat_helper(v, std::forward<A>(vr)), 0)... };
    v.insert(v.end(), v1.begin(), v1.end());
    return v;   // rvo blocked
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int j = 1; j <= 1024; j *= 2)
    for (int i = 0; i <= 1; ++i)
      b->Args({j, i});
}

// Actual tests
static void vec_move(benchmark::State& state) {
  std::vector<cl::Event> v1;
  std::vector<cl::Event> v2;
  std::vector<cl::Event> v3;
  std::vector<cl::Event> v4;
  std::vector<cl::Event> v5;
  std::vector<cl::Event> v6;
  cl::Event test_event;

  for (auto _ : state) {
    state.PauseTiming();
    benchmark::DoNotOptimize(test_event);
    for (int i = 0; i < state.range(0); ++i) {
      v1.push_back(test_event);
      v2.push_back(test_event);
      v3.push_back(test_event);
      v4.push_back(test_event);
      v5.push_back(test_event);
      v6.push_back(test_event);
    }
    benchmark::DoNotOptimize(v1);
    benchmark::DoNotOptimize(v2);
    benchmark::DoNotOptimize(v3);
    benchmark::DoNotOptimize(v4);
    benchmark::DoNotOptimize(v5);
    benchmark::DoNotOptimize(v6);
    state.ResumeTiming();
    if (state.range(1)) {
      std::vector<cl::Event> test = vec_move_concat(v1, v2, v3, v4, v5, v6);
    } else {
      std::vector<cl::Event> test = vec_push_concat(v1, v2, v3, v4, v5, v6);
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK(vec_move)->Apply(CustomArguments);
BENCHMARK_MAIN();
