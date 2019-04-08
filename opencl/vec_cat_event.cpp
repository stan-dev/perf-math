#include <benchmark/benchmark.h>
#include <CL/cl.hpp>

#include <type_traits>
#include <vector>
#include <iterator>

// Nice syntax to allow in-order expansion of parameter packs.
struct do_in_order {
    template<typename T> do_in_order(std::initializer_list<T>&&) { }
};

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

// vec insert
template <typename T>
inline static const std::vector<T>& vec_push_alloc_concat(const std::vector<T>& v1) {
  return v1;
}

template <typename T, typename... Args>
inline static const std::vector<T> vec_push_alloc_concat(const std::vector<T>& v1,
                                       const Args... args) {
  std::size_t s = v1.size();
  do_in_order { s += args.size() ... };
  std::vector<T> vec;
  vec.reserve(s);
  vec = vec_push_concat(args...);
  vec.insert(vec.end(), v1.begin(), v1.end());
  return vec;
}


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
  std::vector<int> func_names = {1, 2, 3};
  for (int j = 0; j <= 24; ++j)
    for (auto&& name : func_names)
      b->Args({j, name});
}

// Actual tests
static void vec_cat(benchmark::State& state) {
  std::vector<cl::Event> v1;
  std::vector<cl::Event> v2;
  std::vector<cl::Event> v3;
  cl::Event test_event;

  for (auto _ : state) {
    state.PauseTiming();
    benchmark::DoNotOptimize(test_event);
    // A lot of the time we'll have an empty matrix with other two having
    // An extra dependency
    if (state.range(0) == 0) {
      v2.push_back(test_event);
      v3.push_back(test_event);
    }
    for (int i = 0; i < state.range(0); ++i) {
      v1.push_back(test_event);
      v2.push_back(test_event);
      v3.push_back(test_event);
    }
    benchmark::DoNotOptimize(v1.data());
    benchmark::DoNotOptimize(v2.data());
    benchmark::DoNotOptimize(v3.data());
    state.ResumeTiming();
    if (state.range(1) == 1) {
      std::vector<cl::Event> test = vec_move_concat(v1, v2, v3);
      benchmark::ClobberMemory();
    } else if (state.range(1) == 2){
      std::vector<cl::Event> test = vec_push_concat(v1, v2, v3);
      benchmark::ClobberMemory();
    } else {
      std::vector<cl::Event> test = vec_push_alloc_concat(v1, v2, v3);
      benchmark::ClobberMemory();
    }
  }
}
BENCHMARK(vec_cat)->Apply(CustomArguments);
BENCHMARK_MAIN();
