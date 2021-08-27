## Getting started
```
git clone --recursive https://github.com/stan-dev/perf-math.git
```

### First run

(unless you have the Google benchmark library installed)
```
cd benchmark
git clone --recursive https://github.com/google/googletest
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make
```

## Writing a benchmark

See https://github.com/google/benchmark for details.

Essential structure:
```cpp
static void BM_Name(benchmark::State& state) {
  setup();
  for (auto _ : state) {
    thing_to_time();
  }
}
BENCHMARK(BM_Name);

BENCHMARK_MAIN();
```

## Making a benchmark
```
$ make ./benches/inv_phi_test && ./benches/inv_phi_test
c++ -Imath/lib/eigen_3.3.3/ -Ibenchmark/include -std=c++1y -Imath/
-Imath/lib/boost_1.66.0  -Lbenchmark/build/src  matstuff.cpp  -lbenchmark -o
matstuff
2018-08-29 04:52:47
Running ./matstuff
Run on (4 X 2200 MHz CPU s)
CPU Caches:
  L1 Data 32K (x2)
    L1 Instruction 32K (x2)
      L2 Unified 262K (x2)
        L3 Unified 4194K (x1)
        ------------------------------------------------------
        Benchmark               Time           CPU Iterations
        ------------------------------------------------------
        BM_LogOrig     2688940713 ns 2669717000 ns          1
        BM_LogProposed 2711084835 ns 2696761000 ns          1
```

If you see the warning

> ***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.

You can run `sudo cpupower frequency-set --governor performance` on linux to turn off scaling. See [here](https://stackoverflow.com/questions/38395070/google-microbenchmarking-cpu-scaling-warning)


## Helper Scripts

See the docs for runit.sh for running a branch of stan math vs. develop for a given benchmark. The graphing folder holds an Rscript for making graphs based on the output of runit.sh.
