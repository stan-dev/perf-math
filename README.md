## first time if you don't have google benchmark library installed

```
cd benchmark
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make
```

## Making a benchmark
There is an example, this matstuff.cpp. You can make your own or reuse this
file. See https://github.com/google/benchmark for details.

## remaking a benchmark
```
$ make matstuff && ./matstuff 
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
