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
make matstuff && ./matstuff
```
