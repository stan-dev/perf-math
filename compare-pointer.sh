#!/bin/bash

# develop without STAN_THREADS
cd math
git checkout develop
git pull
git reset --hard HEAD
git clean -xffd
cd ..
cp tls_speed_eval.cpp tls_speed_eval-develop.cpp
make tls_speed_eval-develop CXX='clang++ -std=c++11'

# Without STAN_THREADS on faster-ad-tls
cd math
git checkout feature/faster-ad-tls-v3
git pull
git reset --hard HEAD
git clean -xffd
make stan/math/rev/core/chainablestack_inst.o
cd ..
cp tls_speed_eval.cpp tls_speed_eval-feature-tls-no-threads.cpp
make tls_speed_eval-feature-tls-no-threads

./tls_speed_eval-develop --benchmark_repetitions=30
./tls_speed_eval-feature-tls-no-threads --benchmark_repetitions=30
