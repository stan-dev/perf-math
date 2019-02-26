#!/bin/bash

cd math
git checkout develop
git pull
git clean -xffd
cd ..
cp tls_speed_eval.cpp tls_speed_eval-develop.cpp
make tls_speed_eval-develop CXX='clang++ -std=c++11'

cp tls_speed_eval.cpp tls_speed_eval-develop-tls.cpp
cd math && git clean -xffd && cd ..
make tls_speed_eval-develop-tls CXX='clang++ -std=c++11 -DSTAN_THREADS'


cd math
git checkout feature/faster-ad-tls
git pull
git clean -xffd
make stan/math/rev/core/chainablestack_inst.o
cd ..
cp tls_speed_eval.cpp tls_speed_eval-feature-tls.cpp
make tls_speed_eval-feature-tls CXX='clang++ -std=c++11 -DSTAN_THREADS -DFEATURE_TLS'

./tls_speed_eval-develop --benchmark_repetitions=4
./tls_speed_eval-develop-tls --benchmark_repetitions=4
./tls_speed_eval-feature-tls --benchmark_repetitions=4
