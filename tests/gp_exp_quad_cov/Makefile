MATH ?=../../math/
include ../../math/make/libraries

CXXFLAGS+=-I../../benchmark/include -std=c++1y -I$(MATH) -O3
CXXFLAGS+=-mtune=native -march=native -I$(BOOST) -I$(SUNDIALS)/include -I$(EIGEN) -I$(OPENCL)
CXXFLAGS+=-DBOOST_RESULT_OF_USE_TR1 -DBOOST_NO_DECLTYPE -DBOOST_DISABLE_ASSERTS -DBOOST_PHOENIX_NO_VARIADIC_EXPRESSION
CXXFLAGS+=-DSTAN_OPENCL -DOPENCL_DEVICE_ID=1 -DOPENCL_PLATFORM_ID=2
CXXFLAGS+=-DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_TARGET_OPENCL_VERSION=120
CXXFLAGS+=-D__CL_ENABLE_EXCEPTIONS -Wno-ignored-attributes  -DGTEST_USE_OWN_TR1_TUPLE
LDLIBS+=-lbenchmark -pthread -lOpenCL
LDFLAGS+=-L../../benchmark/build/src
CXX ?= g++

init:
	git remote add bstatcomp https://github.com/bstatcomp/math
	git fetch bstatcomp
	git checkout --track bstatcomp/gpu_cov_exp_quad2

update:
	git submodule update --init --recursive

benchmark/build/src/libbenchmark.a: benchmark benchmark/googletest update
	mkdir -p benchmark/build && cd benchmark/build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make

benchmark/googletest:
	cd benchmark && git clone https://github.com/google/googletest
