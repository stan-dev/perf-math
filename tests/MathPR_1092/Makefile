MATH ?=../math/
include ../math/make/libraries

CXXFLAGS+=-I../benchmark/include -std=c++1y -I../math/ -O3 -mtune=native -march=native -I$(BOOST) -I$(SUNDIALS)/include -I$(EIGEN)
LDLIBS+=-lbenchmark -pthread
LDFLAGS+=-L../benchmark/build/src
CXX ?= clang++

update: 
	git submodule update --init --recursive

benchmark/build/src/libbenchmark.a: benchmark benchmark/googletest update
	mkdir -p benchmark/build && cd benchmark/build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make

benchmark/googletest:
	cd benchmark && git clone https://github.com/google/googletest
