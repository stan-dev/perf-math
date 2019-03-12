MATH ?=math/
include math/make/libraries

CXXFLAGS+=-Ibenchmark/include -std=c++1y -Imath/ -O3 -I$(BOOST) -I$(SUNDIALS)/include -I$(EIGEN)
LDLIBS+=-lbenchmark
LDFLAGS+=-Lbenchmark/build/src
CXX ?= clang++

update: 
	git submodule update --init --recursive

benchmark/build/src/libbenchmark.a: benchmark benchmark/googletest update
	mkdir -p benchmark/build && cd benchmark/build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make

benchmark/googletest:
	cd benchmark && git clone https://github.com/google/googletest
