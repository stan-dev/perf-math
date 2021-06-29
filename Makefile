MATH ?=math/
-include $(HOME)/.config/stan/make.local  # user-defined variables
-include make/local                       # user-defined variables
-include math/make/local                       # user-defined variables
## Set default compiler
include math/make/compiler_flags               # CXX, CXXFLAGS, LDFLAGS set by the end of this file
include math/make/dependencies                 # rules for generating dependencies
include math/make/libraries

CXXFLAGS+=-Ibenchmark/include -mtune=native -march=native -I./benches
LDLIBS+=-lbenchmark -ltbb
LDFLAGS+=-Lbenchmark/build/src
CXX ?= g++


update:
	git submodule update --init --recursive

benchmark/build/src/libbenchmark.a: benchmark benchmark/googletest update
	mkdir -p benchmark/build && cd benchmark/build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make

benchmark/googletest:
	cd benchmark && git clone https://github.com/google/googletest
