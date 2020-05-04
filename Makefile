MATH ?=math/
include math/make/compiler_flags
include math/make/dependencies                 # rules for generating dependencies
include math/make/libraries
CXXFLAGS+=-O3  -march=native -mtune=native -Ibenchmark/include -std=c++1y -Imath/ -I$(BOOST) -I"/home/steve/stan/origin/stan/src/" -I$(SUNDIALS)/include -I$(EIGEN)
LDLIBS+=-lbenchmark
LDFLAGS+=-Lbenchmark/build/src
CXX ?= clang++

update:
	git submodule update --init --recursive

benchmark/build/src/libbenchmark.a: benchmark benchmark/googletest update
	mkdir -p benchmark/build && cd benchmark/build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make

benchmark/googletest:
	cd benchmark && git clone https://github.com/google/googletest

%$(EXE) : %.o $(MPI_TARGETS) $(TBB_TARGETS)
	$(LINK.cpp) $^ $(LDLIBS) $(OUTPUT_OPTION)
