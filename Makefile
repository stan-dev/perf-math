MATH ?=math/
include math/make/libraries
include make/local
include math/make/compiler_flags               # CXX, CXXFLAGS, LDFLAGS set by the end of this file

LDLIBS+=-lbenchmark
LDFLAGS+=-Lbenchmark/build/src

update:
	git submodule update --init --recursive

benchmark/build/src/libbenchmark.a: benchmark benchmark/googletest update
	mkdir -p benchmark/build && cd benchmark/build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make

benchmark/googletest:
	cd benchmark && git clone https://github.com/google/googletest


opencl_setup_math_gpu_cov_exp_quad2:
	git -C ./math remote add bstatcomp git@github.com:bstatcomp/math.git
	git -C ./math fetch bstatcomp
	git -C ./math checkout --track bstatcomp/gpu_cov_exp_quad2
