CXXFLAGS+=-Imath/lib/eigen_3.3.3/ -Ibenchmark/include -std=c++1y -Imath/ -Imath/lib/boost_1.66.0 -O3
LDLIBS+=-lbenchmark
LDFLAGS+=-Lbenchmark/build/src

update:
	git submodule update --init --recursive
