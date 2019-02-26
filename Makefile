CXXFLAGS+=-Imath/lib/eigen_3.3.3/ -Ibenchmark/include -std=c++1y -Imath/ -Imath/lib/boost_1.66.0 -O3 -Imath/lib/sundials_3.1.0/include
LDLIBS+=-lbenchmark
LDFLAGS+=-Lbenchmark/build/src
CXX=clang++ -std=c++11 
#CXX=g++-8 -static-libstdc++ -std=c++11 

update:
	git submodule update --init --recursive
