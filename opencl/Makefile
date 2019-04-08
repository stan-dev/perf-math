MATH ?=../math/
include ../math/make/libraries
include ./local
ifeq ($(OS),Darwin)  # Mac OS X
  LDLIBS_OPENCL ?= -framework OpenCL
else
  LDLIBS_OPENCL ?= -lOpenCL
endif
OPENCL_DEVICE_ID ?= 0
OPENCL_PLATFORM_ID ?= 0

CPPFLAGS_OPENCL ?= -DSTAN_OPENCL -DOPENCL_DEVICE_ID=$(OPENCL_DEVICE_ID) -DOPENCL_PLATFORM_ID=$(OPENCL_PLATFORM_ID) -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -D__CL_ENABLE_EXCEPTIONS
CXXFLAGS_OPENCL ?= -I $(OPENCL)
CPPFLAGS+= $(CPPFLAGS_OPENCL)
CXXFLAGS+=-I..benchmark/include -std=c++1y -I..math/ -O3 -I$(BOOST) -I$(SUNDIALS)/include -I$(EIGEN) -I$(OPENCL) -Wdeprecated-declarations  -Wignored-attributes $(CXXFLAGS_OPENCL)
LDLIBS+=-lbenchmark $(LDLIBS_OPENCL) -lpthread
LDFLAGS+=-Lbenchmark/build/src
CXX ?= clang++

event_push:
	$(MAKE) ./vec_push_event && ./vec_push_event

event_cat:
	$(MAKE) ./vec_cat_event && ./vec_cat_event --benchmark_repetitions=10 --benchmark_report_aggregates_only=true

clean:
	rm stdvector_event
