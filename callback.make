-include Makefile

BENCHMARKS = divide gp_exp_quad_cov gp_periodic_cov dot_self columns_dot_self tcrossprod determinant inverse multiply_lower_tri_self_transpose columns_dot_product rows_dot_product
BENCHMARK_CSVS = $(patsubst %,callback/%.csv,$(BENCHMARKS))

callback/%.csv : callback/%
	$< --benchmark_color=false | tail -n +5 | sed "s/ns//g" | sed "s/\// /g" | sed "s/manual_time//" | tr -s ' ' | tr ' ' ',' > $@

callback/benchmark.csv : $(BENCHMARK_CSVS)
	echo "benchmark,n,time,cpu,iterations" > $@
	cat $^ >> $@
