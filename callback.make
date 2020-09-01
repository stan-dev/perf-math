-include Makefile

BENCHMARKS = dot_product divide gp_exp_quad_cov gp_periodic_cov dot_self columns_dot_self tcrossprod determinant inverse multiply_lower_tri_self_transpose columns_dot_product rows_dot_product cauchy chi_square double_exponential exp_mod_normal exponential frechet ordered_constrain positive_ordered_constrain simplex_constrain
BENCHMARK_BINARIES = $(patsubst %,callback/%,$(BENCHMARKS))
BENCHMARK_CSVS = $(patsubst %,callback/%.csv,$(BENCHMARKS))

.PHONY : clean_callback

$(BENCHMARK_CSVS) : callback/%.csv : callback/%
	$< --benchmark_color=false --benchmark_report_aggregates_only=false --benchmark_display_aggregates_only=false --benchmark_repetitions=30 | tail -n +5 | sed "s/ns//g" | sed "s/\// /g" | sed "s/manual_time//" | tr -s ' ' | tr ' ' ',' > $@

callback/benchmark.csv : $(BENCHMARK_CSVS)
	echo "benchmark,n,time,cpu,iterations" > $@
	cat $^ >> $@

clean_callback :
	-@rm callback/benchmark.csv
	-@rm $(BENCHMARK_CSVS)
	-@rm $(BENCHMARK_BINARIES)
