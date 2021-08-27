# Run a benchmark with a branch and then with develop to compare the two.
# ./runit.sh '{BASE}' '{BRANCH_NAME}' '{BENCHMARK_NAME}' '{BENCH_ARGUMENTS}' '{DIRECT_TO_BENCH_DATA}'
# @param BASE The base repo of the branch to bench
# @param BRANCH_NAME The Branch on the repo to test
# @param BENCHMARK_NAME name of the benchmark in './benches' to test
# @param BENCH_ARGUMENTS Additional arguments passed to the gbench
# @param DIRECT_TO_BENCH_DATA true writes the results of the test to bench_data
# as a json in the form '{BASE}_{BRANCH_NAME}_{BENCHMARK_NAME}.json'
# Run this like
# ./runit.sh 'spinkney' 'faster_inv_phi' 'inv_phi_test'
# You can also add a fourth argument as a string which will be passed
# to the benchmark such as
# ./runit.sh 'spinkney' 'faster_inv_phi' 'inv_phi_test' '--benchmark_enable_random_interleaving=true' true

cd math
git fetch $1 $2
git checkout $1/$2
cd ..
rm ./benches/$3
make -j2 ./benches/$3
if [ "$5" = true ] ; then
  ./benches/$3 $4 --benchmark_format=console --benchmark_out_format=json --benchmark_out=./bench_data/$1_$2_$3.json
else
  ./benches/$3 $4
fi

cd math
git checkout develop
git pull
cd ..
rm ./benches/$3
make -j2 ./benches/$3
if [ "$5" = true ] ; then
  ./benches/$3 $4 --benchmark_format=console --benchmark_out_format=json --benchmark_out=./bench_data/origin_develop_$3.json
else
  ./benches/$3 $4
fi
