cd math
git checkout feature/isolated-callbacks
git pull
cd ..
rm ./benches/perf_add_test
make -j2 ./benches/perf_add_test
perf record -g --freq=2600 --call-graph dwarf -d -D 10000 -o new_add.data ./benches/perf_add_test
cd math
git checkout develop
git pull
cd ..
rm ./benches/perf_add_test
make -j2 ./benches/perf_add_test
perf record -g --freq=2600 --call-graph dwarf -d -D 10000 -o old_add.data ./benches/perf_add_test
