To reproduce benchmarks:

```sh
cd e2e_bench

# Fetch models
./fetch.sh

# Checkout baseline commit: 40794933d45fdbb05d631c9612dc91cc343d1efe
# Build baseline IREE tools (iree-compile, iree-opt, iree-benchmark-module) and
make sure they can be found in PATH.

# Run baseline benchmarks
cd baseline
./bench_baseline.sh
cd ..

# Checkout data-tiling commit 4cc440bc3599207828585f4b51b685a1585fe431
# Build IREE tools with data-tiling changes.

# Run batch_matmul data-tiling benchmarks
cd baseline
cd dt_and_uk
./bench_dt_and_uk.sh
cd ..
```
