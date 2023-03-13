# IREE Benchmark Suites Configurations

This directory contains the Python scripts that define the benchmark
configrations. To run the benchmark suites, see
[IREE Benchmark Suites](/docs/developers/developing_iree/benchmark_suites.md).

## Updating Benchmarks

1.  Modify the Python scripts of benchmark generators under
    [build_tools/python/benchmark_suites/iree](/build_tools/python/benchmark_suites/iree).
2.  Follow
    [tests/e2e/test_artifacts](https://github.com/openxla/iree/tree/main/tests/e2e/test_artifacts)
    to regenerate the cmake files that will build the benchmarks.

To add a new source model, see
[Adding a new model](/build_tools/python/e2e_test_framework/models/README.md#adding-a-new-model)

## Benchmark Suites Design

> TODO(#12215): Explain the design and the end-to-end flow.
