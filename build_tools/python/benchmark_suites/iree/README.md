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

## Updating TF/TFLite Importer in CI

For TF/TFLite source models, benchmark CI uses `iree-import-tf/tflite` to import
models into MLIR files. CI installs pinned binary releases of these tools. To
bump the tool version, you can change:

-   `iree-import-tflite`:
    -   Update the `iree-tools-tflite` version in
        [build_tools/cmake/setup_tf_python.sh](build_tools/cmake/setup_tf_python.sh).
-   `iree-import-tf`: It is a wrapper of Tensorflow Python API.
    -   Update the Tensorflow version pinned in
        [integrations/tensorflow/test/requirements.txt](integrations/tensorflow/test/requirements.txt).
    -   Follow [build_tools/docker/README.md](build_tools/docker/README.md) to
        rebuild the docker images.

## Benchmark Suites Design

> TODO(#12215): Explain the design and the end-to-end flow.
