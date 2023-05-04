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
models into MLIR files. These tools are just wrappers which call Tensorflow
Python API to do conversion. CI installs a pinned version of Tensorflow in its
docker images. To bump the Tenserflow version, you need to:

1.  Update the Tensorflow pinned version in
    [integrations/tensorflow/test/requirements.txt](integrations/tensorflow/test/requirements.txt).
2.  Follow [build_tools/docker/README.md](build_tools/docker/README.md) to
    rebuild the `frontends` docker image and its descendants.

Here is the command to rebuild and update the docker images:

```sh
python3 build_tools/docker/manage_images.py --image frontends
```

To modify the import tools themselves, you can directly change their code in
[integrations/tensorflow/python_projects](integrations/tensorflow/python_projects)
without updating the dockers.

## Benchmark Suites Design

> TODO(#12215): Explain the design and the end-to-end flow.
