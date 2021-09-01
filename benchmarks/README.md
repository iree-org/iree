# IREE Benchmarks

This directory contains configuration definition for IREE's continuous
benchmarks suite. Benchmark results are posted to https://perf.iree.dev.

The https://buildkite.com/iree/iree-benchmark Buildkite pipeline runs on each
commit to the `main` branch and posts those results to the dashboard. The
pipeline also runs on pull requests with the `buildkite:benchmark` label,
posting results compared against their base commit as comments.

## Types of benchmarks

```
├── TensorFlow
│     * models authored in TensorFlow and imported with `iree-import-tf`
└── TFLite
      * models converted to TensorFlow Lite and imported with `iree-import-tflite`
```

## Adding new benchmarks

### Machine learning model latency

1. Pick the model you want to benchmark and find its source, which could be
   a Python script, TensorFlow SavedModel from https://tfhub.dev/, TensorFlow
   Lite FlatBuffer, or some other format with a supported path into IREE. The
   model can optionally include trained weights if those are important for
   benchmarking.

2. Import the model into an MLIR file that IREE can compile using the core
   `iree-translate` tool. For TensorFlow models use `iree-import-tf`, for
   TensorFlow Lite models use `iree-import-tflite`, etc. Take notes for where
   the model came from and how it was imported in case the MLIR file needs to
   be regenerated in the future.

   We may further automate this over time, such as by importing from Python
   sources as part of the benchmarks pipeline directly (see
   https://github.com/google/iree/issues/6942). For now, here are some
   references:

   * https://gist.github.com/antiagainst/35b0989bd0188dd9df4630bb0cf778f2
   * https://colab.research.google.com/gist/ScottTodd/10838c0ccc87fa6d1b1c72e0fabea064/iree-keyword_spotting_streaming-benchmarks.ipynb

3. Package the imported .mlir model file(s) for storage (see
   [iree_mlir_benchmark_suite.cmake](../build_tools/cmake/iree_mlir_benchmark_suite.cmake)
   and [download_file.py](../scripts/download_file.py)), then upload them to the
   `iree-model-artifacts` Google Cloud Storage bucket with the help of a team
   member. Files currently hosted in that bucket can be viewed at
   https://storage.googleapis.com/iree-model-artifacts/index.html.

4. Edit the appropriate `CMakeLists.txt` file under this directory to include
   your desired benchmark configuration with the `iree_mlir_benchmark_suite`
   function. You can test your change by running the
   https://buildkite.com/iree/iree-benchmark pipeline on a GitHub pull request
   with the `buildkite:benchmark` label.

5. Once your changes are merged to the `main` branch, results will start to
   appear on the benchmarks dashboard at https://perf.iree.dev.

### Other project metrics

TODO(#6161): Collect metrics for miscellaneous IREE system states
