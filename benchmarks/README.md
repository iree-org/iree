# IREE Benchmarks

This directory contains configuration definition for IREE's continuous
benchmarks suite. Benchmark results are posted to https://perf.iree.dev.

The https://buildkite.com/iree/iree-benchmark Buildkite pipeline runs on each
commit to the `main` branch and posts those results to the dashboard. The
pipeline also runs on pull requests with the `buildkite:benchmark` label,
posting results compared against their base commit as comments.

## Types of benchmarks

```
└── TFLite
      * Models originally in TensorFlow Lite Flatbuffer format and imported with `iree-import-tflite`
```

## Adding new benchmarks

### Machine learning model latency

1. Pick the model you want to benchmark and find its source, which could be
   a Python script, TensorFlow SavedModel from https://tfhub.dev/, TensorFlow
   Lite FlatBuffer, or some other format with a supported path into IREE. The
   model can optionally include trained weights if those are important for
   benchmarking.

2. If this is a TFLite Flatbuffer, the benchmark flow can automatically import
   it into the corresponding MLIR file. Otherwise, manually import the model
   into an MLIR file that IREE can compile using the corresponding import tool.
   For example, `iree-import-tf` for TensorFlow SavedModels. Take notes for where
   the model came from and how it was imported in case the MLIR file needs to
   be regenerated in the future.

3. Package the source model or imported MLIR file file(s) for storage (see
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

## Developer notes

These are ad-hoc notes added for developers to help triage errors.

### Repro of TFLite model errors

These steps help reproduce the failures in TFLite models.

1. Install `iree-import-tflite`.
   ```
   $ python -m pip install iree-tools-tflite -f https://github.com/google/iree/releases
   ```

2. Expose and confirm the binary `iree-import-tflite` is in your path by running
   ```
   $ iree-import-tflite --help
   ```

3. Download the TFLite flatbuffer for the failing benchmarks. The location can
   be found from [this CMakeLists.txt file](./TFLite/CMakeLists.txt).

4. Import the TFLite model into MLIR format using:
   ```
   $ iree-import-tflite <tflite-file> -o <mlir-output-file>
   ```

5. Then compile the input MLIR file with `iree-translate`. The exact flags used
   to compile and run the benchmarks can be found in
   [this CMakeLists.txt file](./TFLite/CMakeLists.txt).
