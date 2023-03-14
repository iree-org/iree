# IREE Benchmarks (Legacy)

**We are migrating to the new benchmark suites. Currently IREE benchmark CI
(https://perf.iree.dev) is using the new one for x86_64, CUDA, and all
compilation statistics targets. To reproduce those results, please see the
[docs for IREE new benchmark suites](/docs/developers/developing_iree/benchmark_suites.md)**.

This directory contains configuration definition for IREE's legacy benchmarks
suite.

The https://buildkite.com/iree/iree-benchmark-android Buildkite pipeline has not yet been migrated and runs
Android benchmarks defined here on each commit to the `main` branch and posts
results to the dashboard https://perf.iree.dev. The pipeline also runs on pull
requests with the `buildkite:benchmark-android` label, posting results compared
against their base commit as comments.

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

2. If this is a TFLite Flatbuffer or a TensorFlow SavedModel, the benchmark flow
   can automatically import it into the corresponding MLIR file. Make sure the
   TFLite Flatbuffer ends with `.tflite` and TensorFlow SavedModel ends with
   `tf-model`. Otherwise, manually import the model into an MLIR file that IREE
   can compile using the corresponding import tool. Take notes for where the
   model came from and how it was imported in case the MLIR file needs to be
   regenerated in the future.

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
   with the `buildkite:benchmark-*` label.

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
   $ python -m pip install iree-tools-tflite -f https://openxla.github.io/iree/pip-release-links.html
   ```

2. Expose and confirm the binary `iree-import-tflite` is in your path by running
   ```
   $ iree-import-tflite --help
   ```

3. Download the TFLite FlatBuffer for the failing benchmarks. The location can
   be found from [this CMakeLists.txt file](./TFLite/CMakeLists.txt).

4. Import the TFLite model into MLIR format using:
   ```
   $ iree-import-tflite <tflite-file> -o <mlir-output-file>
   ```

5. Then compile the input MLIR file with `iree-compile`. The exact flags used
   to compile and run the benchmarks can be found in
   [this CMakeLists.txt file](./TFLite/CMakeLists.txt).

### <a name="run-benchmark-locally"></a> Running benchmark suites locally

First you need to have [`iree-import-tflite`](https://openxla.github.io/iree/getting-started/tflite/),
[`iree-import-tf`](https://openxla.github.io/iree/getting-started/tensorflow/),
and `requests` in your python environment. Then you can build the target
`iree-benchmark-suites` to generate the required files. Note that this target
requires the `IREE_BUILD_BENCHMARKS` CMake option.

```sh
# Assume your IREE build directory is $IREE_BUILD_DIR and that cmake build was
# configured with `-DIREE_BUILD_BENCHMARKS=On`.

cmake --build $IREE_BUILD_DIR --target iree-benchmark-suites
```

Once you built the `iree-benchmark-suites` target, you will have a
`benchmark-suites` directory under `$IREE_BUILD_DIR`. You can then use
`run_benchmarks_on_android.py` or `run_benchmarks_on_linux.py` scripts under
`build_tools/benchmarks` to run the benchmark suites. For example:

```sh
build_tools/benchmarks/run_benchmarks_on_linux.py \
  --normal_benchmark_tool_dir=$IREE_BUILD_DIR/tools \
  --output results.json $IREE_BUILD_DIR
```

The benchmark results will be saved in `results.json`. You can use
`build_tools/benchmarks/diff_local_benchmarks.py` script to compare two local
benchmark results and generate the report. More details can be found
[here](/build_tools/benchmarks/README.md).

### <a name="collect-compile-stats"></a> Check compilation statistics on benchmark suites locally

Similar to [running benchmarks locally](#run-benchmark-locally), you need to
first build the target `iree-benchmark-suites`. But in addition to
`-DIREE_BUILD_BENCHMARKS=ON`, `-DIREE_ENABLE_COMPILATION_BENCHMARKS=ON` is also
required. **Note that using [Ninja](https://ninja-build.org/) to build the
project is mandatory**, becuase the tools rely on `.ninja_log` to collect the
compilation time. For example:

```sh
cmake -GNinja -S ${IREE_SOURCE_DIR} -B ${IREE_BUILD_DIR}
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DIREE_ENABLE_LLD=ON \
   -DIREE_BUILD_BENCHMARKS=ON \
   -DIREE_ENABLE_COMPILATION_BENCHMARKS=ON
```

Then run the command below to collect the statistics:

```sh
build_tools/benchmarks/collect_compilation_statistics.py \
  legacy \
  --output "compile-stats.json" \
  "${IREE_BUILD_DIR}"
```

Then `build_tools/benchmarks/diff_local_benchmarks.py` can also compare the
compilation statistics. More details can be found
[here](/build_tools/benchmarks/README.md). For example:

```sh
build_tools/benchmarks/diff_local_benchmarks.py \
  --base-compile-stats "compile-stats-before.json" \
  --target-compile-stats "compile-stats-after.json"
```

### Importing the models only

If you want to run custom benchmarks or do other work with the imported models,
without compiling the full benchmarks suites. You can run the following command
to get the imported `.mlir` files.

```sh
cmake --build $IREE_BUILD_DIR --target iree-benchmark-import-models
```
