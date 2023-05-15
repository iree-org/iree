# IREE Benchmark Suites

**We are in the progress of replacing the legacy benchmark suites. Currently the
new benchmark suites only support `x86_64`, `CUDA`, and `compilation statistics`
benchmarks. For working with the legacy benchmark suites, see
[IREE Benchmarks (Legacy)](/benchmarks/README.md)**.

IREE Benchmarks Suites is a collection of benchmarks for IREE developers to
track performance improvements/regressions during development.

The benchmark suites are run for each commit on the main branch and the results
are uploaded to https://perf.iree.dev for regression analysis (for the current
supported targets). On pull requests, users can add labels `benchmarks:*` to
trigger the benchmark runs. The results will be compared with
https://perf.iree.dev and post in the comments.

Information about the definitions of the benchmark suites can be found in the
[IREE Benchmark Suites Configurations](/build_tools/python/benchmark_suites/iree/README.md).

## Running Benchmark Suites Locally

### Prerequisites

Install `iree-import-tf` and `iree-import-tflite` in your Python environment
(see
[Tensorflow Integration](https://openxla.github.io/iree/getting-started/tensorflow/)
and
[TFLite Integration](https://openxla.github.io/iree/getting-started/tflite/)).

### Build Benchmark Suites

Configure IREE with `-DIREE_BUILD_E2E_TEST_ARTIFACTS=ON`:

```sh
cmake -GNinja -B "${IREE_BUILD_DIR?}" -S "${IREE_REPO?}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DIREE_ENABLE_LLD=ON \
  -DIREE_BUILD_E2E_TEST_ARTIFACTS=ON
```

Build the benchmark suites and tools:

```sh
cmake --build "${IREE_BUILD_DIR?}" --target \
  iree-e2e-test-artifacts \
  iree-benchmark-module
export E2E_TEST_ARTIFACTS_DIR="${IREE_BUILD_DIR?}/e2e_test_artifacts"
```

### Run Benchmarks

Export the execution benchmark config:

```sh
build_tools/benchmarks/export_benchmark_config.py execution > "${E2E_TEST_ARTIFACTS_DIR?}/exec_config.json"
```

Run benchmarks (currently only support running on a Linux host):

```sh
build_tools/benchmarks/run_benchmarks_on_linux.py \
  --normal_benchmark_tool_dir="${IREE_BUILD_DIR?}/tools" \
  --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR?}" \
  --execution_benchmark_config="${E2E_TEST_ARTIFACTS_DIR?}/exec_config.json" \
  --target_device_name="<target_device_name, e.g. c2-standard-16>" \
  --output="${E2E_TEST_ARTIFACTS_DIR?}/benchmark_results.json" \
  --verbose \
  --cpu_uarch="<host CPU uarch, e.g. CascadeLake>"
# Traces can be collected by adding:
# --traced_benchmark_tool_dir="${IREE_TRACED_BUILD_DIR?}/tools" \
# --trace_capture_tool=/path/to/iree-tracy-capture \
# --capture_tarball=captured_tracy_files.tar.gz
```

Note that:

-   `<target_device_name>` selects a benchmark group targets a specific device:
    -   Common options:
        -   `c2-standard-16` for x86_64 CPU benchmarks.
        -   `a2-highgpu-1g` for NVIDIA GPU benchmarks.
    -   All device names are defined under
        [build_tools/python/e2e_test_framework/device_specs](/build_tools/python/e2e_test_framework/device_specs).
-   To run x86_64 benchmarks, right now `--cpu_uarch` needs to be provided and
    only `CascadeLake` is available currently.
-   To build traced benchmark tools, see
    [Profiling with Tracy](/docs/developers/developing_iree/profiling_with_tracy.md).

Filters can be used to select the benchmarks:

```sh
build_tools/benchmarks/run_benchmarks_on_linux.py \
  --normal_benchmark_tool_dir="${IREE_BUILD_DIR?}/tools" \
  --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR?}" \
  --execution_benchmark_config="${E2E_TEST_ARTIFACTS_DIR?}/exec_config.json" \
  --target_device_name="c2-standard-16" \
  --output="${E2E_TEST_ARTIFACTS_DIR?}/benchmark_results.json" \
  --verbose \
  --cpu_uarch="CascadeLake" \
  --model_name_regex="MobileBert*" \
  --driver_filter_regex='local-task' \
  --mode_regex="4-thread"
```

### Generate Compilation Statistics (Compilation Benchmarks)

Export the compilation benchmark config:

```sh
build_tools/benchmarks/export_benchmark_config.py compilation > "${E2E_TEST_ARTIFACTS_DIR?}/comp_config.json"
```

Generate the compilation statistics:

```sh
build_tools/benchmarks/collect_compilation_statistics.py \
  alpha \
  --compilation_benchmark_config=comp_config.json \
  --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR?}" \
  --build_log="${IREE_BUILD_DIR?}/.ninja_log" \
  --output="${E2E_TEST_ARTIFACTS_DIR?}/compile_stats_results.json"
```

Note that you need to use [Ninja](https://ninja-build.org/) to build the
benchmark suites as the tool collects information from its build log.

### Show Execution / Compilation Benchmark Results

See
[Generating Benchmark Report](/build_tools/benchmarks/README.md#generating-benchmark-report).

### Find Compile and Run Commands to Reproduce Benchmarks

Each benchmark has its benchmark ID in the benchmark suites, you will see a
benchmark ID at:

-   In the serie's URL of https://perf.iree.dev
    -   Execution benchmark: `https://perf.iree.dev/serie?IREE?<benchmark_id>`
    -   Compilation benchmark:
        `https://perf.iree.dev/serie?IREE?<benchmark_id>-<metric_id>`
-   In `benchmark_results.json` and `compile_stats_results.json`
    -   Execution benchmark result has a field `run_config_id`
    -   Compilation benchmark result has a field `gen_config_id`
-   In PR benchmark summary or the markdown generated by
    `diff_local_benchmarks.py`, each benchmark has the link to its
    https://perf.iree.dev URL, which includes the benchmark ID.

If you don't have artifacts locally, see
[Fetching Benchmark Artifacts from CI](#fetching-benchmark-artifacts-from-ci) to
find the GCS directory of the CI artifacts. Then fetch the needed files:

```sh
# Get ${E2E_TEST_ARTIFACTS_DIR_URL} from "Fetching Benchmark Artifacts from CI".
export E2E_TEST_ARTIFACTS_DIR="e2e_test_artifacts"

# Download all artifacts
mkdir "${E2E_TEST_ARTIFACTS_DIR?}"
gcloud storage cp -r "${E2E_TEST_ARTIFACTS_DIR_URL?}" "${E2E_TEST_ARTIFACTS_DIR?}"
```

Run the helper tool to dump benchmark commands from benchmark configs:

```sh
build_tools/benchmarks/benchmark_helper.py dump-cmds \
  --execution_benchmark_config="${E2E_TEST_ARTIFACTS_DIR?}/execution-benchmark-config.json" \
  --compilation_benchmark_config="${E2E_TEST_ARTIFACTS_DIR?}/compilation-benchmark-config.json" \
  --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR?}" \
  --benchmark_id="<benchmark_id>"
```

### Get Full List of Benchmarks

The commands below output the full list of execution and compilation benchmarks,
including the benchmark names and their flags:

```sh
build_tools/benchmarks/export_benchmark_config.py execution > "${E2E_TEST_ARTIFACTS_DIR?}/exec_config.json"
build_tools/benchmarks/export_benchmark_config.py compilation > "${E2E_TEST_ARTIFACTS_DIR?}/comp_config.json"
build_tools/benchmarks/benchmark_helper.py dump-cmds \
  --execution_benchmark_config="${E2E_TEST_ARTIFACTS_DIR?}/exec_config.json" \
  --compilation_benchmark_config="${E2E_TEST_ARTIFACTS_DIR?}/comp_config.json"
```

## Fetching Benchmark Artifacts from CI

#### 1. Find the corresponding CI workflow run

On the commit of the benchmark run, you can find the list of the workflow jobs
by clicking the green check mark. Click any job starts with `CI /`:

![image](https://user-images.githubusercontent.com/2104162/234647960-3df9d0f0-a34a-47ad-bda8-095ae44de865.png)

#### 2. Get URLs of GCS artifacts

On the CI page, click `Summary` on the top-left to open the summary page. Scroll
down and the links to artifacts are listed in a section titled "Artifact Links".
Paste the content in your shell to define all needed variables for the following
steps:

![image](https://user-images.githubusercontent.com/2104162/234716421-3a69b6ad-211d-4e39-8f9e-a4f22f91739d.png)

#### 3. Fetch the benchmark artifacts

To fetch files from the GCS URL, the gcloud CLI tool
(https://cloud.google.com/sdk/docs/install) can list the directory contents and
download files (see https://cloud.google.com/sdk/gcloud/reference/storage for
more usages). If you want to use CI artifacts to reproduce benchmarks locally,
see
[Find Compile and Run Commands to Reproduce Benchmarks](#find-compile-and-run-commands-to-reproduce-benchmarks).

Assume you get the GCS URL variables from
[Get URLs of GCS artifacts](#2-get-urls-of-gcs-artifacts).

Download artifacts:

```sh
# The GCS directory has the same structure as your local ${IREE_BUILD_DIR?}/e2e_test_artifacts.
gcloud storage ls "${E2E_TEST_ARTIFACTS_DIR_URL?}"

# Download all source and imported MLIR files:
gcloud storage cp "${E2E_TEST_ARTIFACTS_DIR_URL?}/*.mlir" "<target_dir>"
```

Execution and compilation benchmark configs can be downloaded at:

```sh
# Execution benchmark config:
gcloud storage cp \
  "${E2E_TEST_ARTIFACTS_DIR_URL?}/execution-benchmark-config.json" \
  "${E2E_TEST_ARTIFACTS_DIR?}/exec_config.json"

# Compilation benchmark config:
gcloud storage cp \
  "${E2E_TEST_ARTIFACTS_DIR_URL?}/compilation-benchmark-config.json" \
  "${E2E_TEST_ARTIFACTS_DIR?}/comp_config.json"
```

Benchmark raw results and traces can be downloaded at:

```sh
# Execution benchmark raw results
gcloud storage cp "${EXECUTION_BENCHMARK_RESULTS_DIR_URL?}/benchmark-results-*.json" .

# Execution benchmark traces
gcloud storage cp "${EXECUTION_BENCHMARK_RESULTS_DIR_URL?}/benchmark-traces-*.tar.gz" .

# Compilation benchmark results
gcloud storage cp "${COMPILATION_BENCHMARK_RESULTS_URL?}" .
```
