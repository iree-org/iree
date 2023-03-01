# IREE Benchmark Suites Tool (Legacy)

**For working with the new benchmark suite, see the [section below](#iree-new-benchmark-suite)**

This directory contains the tools to run IREE benchmark suites and generate
reports. More information about benchmark suites can be found [here](/benchmarks/README.md).

## Benchmark Tools

Currently we have `run_benchmarks_on_android.py` and
`run_benchmarks_on_linux.py` scripts to run benchmark suites on Android devices
(with `adb`) and Linux machines.

The available arguments can be shown with `--help`. Some common usages are
listed below. Here we assume:

```sh
IREE_BUILD_DIR="/path/to/IREE build root dir". It should contain the "benchmark_suites" directory built with the target "iree-benchmark-suites".

IREE_NORMAL_TOOL_DIR="/path/to/IREE tool dir". It is usually "$IREE_BUILD_DIR/tools".

IREE_TRACED_TOOL_DIR="/path/to/IREE tool dir built with IREE_ENABLE_RUNTIME_TRACING=ON".
```

See details about `IREE_ENABLE_RUNTIME_TRACING` [here](/docs/developers/developing_iree/profiling_with_tracy.md).

**Run all benchmarks**
```sh
./run_benchmarks_on_linux.py \
  --normal_benchmark_tool_dir=$IREE_NORMAL_TOOL_DIR \
  --output=results.json $IREE_BUILD_DIR
```

**Run all benchmarks and perform the Tracy captures**
```sh
./run_benchmarks_on_linux.py \
  --normal_benchmark_tool_dir=$IREE_NORMAL_TOOL_DIR \
  --traced_benchmark_tool_dir=$IREE_TRACED_TOOL_DIR \
  --trace_capture_tool=/path/to/iree-tracy-capture \
  --capture_tarball=captured_tracy_files.tar.gz
  --output=results.json $IREE_BUILD_DIR
```

**Run selected benchmarks with the filters**
```sh
./run_benchmarks_on_linux.py \
  --normal_benchmark_tool_dir=$IREE_NORMAL_TOOL_DIR \
  --model_name_regex="MobileBertSquad" \
  --driver_filter_regex="local-task" \
  --mode_regex="4-threads" \
  --output=results.json $IREE_BUILD_DIR
```

**Collect compilation statistics**

See [here](/benchmarks/README.md#collect-compile-stats) for additional build
steps to enable compilation statistics collection.
```sh
./collect_compilation_statistics.py \
  legacy \
  --output "compile-stats.json" \
  "${IREE_BUILD_DIR}"
```

## Generating Benchmark Report

The tools here are mainly designed for benchmark automation pipelines.
The `post_benchmarks_as_pr_comment.py` and `upload_benchmarks_to_dashboard.py`
scripts are used to upload and post reports to pull requests or the
[dashboard](https://perf.iree.dev/).

If you want to generate a comparison report locally, you can use
`diff_local_benchmarks.py` script to compare two result json files and generate
the report. For example:

```sh
./diff_local_benchmarks.py --base before.json --target after.json > report.md
```

An example that compares compilation statistics:

```sh
./diff_local_benchmarks.py \
  --base-compile-stats "compile-stats-before.json" \
  --target-compile-stats "compile-stats-after.json" \
  > report.md
```

# IREE New Benchmark Suite
We are in progrss to replace the [legacy cmake benchmark suite](/benchmarks)
with a new one written in python. Currently it only supports
`x86_64`, `CUDA`, and `compilation statistics` benchmarks.

**Our benchmark CI (https://perf.iree.dev) is using the new benchmark suite to
benchmark those targets.**

## Run benchmark suite locally 

### Prerequisites
- Have `iree-import-tf` and `iree-import-tflite` installed in
your Python environment. You can check the page
[Tensorflow Integration](https://openxla.github.io/iree/getting-started/tensorflow/)
and [TFLite Integration](https://openxla.github.io/iree/getting-started/tflite/)
to learn how to install those tools.
- Have the `jq` installed for manipulating JSON in command lines (recommended).

### Build the benchmark suite
Configure IREE with `-DIREE_BUILD_E2E_TEST_ARTIFACTS=ON` and CUDA
backend enabled. For example:
```sh
cmake -GNinja -B "${IREE_BUILD_DIR}" -S "${IREE_REPO}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DIREE_ENABLE_LLD=ON \
  -DIREE_TARGET_BACKEND_CUDA=ON \
  -DIREE_HAL_DRIVER_CUDA=ON \
  -DIREE_BUILD_E2E_TEST_ARTIFACTS=ON
```
Then build the benchmark suites and tools by:
```sh
cmake --build "${IREE_BUILD_DIR}" --target \
  iree-e2e-test-artifacts \
  iree-benchmark-module
```

### Run benchmarks
The script runs the benchmarks defined by a JSON config. To generate an execution
benchmark config:
```sh
build_tools/benchmarks/export_benchmark_config.py execution > exec_config.json
```
Then run benchmarks locally:
```sh
build_tools/benchmarks/run_benchmarks_on_linux.py \
  --normal_benchmark_tool_dir="$IREE_BUILD_DIR/tools" \
  --e2e_test_artifacts_dir="$IREE_BUILD_DIR/e2e_test_artifacts" \
  --execution_benchmark_config=exec_config.json \
  --target_device_name="<target_device_name>" \
  --output=benchmark_results.json \
  --cpu_uarch="<CPU uarch>" \
  --verbose
# Traces can be collected by adding:
# --traced_benchmark_tool_dir="$IREE_TRACED_BUILD_DIR/tools" \
# --trace_capture_tool=/path/to/iree-tracy-capture \
# --capture_tarball=captured_tracy_files.tar.gz
```
Note that:
- `<target_device_name>` selects a group of benchmarks target at a benchmark
device. Common options are `c2-standard-16` for CPU benchmarks, and
`a2-highgpu-1g` for GPU benchmarks. You can find all device names defined
[here](/build_tools/python/e2e_test_framework/device_specs).
- To run x86_64 benchmarks, right now `--cpu_uarch` needs to be provided and
  only only `CascadeLake` is available currently.
- To build traced benchmark tools, see the details
[here](/docs/developers/developing_iree/profiling_with_tracy.md).

Filters can be used to select the benchmarks:
```sh
build_tools/benchmarks/run_benchmarks_on_linux.py \
  --normal_benchmark_tool_dir="$IREE_BUILD_DIR/tools" \
  --e2e_test_artifacts_dir="$IREE_BUILD_DIR/e2e_test_artifacts" \
  --execution_benchmark_config=exec_config.json \
  --target_device_name="c2-standard-16" \
  --output=benchmark_results.json \
  --verbose \
  --cpu_uarch="CascadeLake" \
  --model_name_regex="MobileBert*" \
  --driver_filter_regex='local-task' \
  --mode_regex="4-thread"
```

### Get compilation statistics (compilation benchmarks)
First export the compilation benchmark config:
```sh
build_tools/benchmarks/export_benchmark_config.py compilation > comp_config.json
```
Generate compilation statistics:
```sh
build_tools/benchmarks/collect_compilation_statistics.py \
  alpha \
  --compilation_benchmark_config=comp_config.json \
  --e2e_test_artifacts_dir="${IREE_BUILD_DIR}/e2e_test_artifacts" \
  --build_log="${IREE_BUILD_DIR}/.ninja_log" \
  --output=compile_stats_results.json
```

### Print execution / compilation benchmark results
See the section [Generating Benchmark Report](#generating-benchmark-report)

### Find compile and run flags of a benchmark
Benchmarks are represented with their artificial ids (sha256 hex) in the
benchmark suite, so you need to get the artificial id first. Here are the places
you can find their artificial ids:
- On https://perf.iree.dev, each serie's URL is in the format:
  - Execution benchmark: `https://perf.iree.dev/serie?IREE?<artificial_id>`
  - Compilation benchmark: `https://perf.iree.dev/serie?IREE?<artificial_id>-<metric_id>`
- In `benchmark_results.json` and `compile_stats_results.json`
  - Each execution benchmark has the field `run_config_id`
  - Each compilation benchmark has the field `gen_config_id`
- In the markdown generated by `diff_local_benchmarks.py`, each benchmark shows
  its https://perf.iree.dev URL, which contains its artificial id.

We refer execution benchmark artificial id as `${EXEC_ARTIFICIAL_ID}` and
compilation benchmark artificial id as `${COMP_ARTIFICIAL_ID}` below.

> TODO(#12215): Add helper tool to search and access these information.

With the artificial id of a benchmark, you can look into the exported benchmark
configs for their flags and metadata. There are three kinds of related objects:
```
"iree_e2e_model_run_configs:<exec_artificial_id>": {
  "module_generation_config": <id to its module generation config>
  "run_flags": [...]
}

"iree_module_generation_configs:<comp_artificial_id>": {
  "imported_model": <id to its imported model>
  "compile_flags": [...]
}

"iree_imported_models:<imported_model_id>": {
  "model": <id to its model>
}
```
You can check
[build_tools/python/e2e_test_framework/definitions/iree_definitions.py](/build_tools/python/e2e_test_framework/definitions/iree_definitions.py)
for their definitions.

Here are some useful commands to extract information:

<a name="get-run-flags"></a>**Get run flags of an execution benchmark**
```sh
cat exec_config.json | \
  jq --arg exec_id "${EXEC_ARTIFICIAL_ID}" \
  'map_values(.[].keyed_obj_map? | ."iree_e2e_model_run_configs:\($exec_id)")'

This shows an E2E model run object like:
{
  "c2-standard-16": {
    "composite_id": "e496c2ea8de7fdffdb7597da63eed12cc5fb0595605d70e1f9d67a33857a299a",
    "module_generation_config": "7a0add4835462bc66025022cdb6e87569da79cf103825a809863b8bd57a49055",
    "module_execution_config": "13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03",
    "target_device_spec": "9a4804f1-b1b9-46cd-b251-7f16a655f782",
    "input_data": "8d4a034e-944d-4725-8402-d6f6e61be93c",
    "run_flags": [
      "--function=forward",
      "--input=1x224x224x3xf32=0",
      "--device_allocator=caching",
      "--device=local-sync"
    ]
  },
  "a2-highgpu-1g": null,
}

You can find the VMFB module at:
echo ${IREE_BUILD_DIR}/e2e_test_artifacts/iree_*_<module_generation_config>/module.vmfb
```
**Get compile flags, module, and input MLIR of an execution / compilation benchmark**

First you need to get the compilation artificial id. It can be either

- The field `module_generation_config` of an E2E model run object from an
  execution benchmark (see [Get run flags of an execution benchmark](#get-run-flags)).
- Compilation benchmark artificial id from https://perf.iree.dev or compilation
  benchmark results.

> Note that the configs for execution and compilation benchmarks might contain
> different set of artificial ids. Make sure you query the right config file.

```sh
# For execution benchmarks
cat exec_config.json | \
  jq --arg gen_id "${COMP_ARTIFICIAL_ID}" \
  'map_values(.[].keyed_obj_map? | ."iree_module_generation_configs:\($gen_id)")'

# For compilation benchmarks
cat comp_config.json | \
  jq --arg gen_id "${COMP_ARTIFICIAL_ID}" \
  '.[].keyed_obj_map? | ."iree_module_generation_configs:\($gen_id)"'

This shows a module generation obj like:
{
  "composite_id": "1d26fcfdb7387659356dd99ce7e10907c8560b0925ad839334b0a6155d25167a",
  "imported_model": "394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337",
  "compile_config": "32a56c8d-cc6c-41b8-8620-1f8eda0b8223-compile-stats",
  "compile_flags": [
    "--iree-hal-target-backends=vulkan-spirv",
    "--iree-input-type=tosa",
    "--iree-vulkan-target-triple=valhall-unknown-android31",
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops",
    "--iree-vm-emit-polyglot-zip=true",
    "--iree-llvm-debug-symbols=false"
  ]
}

You can find the VMFB module at:
echo ${IREE_BUILD_DIR}/e2e_test_artifacts/iree_*_${COMP_ARTIFICIAL_ID}/module.vmfb

And the input imported MLIR at:
echo ${IREE_BUILD_DIR}/e2e_test_artifacts/iree_*_<imported_model>.mlir
```
**Find the input model of an imported MLIR**
```sh
# For execution benchmarks
cat exec_config.json | \
  jq --arg IMPORTED_ID "${IMPORTED_MODEL_ID}" \
  'map_values(.[].keyed_obj_map? | ."iree_imported_models:\($IMPORTED_ID)")'

# For compilation benchmarks
cat comp_config.json | \
  jq --arg IMPORTED_ID "${IMPORTED_MODEL_ID}" \
  '.[].keyed_obj_map? | ."iree_imported_models:\($IMPORTED_ID)"'

This shows an imported model object:
{
  "composite_id": "213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0",
  "model": "ebe7897f-5613-435b-a330-3cb967704e5e",
  "import_config": "8b2df698-f3ba-4207-8696-6c909776eac4"
}

You can find the input model at:
echo ${IREE_BUILD_DIR}/e2e_test_artifacts/model_ebe7897f-5613-435b-a330-3cb967704e5e_*
```

### Find and manipulate the benchmark definitions
All benchmarks are defined by the scripts under
[build_tools/python/benchmark_suites/iree](/build_tools/python/benchmark_suites/iree).

> TODO(#12215): Add a doc to explain how to hack the benchmark suite.

To find the code that generates a benchmark config:
1. Get the ids from `module_execution_config` and `compile_config` fields of the
   benchmark.
2. Find the defined constants of these ids (or their prefixes) in
   [build_tools/python/e2e_test_framework/unique_ids.py](/build_tools/python/e2e_test_framework/unique_ids.py).
3. Search in the code to see which generator use these constants.

To manipulate the benchmarks:
1. Modify the benchmark generation code under
   [build_tools/python/benchmark_suites/iree](/build_tools/python/benchmark_suites/iree).
2. Follow
   [tests/e2e/test_artifacts](https://github.com/openxla/iree/tree/main/tests/e2e/test_artifacts)
   to regenerate the cmake files.
3. Rebuild the benchmark suite.
4. Make sure to export the new benchmark configs again before running the
   benchmarks.
