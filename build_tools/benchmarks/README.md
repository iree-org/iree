# IREE Benchmark Suites Tool

This directory contains the tools to run IREE benchmark suites and generate
reports. More information about benchmark suites can be found [here](benchmarks/README.md).

## Benchmark Tools
Currently we have `run_benchmarks_on_android.py` and `run_benchmarks_on_linux.py` to run
benchmark suites on Android devices (with `adb`) and Linux local machines.

The available arguments can be shown with `--help`. Here we list some common
usages (assume the target `iree-benchmark-suites` is built under `$IREE_BUILD`):

**Run all benchmarks**
```sh
./run_benchmarks_on_linux.py \
  --normal_benchmark_tool_dir=$IREE_BUILD/iree/tools \
  --output=results.json $IREE_BUILD
```

**Run all benchmarks and capture the tracy files**
```sh
./run_benchmarks_on_linux.py \
  --normal_benchmark_tool_dir=$IREE_BUILD/iree/tools \
  --traced_benchmark_tool_dir=$IREE_TRACED_BUILD/iree/tools \
  --trace_capture_tool=${path of iree-tracy-capture} \
  --capture_tarball=captured_tracy_files.tar
  --output=results.json $IREE_BUILD
```

**Run selected benchmarks with the filters**
```sh
./run_benchmarks_on_linux.py \
  --normal_benchmark_tool_dir=$IREE_BUILD/iree/tools \
  --model_name_regex="MobileBertSquad" \
  --driver_filter_regex="dylib" \
  --mode_regex="4-threads" \
  --output=results.json $IREE_BUILD
```

## Generate Benchmark Report
The tools here are mainly designed for benchmark automation pipeline.
The `post_benchmarks_as_pr_comment.py` and `upload_benchmarks_to_dashboard.py`
are used to upload and post reports to the dashboard.

If you want to generate a comparison report locally, you can use `diff_local_benchmarks.py` 
to compare two result json files and generate the report. For example:
```sh
./diff_local_benchmarks.py --base before.json --target after.json > report.md
```
