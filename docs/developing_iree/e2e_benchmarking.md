---
layout: default
permalink: developing-iree/e2e-benchmarking
title: Benchmarking TensorFlow with IREE and TFLite
nav_order: 6
parent: Developing IREE
---

# Benchmark IREE and TFLite
{: .no_toc }

<!-- TODO(meadowlark): Update this doc once the API is stable and change default to cmake. -->

> Note
> {: .label .label-blue }
> The TensorFlow integrations are currently being
  refactored. The `bazel` build is deprecated. Refer to
  https://google.github.io/iree/get-started/getting-started-python for a general
  overview of how to build and execute the e2e tests.

We use our end-to-end TensorFlow integration tests to test compilation and
numerical accuracy, and to generate compilation and benchmarking artifacts.
This allows us to validate that our benchmarks are behaving as we expect them
to, and to run them using valid inputs for each model.

This guide assumes that you can run the tensorflow integration tests. See
[this doc](https://google.github.io/iree/get-started/getting-started-python)
for more information. That doc also covers writing new tests, which you'll need
to do if you'd like to benchmark a new TensorFlow model.

## 1. Run IREE's E2E TensorFlow tests to generate the benchmarking artifacts

```shell
# Continuing from the "Running Python Tests" section of the doc linked above.
# We need not only Python bindings, but also the TensorFlow compiler frontend.
# Self-contained commands from that doc --- skip that if you have already
# completed a build with Python bindings AND TensorFlow compiler frontend.
$ cd iree-build/  # Make and cd into some build directory
$ cmake ../iree -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_BUILD_PYTHON_BINDINGS=ON  \
    -DIREE_BUILD_TENSORFLOW_COMPILER=ON
$ cmake --build .
# Also from the Python get-started doc, set this environment variable:
$ export PYTHONPATH=$(pwd)/bindings/python
```

```shell
# --target_backends: All tests allow you to specify one or more backends to generate benchmarking artifacts for.
# --artifacts_dir: The default location for these artifacts is under /tmp/iree/modules
# This is a Python 3 program. On some systems, such as Debian derivatives,
# use 'python3' instead of 'python'.
$ python ../iree/integrations/tensorflow/e2e/matrix_ops_static_test.py \
    --target_backends=iree_vmla
# View the generated artifacts:
$ tree /tmp/iree/modules/MatrixOpsStaticModule/
```

```shell
# Some tests use additional flags to specify features of the Module to test/benchmark:
# --model: the tf.keras.applications model to test
# --data: the dataset (and corresponding image shapes) to create the model for
$ python ../iree/integrations/tensorflow/e2e/keras/applications/applications_test.py \
    --target_backends=iree_vmla \
    --model=MobileNetV3Small \
    --data=imagenet
# View the generated artifacts:
$ tree /tmp/iree/modules/MobileNetV3Small/
```

Each test/module has a folder with the following artifacts (filtered to only
include those relevant for benchmarking):

```shell
# Example for a generic module `ModuleName`:
/tmp/iree/modules/ModuleName
  ├── iree_vmla  # Or any other IREE backend.
  │   ├── compiled.vmfb
  │   │   # A flatbuffer containing IREE's compiled code.
  │   └── traces
  │       # Directory with a trace for each unittest in vision_model_test.py.
  │       ├── traced_function_1
  │       │   # Directory storing logs and serialization for a specific trace.
  │       │   └── flagfile
  │       │       # An Abseil flagfile containing arguments
  │       │       # iree-benchmark-module needs to benchmark this trace.
  │       ├── traced_function_2
  │       └── ...
  └── tflite
      ├── module_method_1.tflite
      │   # A method on ModuleName compiled to bytes with TFLite, which can
      │   # be used by the TFLite's benchmark_model binary.
      ├── module_method_2.tflite
      ├── ...
      └── traces
          ├── traced_function_1
          │   └── graph_path
          │       # In general, a trace's name does not have to match the name
          │       # of the method(s) on the tf.Module that it calls. This file
          │       # points to the correct module_method_*.tflite graph file
          │       # for TFLite's benchmark_model to use.
          ├── traced_function_2
          └── ...

# Example for MatrixOpsStaticModule:
/tmp/iree/modules/MatrixOpsStaticModule
  ├── iree_llvmaot
  │   ├── compiled.vmfb
  │   └── traces
  │       ├── basic_matmul
  │       │   └── flagfile
  │       ├── matmul_broadcast_singleton_dimension
  │       │   └── flagfile
  │       ├── matmul_lhs_batch
  │       │   └── flagfile
  │       └── matmul_rhs_batch
  │           └── flagfile
  ├── iree_vmla
  │   ├── compiled.vmfb
  │   └── traces  # ...same as iree_llvmaot/traces above.
  ├── iree_vulkan
  │   ├── compiled.vmfb
  │   └── traces  # ...same as iree_llvmaot/traces above.
  └── tflite
      ├── basic_matmul.tflite
      ├── matmul_broadcast_singleton_dimension.tflite
      ├── matmul_lhs_batch.tflite
      ├── matmul_rhs_batch.tflite
      └── traces
          ├── basic_matmul
          │   └── graph_path
          ├── matmul_broadcast_singleton_dimension
          │   └── graph_path
          ├── matmul_lhs_batch
          │   └── graph_path
          └── matmul_rhs_batch
              └── graph_path
```

## 2. Benchmarking IREE on desktop

### 2.1 Optional: Build the `iree-benchmark-module`

This step is optional, but allows running the benchmarks without running `bazel`
at the same time.

```shell
$ bazel build -c opt //iree/tools:iree-benchmark-module
```

This creates `bazel-bin/iree/tools/iree-benchmark-module`. The rest of the guide
will use this binary, but you could also use
`bazel run iree/tools:iree-benchmark-module` in its place if your prefer.

### 2.2 Benchmark the model on IREE

The E2E tests generate a flagfile with all of the information that
`iree-benchmark-module` needs to benchmark each trace. Namely it handles
providing the following flags:

| Flag              | Description                                      |
|-------------------|--------------------------------------------------|
| --module_file     | Absolute path to the IREE compiled VM flatbuffer |
| --function_inputs | A comma delimited string of input tensors        |
| --driver          | The backend driver to use for the benchmark      |
| --entry_function  | The method on the TensorFlow module to benchmark |

You can find the flagfile to benchmark a specific TensorFlow module on a
specific IREE backend and trace at the following path:

```shell
/tmp/iree/modules/ModuleName/backend/traces/trace_name/flagfile
```

For example, if we wanted to benchmark a static left-hand-side batched matmul
using `MatrixOpsStaticModule` on VMLA we would run the following command:

```shell
$ ./bazel-bin/iree/tools/iree-benchmark-module \
  --flagfile="/tmp/iree/modules/MatrixOpsStaticModule/iree_vmla/traces/matmul_lhs_batch/flagfile"
```

If you ran `applications_test.py` then you'll be able to benchmark
`MobileNetV3Small` on `imagenet` input shapes. For example:

```shell
$ ./bazel-bin/iree/tools/iree-benchmark-module \
  --flagfile="/tmp/iree/modules/MobileNetV3Small/imagenet/iree_vmla/traces/predict/flagfile"
```

## 3. Benchmarking TFLite on desktop

### 3.1 Build TFLite's `benchmark_model` binary

```shell
# Enter the TensorFlow Bazel workspace.
$ cd third_party/tensorflow/

# Build the benchmark_model binary.
$ bazel build --copt=-mavx2 -c opt \
  //tensorflow/lite/tools/benchmark:benchmark_model

# By default, TFLite/x86 uses various matrix multiplication libraries.
# It is possible to force it to only use Ruy for all matrix multiplications.
# That is the default on ARM but not on x86. This will overwrite the
# previous binary unless you move it.
#
# Note that Ruy takes care of -mavx2 and other AVX extensions internally,
# so this passing this flag here isn't going to make a difference to
# matrix multiplications. However, the rest of TFLite's kernels outside
# of ruy will still benefit from -mavx2.
$ bazel build --copt=-mavx2 -c opt \
  --define=tflite_with_ruy=true \
  //tensorflow/lite/tools/benchmark:benchmark_model

# The binary can now be found in the following directory:
$ ls bazel-bin/tensorflow/lite/tools/benchmark/
```

### 3.2 Benchmark the model on TFLite

TFLite doesn't support flagfiles, so we need to manually pass the path to the
graph file via `cat`. TFLite will generate fake inputs for the model.

Using `MatrixOpsStaticModule`'s left-hand-side batched matmul again as an
example we can run the benchmark as follows:

```shell
# Run within `third_party/tensorflow/`.
$ ./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=$(cat "/tmp/iree/modules/MatrixOpsStaticModule/tflite/traces/matmul_lhs_batch/graph_path") \
  --warmup_runs=1 \
  --num_threads=1 \
  --num_runs=100 \
  --enable_op_profiling=true
```

## 4. Benchmarking IREE on Android

### 4.1 Prepare the benchmarking tools

IREE only supports compiling to Android with CMake. Documentation on setting up
your environment to cross-compile to Android can be found
[here](https://google.github.io/iree/get-started/getting-started-android-cmake).

```shell
# After following the instructions above up to 'Build all targets', the
# iree-benchmark-module binary should be in the following directory:
$ ls build-android/iree/tools/

# Copy the benchmarking binary to phone.
$ adb push build-android/iree/tools/iree-benchmark-module /data/local/tmp
```

### 4.2 Push the IREE's compilation / benchmarking artifacts to the device

In this example we'll only copy over the files we need to benchmark a single
module on a single backend, but you can easily copy all of the modules over
as well.

Using `MatrixOpsStaticModule`'s left-hand-side batched matmul again as an
example:

```shell
# Make a directory for the module/backend pair we want to benchmark.
$ adb shell mkdir -p /data/local/tmp/MatrixOpsStaticModule/iree_vmla/

# Transfer the files.
$ adb push /tmp/iree/modules/MatrixOpsStaticModule/iree_vmla/* \
  /data/local/tmp/MatrixOpsStaticModule/iree_vmla/
```

### 4.3 Benchmark the module

```shell
$ adb shell /data/local/tmp/iree-benchmark-module \
  --flagfile="/data/local/tmp/MatrixOpsStaticModule/iree_vmla/traces/matmul_lhs_batch/flagfile" \
  --module_file="/data/local/tmp/MatrixOpsStaticModule/iree_vmla/compiled.vmfb"
```

Note: Because the flagfile uses absolute paths, the `--module_file` flag must be
specified manually if the location of the compiled flatbuffer (`compiled.vmfb`)
changes. The flagfile can still take care of specifying the input data, driver
and entry function however.

## 5. Benchmarking TFLite on Android

### 5.1 Prepare the benchmarking tools

There are three options for getting TFLite's `benchmark_model` binary for
Android.

The first two are to build it directly, either in a
[`docker` container](https://www.tensorflow.org/lite/guide/build_android#set_up_build_environment_using_docker)
or
[in your own
environment](https://www.tensorflow.org/lite/guide/build_android#set_up_build_environment_without_docker).
To build TensorFlow tools with Android:

- Run `./configure` under TensorFlow repo.
- Add the following section to the TensorFlow WORKSPACE file.

```
android_ndk_repository(
    name="androidndk",
    path="/full/path/to/android_ndk",
)
```

TODO(hanchung): Place the Android setup to somewhere outside IREE, e.g.,
TensorFlow.

Then you can configure the TFLite `benchmark_model` binary in the following
ways:

```shell
# Build the benchmark_model binary without any add-ons.
# Note that unlike TFLite/x86, TFLite/ARM uses Ruy by default for all
# matrix multiplications (No need to pass tflite_with_ruy), except for some
# matrix*vector products. Below we show how to force using ruy also for that.
$ bazel build -c opt \
  --config=android_arm64 \
  --cxxopt='--std=c++17' \
  //tensorflow/lite/tools/benchmark:benchmark_model

# Copy the benchmarking binary to phone and allow execution.
$ adb push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  /data/local/tmp
$ adb shell chmod +x /data/local/tmp/benchmark_model
```

```shell
# Build the benchmark_model binary using ruy even for matrix*vector
# products. This is only worth trying in models that are heavy on matrix*vector
# shapes, typically LSTMs and other RNNs.
$ bazel build -c opt \
  --config=android_arm64 \
  --cxxopt='--std=c++17' \
  --copt=-DTFLITE_WITH_RUY_GEMV \
  //tensorflow/lite/tools/benchmark:benchmark_model

# Rename the binary for comparison with the standard benchmark_model.
$ mv bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_plus_ruy_gemv
$ adb push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_plus_ruy_gemv \
  /data/local/tmp/
$ adb shell chmod +x /data/local/tmp/benchmark_model_plus_ruy_gemv
```

```shell
# Build the benchmark_model binary with flex.
$ bazel build -c opt \
  --config=android_arm64 \
  --cxxopt='--std=c++17' \
  //tensorflow/lite/tools/benchmark:benchmark_model_plus_flex

# Copy the benchmarking binary to phone and allow execution.
$ adb push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_plus_flex \
  /data/local/tmp
$ adb shell chmod +x /data/local/tmp/benchmark_model_plus_flex
```

Alternatively, you can download and install the
[Android Benchmark App](https://www.tensorflow.org/lite/performance/measurement#android_benchmark_app). If you choose to install the app then
you'll have to modify the benchmarking commands below slightly, as shown in
[this example](https://www.tensorflow.org/lite/performance/measurement#run_benchmark).

### 5.2 Run the benchmark

```shell
# Copy the data over to the phone.
$ mkdir -p /data/local/tmp/MatrixOpsStaticModule/tflite
$ adb push /tmp/iree/modules/MatrixOpsStaticModule/tflite/* \
  /data/local/tmp/MatrixOpsStaticModule/tflite/
```

```shell
# Benchmark with TFLite.
$ adb shell taskset f0 /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/MatrixOpsStaticModule/tflite/matmul_lhs_batch.tflite \
  --warmup_runs=1 \
  --num_threads=1 \
  --num_runs=10 \
```

```shell
# Benchmark with TFLite + RUY GEMV
$ adb shell taskset f0 /data/local/tmp/benchmark_model_plus_ruy_gemv \
  --graph=/data/local/tmp/MatrixOpsStaticModule/tflite/matmul_lhs_batch.tflite \
  --warmup_runs=1 \
  --num_threads=1 \
  --num_runs=10 \
```

```shell
# Benchmark with TFLite + Flex.
$ adb shell taskset f0 /data/local/tmp/benchmark_model_plus_flex \
  --graph=/data/local/tmp/MatrixOpsStaticModule/tflite/matmul_lhs_batch.tflite \
  --warmup_runs=1 \
  --num_threads=1 \
  --num_runs=10 \
```

```shell
# Benchmark with TFLite running on GPU.
$ adb shell taskset f0 /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/MatrixOpsStaticModule/tflite/matmul_lhs_batch.tflite \
  --warmup_runs=1 \
  --num_threads=1 \
  --num_runs=10 \
  --use_gpu=true
```

Running benchmark on GPU won't give op profiling. To detailed profiling
information for GPU you can run the following script:

```shell
# Op profiling on GPU using OpenCL backend.
$ sh tensorflow/lite/delegates/gpu/cl/testing/run_performance_profiling.sh \
  -m /data/local/tmp/MatrixOpsStaticModule/tflite/matmul_lhs_batch.tflite
```

Note: You will have to manually specify the TFLite graph that you want to
benchmark, as the `graph_path` file assumes that the graph has not moved. The
name of the `.tflite` graph that you need to benchmark _may_ be different from
the name of the trace that you want to benchmark, but you can use `cat` on
the `graph_path` file to verify the correct `.tflite` filename if you're unsure.

Tip:<br>
nbsp;&nbsp;&nbsp;&nbsp;Sometimes `benchmark_tool` falls back to use CPU even
when `use_gpu` is set. To get more information, you can turn on traces in the
tool by `adb shell setprop debug.tflite.trace 1`.

### Profile

There are 2 profilers built into TFLite's `benchmark_model` program. Both of
them impact latencies, so they should only be used to get a breakdown of the
relative time spent in each operator type, they should not be enabled for the
purpose of measuring a latency.

The first is `enable_op_profiling`. It's based on timestamps before and after
each op. It's a runtime command-line flag taken by `benchmark_model`. Example:

```
$ adb shell taskset f0 /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/MatrixOpsStaticModule/tflite/matmul_lhs_batch.tflite \
  --warmup_runs=1 \
  --num_threads=1 \
  --num_runs=10 \
  --enable_op_profiling=true
```

The second is `ruy_profiler`. Despite its name, it's available regardless of
whether `ruy` is used for the matrix multiplications. It's a sampling profiler,
which allows it to provide some more detailed information, particularly on
matrix multiplications. It's a build-time switch:

```
$ bazel build \
  --define=ruy_profiler=true \
  -c opt \
  --config=android_arm64 \
  //tensorflow/lite/tools/benchmark:benchmark_model
```

The binary thus built can be run like above, no command-line flag needed.