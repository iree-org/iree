# Benchmark Comparisons

This directory includes a set of scripts to run benchmarks with both IREE and
TFLite in order to get apples-to-apples comparisons in latency and memory usage.
The output is a .csv file.

It assumes a directory structure like below:

```text
<root-benchmark-dir>/
  └── ./benchmark_model (TFLite benchmark binary)
      ./iree-benchmark-module (IREE benchmark binary)
  ├── setup/
        ├── set_adreno_gpu_scaling_policy.sh
        ├── set_android_scaling_governor.sh
        └── set_pixel6_gpu_scaling_policy.sh
  ├── test_data/
  └── models/
        ├── tflite/*.tflite
        └── iree/
              └── <target>/*.vmfb e.g. llvm-cpu, vulkan, cuda.
```

# Prerequisites

## Android

When running benchmarks on an Android device, some initial setup is involved.

### Install Android NDK and ADB

Detailed steps
[here](https://openxla.github.io/iree/building-from-source/android/#install-android-ndk-and-adb).

### Install the Termux App and the Python Interpreter

1.  Download Termux .apk
    [here](https://github.com/termux/termux-app/releases/download/v0.118.0/termux-app_v0.118.0+github-debug_arm64-v8a.apk)
2.  With the device connected, run `adb install -g <termux.apk>`
3.  Open the app on the device and in the terminal, install python: `pkg install
    python`.

## CUDA

If benchmarking on desktop with CUDA, make sure you have the
[latest CUDA Toolkit SDK](https://developer.nvidia.com/cuda-downloads)
installed.

# Setup

The scripts `setup_desktop.sh` and `setup_mobile.sh` will run through the steps
of retrieving benchmarking artifacts, compiling binaries, model files, etc. and
then run the benchmarks. Note that some parts are interactive and require user
input.

# Running Benchmarks

Once all benchmarking artifacts are setup, benchmarks can be run with the
command:

```shell
ROOT_DIR=/tmp/benchmarks

python build_tools/benchmarks/comparisons/run_benchmarks.py \
  --device_name=desktop --base_dir=${ROOT_DIR} \
  --output_dir=${ROOT_DIR}/output --mode=desktop
```

# Adding Models and Runtimes

To add a new model or runtime, simply create `BenchmarkCommand` and
`BenchmarkCommandFactory` classes. For an example, see
`mobilebert_fp32_commands.py`, which includes commands to run a MobileBert FP32
model on (Desktop+Mobile) x (CPU+GPU) x (IREE+TFLite).

Once these classes are created, an instance of the new factory can be added to
the `command_factory` list in `run_benchmarks.py`.

```python
def main(args):
  # Create factories for all models to be benchmarked.
  command_factory = []
  command_factory.append(MobilebertFP32CommandFactory(args.base_dir))
  command_factory.append(MyNewModelCommandFactory(args.base_dir))
  ...
```

Also make sure to add the necessary setup commands for the new model in
`setup_desktop.sh` and `setup_mobile.sh`.
