---
hide:
  - tags
tags:
  - CPU
---

# CPU Deployment

IREE supports efficient program execution on CPU devices by using
[LLVM](https://llvm.org/) to compile all dense computations in each program into
highly optimized CPU native instruction streams, which are embedded in one of
IREE's deployable formats.

To compile a program for CPU execution, pick one of IREE's supported executable
formats:

| Executable Format | Description                                           |
| ----------------- | ----------------------------------------------------- |
| embedded ELF      | portable, high performance dynamic library            |
| system library    | platform-specific dynamic library (.so, .dll, etc.)   |
| VMVX              | reference target                                      |

At runtime, CPU executables can be loaded using one of IREE's CPU HAL drivers:

* `local-task`: asynchronous, multithreaded driver built on IREE's "task"
   system
* `local-sync`: synchronous, single-threaded driver that executes work inline

!!! todo

    Add IREE's CPU support matrix: what architectures are supported; what
    architectures are well optimized; etc.

<!-- TODO(??): when to use CPU vs GPU vs other backends -->

## Get compiler and runtime

### Get compiler for CPU native instructions

#### Download as Python package

Python packages for various IREE functionalities are regularly published
to [PyPI](https://pypi.org/user/google-iree-pypi-deploy/). See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core `iree-compiler` package includes the LLVM-based CPU compiler:

``` shell
python -m pip install iree-compiler
```

!!! tip
    `iree-compile` is installed to your python module installation path. If you
    pip install with the user mode, it is under `${HOME}/.local/bin`, or
    `%APPDATA%Python` on Windows. You may want to include the path in your
    system's `PATH` environment variable:

    ```shell
    export PATH=${HOME}/.local/bin:${PATH}
    ```

#### Build compiler from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build IREE
for your host platform and the
[Android cross-compilation](../../building-from-source/android.md) or
[iOS cross-compilation](../../building-from-source/ios.md) page if you are cross
compiling for a mobile device. The LLVM (CPU) compiler backend is compiled in by
default on all platforms.

Ensure that the `IREE_TARGET_BACKEND_LLVM_CPU` CMake option is `ON` when
configuring for the host.

!!! tip
    `iree-compile` is under `iree-build/tools/` directory. You may want to
    include this path in your system's `PATH` environment variable.

## Compile and run the model

With the compiler and runtime for local CPU execution, we can now compile a
model and run it.

### Compile the model

The IREE compiler transforms a model into its final deployable format in many
sequential steps. A model authored with Python in an ML framework should use the
corresponding framework's import tool to convert into a format (i.e.,
[MLIR](https://mlir.llvm.org/)) expected by the IREE compiler first.

Using MobileNet v2 as an example, you can download the SavedModel with trained
weights from
[TensorFlow Hub](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification)
and convert it using IREE's
[TensorFlow importer](../ml-frameworks/tensorflow.md). Then,

#### Compile using the command-line

Run the following command (passing `--iree-input-type=` as needed for your
import tool):

``` shell hl_lines="2"
iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    --iree-input-type=stablehlo \
    iree_input.mlir -o mobilenet_cpu.vmfb
```

where `iree_input.mlir` is the imported program.

!!! tip

    The `--iree-llvmcpu-target-triple=` flag tells the compiler to generate code
    for a specific type of CPU. You can see the list of supported targets with
    `iree-compile --iree-llvmcpu-list-targets`, or omit the flag to let LLVM
    infer the triple from your host machine (e.g. `x86_64-linux-gnu`).

### Get IREE runtime with local CPU HAL driver

You will need to get an IREE runtime that supports the local CPU HAL driver,
along with the appropriate executable loaders for your application.

#### Build runtime from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build IREE
for your host platform and the
[Android cross-compilation](../../building-from-source/android.md) page if you are
cross compiling for Android. The local CPU HAL drivers are compiled in by
default on all platforms.

<!-- TODO(??): a way to verify the driver is compiled in and supported -->

Ensure that the `IREE_HAL_DRIVER_LOCAL_TASK` and
`IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF` (or other executable loader) CMake
options are `ON` when configuring for the target.

### Run the model

#### Run using the command-line

In the build directory, run the following command:

``` shell hl_lines="2"
tools/iree-run-module \
    --device=local-task \
    --module=mobilenet_cpu.vmfb \
    --function=predict \
    --input="1x224x224x3xf32=0"
```

The above assumes the exported function in the model is named as `predict` and
it expects one 224x224 RGB image. We are feeding in an image with all 0 values
here for brevity, see `iree-run-module --help` for the format to specify
concrete values.

<!-- TODO(??): deployment options -->

<!-- TODO(??): measuring performance -->

<!-- TODO(??): troubleshooting -->
