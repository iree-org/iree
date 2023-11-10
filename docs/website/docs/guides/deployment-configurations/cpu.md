---
hide:
  - tags
tags:
  - CPU
icon: octicons/cpu-16
---

# CPU deployment

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

## :octicons-download-16: Prerequisites

### Get the IREE compiler

#### :octicons-package-16: Download the compiler from a release

Python packages are regularly published to
[PyPI](https://pypi.org/user/google-iree-pypi-deploy/). See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core `iree-compiler` package includes the LLVM-based CPU compiler:

=== "Stable releases"

    Stable release packages are
    [published to PyPI](https://pypi.org/user/google-iree-pypi-deploy/).

    ``` shell
    python -m pip install iree-compiler
    ```

=== ":material-alert: Nightly releases"

    Nightly releases are published on
    [GitHub releases](https://github.com/openxla/iree/releases).

    ``` shell
    python -m pip install \
      --find-links https://iree.dev/pip-release-links.html \
      --upgrade iree-compiler
    ```

!!! tip
    `iree-compile` is installed to your python module installation path. If you
    pip install with the user mode, it is under `${HOME}/.local/bin`, or
    `%APPDATA%Python` on Windows. You may want to include the path in your
    system's `PATH` environment variable:

    ```shell
    export PATH=${HOME}/.local/bin:${PATH}
    ```

#### :material-hammer-wrench: Build the compiler from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build
IREE for your host platform and the
[Android cross-compilation](../../building-from-source/android.md) or
[iOS cross-compilation](../../building-from-source/ios.md) page if you are cross
compiling for a mobile device. The `llvm-cpu` compiler backend is compiled in by
default on all platforms.

Ensure that the `IREE_TARGET_BACKEND_LLVM_CPU` CMake option is `ON` when
configuring for the host.

!!! tip
    `iree-compile` will be built under the `iree-build/tools/` directory. You
    may want to include this path in your system's `PATH` environment variable.

### Get the IREE runtime

You will need to get an IREE runtime that supports the local CPU HAL driver,
along with the appropriate executable loaders for your application.

You can check for CPU support by looking for the `local-sync` and `local-task`
drivers:

```console hl_lines="4 5"
$ iree-run-module --list_drivers

        cuda: CUDA (dynamic)
  local-sync: Local execution using a lightweight inline synchronous queue
  local-task: Local execution using the IREE multithreading task system
      vulkan: Vulkan 1.x (dynamic)
```

#### :material-hammer-wrench: Build the runtime from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build
IREE for your host platform and the
[Android cross-compilation](../../building-from-source/android.md) page if you
are cross compiling for Android. The local CPU HAL drivers are compiled in by
default on all platforms.

Ensure that the `IREE_HAL_DRIVER_LOCAL_TASK` and
`IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF` (or other executable loader) CMake
options are `ON` when configuring for the target.

## Compile and run a program

With the requirements out of the way, we can now compile a model and run it.

### :octicons-file-code-16: Compile a program

The IREE compiler transforms a model into its final deployable format in many
sequential steps. A model authored with Python in an ML framework should use the
corresponding framework's import tool to convert into a format (i.e.,
[MLIR](https://mlir.llvm.org/)) expected by the IREE compiler first.

Using MobileNet v2 as an example, you can download the SavedModel with trained
weights from
[TensorFlow Hub](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification)
and convert it using IREE's
[TensorFlow importer](../ml-frameworks/tensorflow.md). Then run the following
command to compile with the `llvm-cpu` target:

``` shell hl_lines="2"
iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    mobilenet_iree_input.mlir -o mobilenet_cpu.vmfb
```

!!! tip "Tip - CPU targets"

    The `--iree-llvmcpu-target-triple` flag tells the compiler to generate code
    for a specific type of CPU. You can see the list of supported targets with
    `iree-compile --iree-llvmcpu-list-targets`, or pass "host" to let LLVM
    infer the triple from your host machine (e.g. `x86_64-linux-gnu`).

    ```console
    $ iree-compile --iree-llvmcpu-list-targets

      Registered Targets:
        aarch64    - AArch64 (little endian)
        aarch64_32 - AArch64 (little endian ILP32)
        aarch64_be - AArch64 (big endian)
        arm        - ARM
        arm64      - ARM64 (little endian)
        arm64_32   - ARM64 (little endian ILP32)
        armeb      - ARM (big endian)
        riscv32    - 32-bit RISC-V
        riscv64    - 64-bit RISC-V
        wasm32     - WebAssembly 32-bit
        wasm64     - WebAssembly 64-bit
        x86        - 32-bit X86: Pentium-Pro and above
        x86-64     - 64-bit X86: EM64T and AMD64
    ```

!!! tip "Tip - CPU features"

    The `--iree-llvmcpu-target-cpu-features` flag tells the compiler to generate
    code using certain CPU "features", like SIMD instruction sets. Like the
    target triple, you can pass "host" to this flag to let LLVM infer the
    features supported by your host machine.

### :octicons-terminal-16: Run a compiled program

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

<!-- TODO(??): measuring performance -->

<!-- TODO(??): troubleshooting -->
