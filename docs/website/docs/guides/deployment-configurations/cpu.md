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

To compile a program for CPU execution:

1. Pick a CPU target supported by LLVM. By default, IREE includes these LLVM
   targets:

    * X86
    * ARM
    * AArch64
    * RISCV

    Other targets may work, but in-tree test coverage and performance work is
    focused on that list.

2. Pick one of IREE's supported executable formats:

    | Executable Format | Description                                           |
    | ----------------- | ----------------------------------------------------- |
    | Embedded ELF      | (Default) Portable, high performance dynamic library  |
    | System library    | Platform-specific dynamic library (.so, .dll, etc.)   |
    | VMVX              | Reference target                                      |

At runtime, CPU executables can be loaded using one of IREE's CPU HAL devices:

* `local-task`: asynchronous, multithreaded device built on IREE's "task"
   system
* `local-sync`: synchronous, single-threaded devices that executes work inline

## :octicons-download-16: Prerequisites

### Get the IREE compiler

#### :octicons-download-16: Download the compiler from a release

Python packages are distributed through multiple channels. See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core [`iree-base-compiler`](https://pypi.org/project/iree-base-compiler/)
package includes the compiler tools:

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-compiler-from-release.md"

#### :material-hammer-wrench: Build the compiler from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build
IREE for your host platform. The `llvm-cpu` compiler backend is compiled in by
default on all platforms, though you should ensure that the
`IREE_TARGET_BACKEND_LLVM_CPU` CMake option is `ON` when configuring.

!!! tip
    `iree-compile` will be built under the `iree-build/tools/` directory. You
    may want to include this path in your system's `PATH` environment variable.

### Get the IREE runtime

You will need to get an IREE runtime that supports the local CPU HAL driver,
along with the appropriate executable loaders for your application.

#### :octicons-download-16: Download the runtime from a release

Python packages are distributed through multiple channels. See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core [`iree-base-runtime`](https://pypi.org/project/iree-base-runtime/)
package includes the local CPU HAL drivers:

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-runtime-from-release.md"

#### :material-hammer-wrench: Build the runtime from source

Please make sure you have followed one of the
[Building from source](../../building-from-source/index.md) pages to build
IREE for your target platform. The local CPU HAL drivers and devices are
compiled in by default on all platforms, though you should ensure that the
`IREE_HAL_DRIVER_LOCAL_TASK` and `IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF`
(or other executable loader) CMake options are `ON` when configuring.

#### :octicons-checklist-24: Check for CPU devices

You can check for CPU support by looking for the `local-sync` and `local-task`
drivers and devices:

```console hl_lines="10-11"
--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-driver-list.md:2"
```

```console hl_lines="4-5"
--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-device-list-amd.md"
```

## Compile and run a program

With the requirements out of the way, we can now compile a model and run it.

### :octicons-file-code-16: Compile a program

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-import-onnx-mobilenet.md"

Then run the following command to compile with the `local` device, `llvm-cpu`
target compilation backend, and recommended optimization flags:

``` shell hl_lines="2-4"
iree-compile \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --iree-opt-level=O2 \
    --iree-llvmcpu-opt-level=O2 \
    --iree-opt-data-tiling \
    mobilenetv2.mlir -o mobilenet_cpu.vmfb
```

???+ tip "Tip - Target CPUs and CPU features"

    By default, the compiler will use a generic CPU target which will result in
    poor performance. A target CPU or target CPU feature set should be selected
    using one of these options:

    * `--iree-llvmcpu-target-cpu=...`
    * `--iree-llvmcpu-target-cpu-features=...`

    When not cross compiling, passing `--iree-llvmcpu-target-cpu=host` is
    usually sufficient on most devices.

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-optimization-options.md"

???+ tip "Tip - Codegen Optimizations"

    Use --iree-llvmcpu-opt-level=[O0,O1,O2,O3] to enable additional codegen
    optimizations. E.g., Reassociation for FP reductions is enabled at O2 and
    above. See
    [CodegenOptions.cpp](https://github.com/iree-org/iree/tree/main/compiler/src/iree/compiler/Codegen/Utils/CodegenOptions.cpp)
    for more details.

???+ tip "Tip - Data-Tiling"

    Use `--iree-opt-data-tiling` to enable the optimization that IREE developers
    have been working on. The option is default off for many reasons, but it has
    been used for performance when users target CPU. See
    [Data-Tiling](../../reference/optimization-options.md#data-tiling-iree-opt-data-tiling-off)
    for more details.

#### Choosing CPU targets

The `--iree-llvmcpu-target-triple` flag tells the compiler to generate code
for a specific type of CPU. You can see the list of supported targets with
`iree-compile --iree-llvmcpu-list-targets`, or use the default value of
"host" to let LLVM infer the triple from your host machine
(e.g. `x86_64-linux-gnu`).

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

### :octicons-terminal-16: Run a compiled program

To run the compiled program:

``` shell hl_lines="2"
iree-run-module \
    --device=local-task \
    --module=mobilenet_cpu.vmfb \
    --function=torch-jit-export \
    --input="1x3x224x224xf32=0"
```

The above assumes the exported function in the model is named `torch-jit-export`
and it expects one 224x224 RGB image. We are feeding in an image with all 0
values here for brevity, see `iree-run-module --help` for the format to specify
concrete values.

<!-- TODO(??): measuring performance -->

<!-- TODO(??): troubleshooting -->
