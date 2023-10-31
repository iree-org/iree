---
hide:
  - tags
tags:
  - CPU
icon: octicons/cpu-16
---

# Running on a bare-metal platform

IREE supports model execution via CPU on bare-metal platforms. Bare metal
platforms have no operating system support, and executables are built using
machine-specific linker scripts and/or board support packages (BSPs).

Bare-metal deployment typically uses IREE's LLVM compiler target backend much
like the [CPU configuration](./cpu.md), but using a limited subset of IREE's CPU
HAL driver code at runtime to load and execute compiled programs.

## :octicons-download-16: Prerequisites

Out-of-tree bare-metal platform tools and source code for the system should be
ready, such as

* Compilation toolchain
* Platform linker script
* Firmware libraries

Please follow the
[instructions](./cpu.md#get-the-iree-compiler)
to retrieve the IREE compiler.

## :octicons-file-code-16: Compile the model for bare-metal

The model can be compiled with the following command:

``` shell
iree-compile \
    --iree-stream-partitioning-favor=min-peak-memory \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=x86_64-pc-linux-elf \
    --iree-llvmcpu-debug-symbols=false \
    samples/models/simple_abs.mlir \
    -o /tmp/simple_abs_cpu.vmfb

```

In which

* `--iree-stream-partitioning-favor=min-peak-memory`: Optimize for minimum peak
    memory usage at the cost of concurrency - include when targeting
    single-threaded execution to reduce memory consumption.
* `--iree-hal-target-backends=llvm-cpu`: Compile using the LLVM CPU target
* `--iree-llvmcpu-target-triple`: Use the `<arch>-pc-linux-elf` LLVM target triple
    so the artifact has a fixed ABI to be rendered by the
    [elf_module library](https://github.com/openxla/iree/tree/main/runtime/src/iree/hal/local/elf)
* `--iree-llvmcpu-debug-symbols=false`: To reduce the artifact size

See [generate.sh](https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/local/elf/testdata/generate.sh)
for example command-line instructions of some common architectures.

You can replace the MLIR file with the other MLIR model files, following the
[instructions](./cpu.md#compile-a-program).

### Compiling the bare-metal model for static-library support

See the [static_library](https://github.com/openxla/iree/tree/main/samples/static_library)
demo sample for an example and instructions on running a model with IREE's
`static_library_loader`.

By default, the demo targets the host machine when compiling. To produce a
bare-metal compatible model, run `iree-compile` as in the previous example
and add the additional `-iree-llvmcpu-static-library-output-path=` flag to specify
the static library destination. This will produce a `.h\.o` file to link
directly into the target application.

## :material-hammer-wrench: Build bare-metal runtime from source

A few CMake options and macros should be set to build a subset of IREE runtime
libraries compatible with the bare-metal platform. We assume there's no
multi-thread control nor system library support in the bare-metal system. The
model execution is in a single-thread synchronous fashion.

### :octicons-sliders-16: Set CMake options

``` cmake
# Build the IREE runtime only
set(IREE_BUILD_COMPILER OFF)

# Tell CMake to skip targeting a specific operating system
set(CMAKE_SYSTEM_NAME Generic)

# Disable multi-thread library support
set(IREE_ENABLE_THREADING OFF)

# Only enable the local synchronous HAL driver
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)

# Only enable some executable loaders
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON)
set(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE ON)

# Only enable the embedded ELF executable plugin
set(IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS OFF)
set(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF ON)

# Disable tests until IREE supports running them on bare-metal platforms
set(IREE_BUILD_TESTS OFF)

# Build samples
set(IREE_BUILD_SAMPLES ON)
```

!!! todo
    Clean the list up after [#6353](https://github.com/openxla/iree/issues/6353)
    is fixed.

Also, set the toolchain-specific cmake file to match the tool path, target
architecture, target abi, linker script, system library path, etc.

### :octicons-gear-16: Define IREE macros

These macros should be defined, either in C/C++ or via CMake options like

``` cmake
set(MY_FLAGS "-DIREE_PLATFORM_GENERIC=1")
set(CMAKE_C_FLAGS ${MY_FLAGS} ${CMAKE_C_FLAGS})
set(CMAKE_CXX_FLAGS ${MY_FLAGS} ${CMAKE_CXX_FLAGS})
```

| Macro | Description |
| ----- | ----------- |
| `IREE_PLATFORM_GENERIC` | Let IREE build the runtime library without targeting a specific platform. |
| `IREE_SYNCHRONIZATION_DISABLE_UNSAFE=1` | Disable thread synchronization support.<br>Must only be used if there's a single thread. |
| `IREE_FILE_IO_ENABLE=0` | Disable file I/O. |
| `IREE_TIME_NOW_FN` | A function to return the system time. For the bare-metal systems, it can be set as `IREE_TIME_NOW_FN=\"\{ return 0;\}\"` as there's no asynchronous wait handling. |
| `IREE_WAIT_UNTIL_FN` | A function to wait until the given time in nanoseconds. Must match the signature `bool(uint64_t nanos)` and return false if the wait failed. |

Examples of how to setup the CMakeLists.txt and .cmake file:

* [IREE RISC-V toolchain cmake](https://github.com/openxla/iree/blob/main/build_tools/cmake/riscv.toolchain.cmake)
* [IREE Bare-Metal Arm Sample](https://github.com/iml130/iree-bare-metal-arm)
* [IREE Bare-Metal RV32 Sample](https://github.com/AmbiML/iree-rv32-springbok)

## Bare-metal execution example

See
[simple_embedding for generic platform](https://github.com/openxla/iree/blob/main/samples/simple_embedding/README.md#generic-platform-support)
to see how to use the IREE runtime library to build/run the IREE model for the
bare-metal target.
