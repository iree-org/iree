# Run on a Bare-Metal Platform

IREE supports CPU model execution on bare-metal platforms. That is, platforms
without operating system support, for which executables are built using
machine-specific linker scripts and/or board support packages (BSPs).

Bare-metal deployment typically uses IREE's LLVM compiler target much like the
[CPU configuration](./cpu.md), but using a limited subset of IREE's CPU HAL
driver code at runtime to load and execute compiled programs.

## Prerequisites

Out-of-tree bare-metal platform tools and source code for the system should be
ready, such as

* Compilation toolchain
* Platform linker script
* Firmware libraries

Please follow the
[instructions](./cpu.md#get-compiler-for-cpu-native-instructions)
to retrieve the IREE compiler.

## Compile the model for bare-metal

The model can be compiled with the following command (assuming the path to
`iree-compile` is in your system's `PATH`):

``` shell
iree-compile \
    --iree-stream-partitioning-favor=min-peak-memory \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvm-target-triple=x86_64-pc-linux-elf \
    --iree-llvm-debug-symbols=false \
    samples/models/simple_abs.mlir \
    -o /tmp/simple_abs_cpu.vmfb

```

In which

* `-iree-stream-partitioning-favor=min-peak-memory`: Optimize for minimum peak
    memory usage at the cost of concurrency - include when targeting
    single-threaded execution to reduce memory consumption.
* `iree-hal-target-backends=llvm-cpu`: Compile using the LLVM CPU target
* `iree-llvm-target-triple`: Use the `<arch>-pc-linux-elf` LLVM target triple
    so the artifact has a fixed ABI to be rendered by the
    [elf_module library](https://github.com/openxla/iree/tree/main/iree/hal/local/elf)
* `iree-llvm-debug-symbols=false`: To reduce the artifact size

See [generate.sh](https://github.com/openxla/iree/blob/main/iree/hal/local/elf/testdata/generate.sh)
for example command-line instructions of some common architectures

You can replace the MLIR file with the other MLIR model files, following the
[instructions](./cpu.md#compile-the-model)

### Compiling the bare-metal model for static-library support

See the [static_library](https://github.com/openxla/iree/tree/main/samples/static_library)
demo sample for an example and instructions on running a model with IREE's
`static_library_loader`.

By default, the demo targets the host machine when compiling. To produce a
bare-metal compatible model, run `iree-compile` as in the previous example
and add the additional `-iree-llvm-static-library-output-path=` flag to specify
the static library destination. This will produce a `.h\.o` file to link
directly into the target application.

## Build bare-metal runtime from the source

A few CMake options and macros should be set to build a subset of IREE runtime
libraries compatible with the bare-metal platform. We assume there's no
multi-thread control nor system library support in the bare-metal system. The
model execution is in a single-thread synchronous fashion.

### Set CMake options

* `set(IREE_BUILD_COMPILER OFF)`: Build IREE runtime only
* `set(CMAKE_SYSTEM_NAME Generic)`: Tell CMake to skip targeting a specific
  operating system
* `set(IREE_BINDINGS_TFLITE OFF)`: Disable the TFLite binding support
* `set(IREE_ENABLE_THREADING OFF)`: Disable multi-thread library support
* `set(IREE_HAL_DRIVER_DEFAULTS OFF)`: Disable HAL drivers by default, then
  enable the synchronous HAL drivers with `set(IREE_HAL_DRIVER_LOCAL_SYNC ON)`
* `set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF)`: Disable HAL executable
  loaders by default, then enable the CPU codegen and VMVX loaders with
  `set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON)` and
  `set(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE ON)`
* `set(IREE_BUILD_TESTS OFF)`: Disable tests until IREE supports running them
  on bare-metal platforms
* `set(IREE_BUILD_SAMPLES ON)`: Build
  [simple_embedding](https://github.com/openxla/iree/tree/main/samples/simple_embedding)
  example

!!! todo
    Clean the list up after [#6353](https://github.com/openxla/iree/issues/6353)
    is fixed.

Also, set the toolchain-specific cmake file to match the tool path, target
architecture, target abi, linker script, system library path, etc.

### Define IREE macros

* `-DIREE_PLATFORM_GENERIC`: Let IREE to build the runtime library without
  targeting a specific platform.
* `-DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=1`: Disable thread synchronization
  support. Must only be used if there's a single thread.
* `-DIREE_FILE_IO_ENABLE=0`: Disable file I/O.
* `-DIREE_TIME_NOW_FN`: A function to return the system time. For the bare-metal
  system, it can be set as `-DIREE_TIME_NOW_FN=\"\{ return 0;\}\"` as there's no
  asynchronous wait handling.
* `-DIREE_WAIT_UNTIL_FN`: A function to wait until the given time in
  nanoseconds. Must match the signature `bool(uint64_t nanos)` and return
  false if the wait failed.

Examples of how to setup the CMakeLists.txt and .cmake file:

* [IREE RISC-V toolchain cmake](https://github.com/openxla/iree/blob/main/build_tools/cmake/riscv.toolchain.cmake)
* [IREE Bare-Metal Arm Sample](https://github.com/iml130/iree-bare-metal-arm)
* [IREE Bare-Metal RV32 Sample](https://github.com/AmbiML/iree-rv32-springbok)

## Bare-metal execution example

See
[simple_embedding for generic platform](https://github.com/openxla/iree/blob/main/samples/simple_embedding/README.md#generic-platform-support)
to see how to use the IREE runtime library to build/run the IREE model for the
bare-metal target.
