# Run on a Bare-Metal Platform

IREE supports CPU model execution on a bare-metal platform. That is, a platform
without operating system support, and the executable is built with the
machine-specific linker script and/or the board support package (BSP).

Bare-metal deployment typically uses IREE's LLVM compiler target much like the
[CPU - Dylib](./cpu-dylib.md)
configuration, but using a limited subset of IREE's CPU HAL driver at runtime to
load and execute compiled programs.

## Prerequisites

Out-of-tree bare-metal platform tools and source code for the system should be
ready, such as

* Compilation toolchain
* Platform linker script
* Firmware libraries

Please follow the
[instructions](./cpu-dylib.md#get-compiler-for-cpu-native-instructions)
to retrieve the IREE compiler.

## Compile the model for bare-metal

The model can be compiled with the following command from the IREE compiler
build directory

``` shell hl_lines="3 4 5 6"
iree/tools/iree-translate \
    -iree-mlir-to-vm-bytecode-module \
    -iree-hal-target-backends=dylib-llvm-aot \
    -iree-llvm-link-embedded=true \
    -iree-llvm-target-triple=x86_64-pc-linux-elf \
    -iree-llvm-debug-symbols=false \
    iree/samples/models/simple_abs.mlir \
    -o /tmp/simple_abs_dylib.vmfb

```

In which

* `iree-hal-target-backends=dylib-llvm-aot`: Build the model for the dynamic
library CPU HAL driver
* `iree-llvm-link-embedded=true`: Generate the dynamic library with
[LLD](https://lld.llvm.org/) and the artifact can be loaded with the
[embedded library loader](https://github.com/google/iree/blob/main/iree/hal/local/loaders/embedded_library_loader.h)
without invoking the dynamic library support
* `iree-llvm-target-triple`: Use the `<arch>-pc-linux-elf` LLVM target triple so
the artifact has a fixed ABI to be rendered by the
[elf_module library](https://github.com/google/iree/tree/main/iree/hal/local/elf)
* `iree-llvm-debug-symbols=false`: To reduce the artifact size

See [generate.sh](https://github.com/google/iree/blob/main/iree/hal/local/elf/testdata/generate.sh)
for example command-line instructions of some common architectures

You can replace the MLIR file with the other MLIR model files, following the
[instructions](./cpu-dylib.md#compile-the-model)

### Compiling the bare-metal model for static-library support

See the [static_library](https://github.com/google/iree/tree/main/iree/samples/static_library)
demo sample for an example and instructions on running a model with IREE's
`static_library_loader`.

By default, the demo targets the host machine when compiling. To produce a
bare-metal compatible model, run `iree-translate` as in the previous example
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
* `set(IREE_ENABLE_MLIR OFF)`: Disable LLVM/MLIR dependencies for the runtime
library
* `set(CMAKE_SYSTEM_NAME Generic)`: Tell CMake to skip targeting a specific
operating system
* `set(IREE_BINDINGS_TFLITE OFF)`: Disable the TFLite binding support
* `set(IREE_ENABLE_THREADING OFF)`: Disable multi-thread library support
* `set(IREE_HAL_DRIVERS_TO_BUILD "Dylib;VMVX")`: Build only the dynamic library
and VMVX runtime HAL drivers
* `set(IREE_BUILD_TESTS OFF)`: Disable tests until IREE supports running them on
bare-metal platforms
* `set(IREE_BUILD_SAMPLES ON)`: Build
[simple_embedding](https://github.com/google/iree/tree/main/iree/samples/simple_embedding)
example

!!! todo
    Clean the list up after [#6353](https://github.com/google/iree/issues/6353)
    is fixed.

Also, set the toolchain-specific cmake file to match the tool path, target
architecture, target abi, linker script, system library path, etc.

### Define IREE macros

* `-DIREE_PLATFORM_GENERIC`: Let IREE to build the runtime library without
targeting a specific platform
* `-DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=1`: Disable thread synchronization
support
* `-DIREE_FILE_IO_ENABLE=0`: Disable file I/O
* `-DIREE_TIME_NOW_FN`: A function to return the system time. For the bare-metal
system, it can be set as `-DIREE_TIME_NOW_FN=\"\{ return 0;\}\"` as there's no
asynchronous wait handling

Examples of how to setup the CMakeLists.txt and .cmake file:

* [IREE RISC-V toolchain cmake](https://github.com/google/iree/blob/main/build_tools/cmake/riscv.toolchain.cmake)
* [IREE Bare-Metal Arm Sample](https://github.com/iml130/iree-bare-metal-arm)

## Bare-metal execution example

See
[simple_embedding for generic platform](https://github.com/google/iree/blob/main/iree/samples/simple_embedding/README.md#generic-platform-support)
to see how to use the IREE runtime library to build/run the IREE model for the
bare-metal target.
