IREE CPU Device Library: `libdevice`
====================================

This library provides builtin functions to the IREE generated CPU code. It
covers the role of a compiler runtime library handling things like soft float
builtin calls produced during code generation and a support library to ease
implementation of more complex intrinsic-like functionality. The code in this
library is compiled into bitcode files and embedded inside the IREE compiler
which then links it into the generated code before emitting the final user
output.

```
+------------+      +-------+      +-------------------------------+
| device_*.c | ---> | clang | ---> |+-------------------------------+
+------------+      +-------+      +| libdevice_[arch]_[variant].bc |
                                    +-------------------------------+
                                                  |||
                                                  vvv
      +------------+      +---------+      +================+
      | input.mlir | ---> | codegen | ---> | iree-compile   |
      +------------+      +---------+      +================+
                                                   |
                      +----------------------------+
                      v                            v
         +------------------------+   +----------------------------+
         | static library (.o/.a) |   | dynamic library (.so/.dll) |
         +------------------------+   +----------------------------+
```

Good examples of things this library can provide:
* float16/half support functions
* MMA-like intrinsics for architecture-optimized tiled matrix multiplies
* Atomic intrinsics

Bad examples:
* A full convolution kernel
* Anything only used in only one particular configuration or target
* Frequently changing code

### Why Not C++ Passes?

This approach of an external library that is linked in via bitcode is a tradeoff
that favors a familiar environment for architecture-specific implementations and
reusable code to custom MLIR passes that directly construct the IR. It will
always be better from a technical standpoint to directly perform these
specializations inside compiler passes as all information is available, multiple
levels of optimization at MLIR `vector` and `llvm` dialect levels can hoist and
fold aggressively, and specialization is possible using the entire context. It's
encouraged that work is done there when possible and some of the cases handled
by this library may end up being done in that environment.

As a reusable library this approach allows for other backends - such as the IREE
VMVX backend - to share the same optimized implementations. Having standalone
tests and benchmarks also allows for fast iteration without needing to modify
the compiler.

The hope is that over time things added here will be moved into the compiler and
this becomes mostly a lightweight intrinsics library and staging ground for
experimental features that require quick iteration in C.

## Bitcode Files

The IREE compiler embeds bitcode files and when producing executable libraries
will select one for linkage based on the specified target machine. As these
bitcode files can only be produced by a cross-compilation-enabled Clang they are
built offline and checked into the repository. Future improvements to the
compiler could also allow for external files to be specified to avoid the need
to rebuild the compiler however for now this keeps things simple and hermetic.

The naming convention is `libdevice_[arch]_[features].bc`, corresponding to the
source files of `device_[arch].c` with the features specifying conditional
target CPU features such as extended instruction sets. When no special features
are required `generic` is used.

For example, the implementations for all ISA variants of AArch64 would be found
in a `device_aarch64.c` and an implementation for the baseline ISA
is compiled into `libdevice_aarch64_generic.bc`. When the dot product
instructions are available (`-march=armv8.2-a+dotprod`) the more specialized
`libdevice_aarch64_dotprod.bc` bitcode file would be used.

### Updating Bitcode Files

The bitcode files need to be rebuilt whenever the source is modified, new
variants are added, or new architectures are targeted. The
[`bin/build.sh`](bin/build.sh) uses a compatible Clang and LLVM toolchain to
produce the files in the correct format and location.

Requirements:
* A modern version of Clang/LLVM (tested with 13)
* A build of llvm-as with all target architectures linked in

This script could use some usability improvements, but for now a common
invocation will look like:
```sh
LLVM_AS=/usr/bin/llvm-as \
CLANG=/usr/bin/clang-13 \
./iree/builtins/device/bin/build.sh
```

If there are complaints that llvm-as does not support a target architecture then
the llvm-as included in the IREE CMake distribution should be built and provided
by way of the `IREE_BUILD_DIR`:
```sh
IREE_BUILD_DIR=../iree-build \
CLANG=/usr/bin/clang-13 \
./iree/builtins/device/bin/build.sh
```

After this the newly updated/added bitcode files can be added to git.

### Compiler Bitcode Selection

The logic in the compiler for selecting which bitcode file to use is found in
[`iree/compiler/Dialect/HAL/Target/LLVMCPU/Builtins/Device.cpp`](/iree/compiler/Dialect/HAL/Target/LLVMCPU/Builtins/Device.cpp).
The `lookupDeviceFile` function uses the `llvm::TargetMachine` to query the
architecture, CPU features, and other properties to choose the corresponding
bitcode file. If no matching bitcode file is found a fallback of the WebAssembly
generic implementation is used as its bitcode is generally portable. It's not
fast, though, and should only be used for correctness testing during bringup.

### Adding an Architecture/ISA Bitcode File

First copy [`device_generic.c`](device_generic.c) and name it consistent with
the canonical LLVM architecture (the first part of the target triple, e.g. if
you pass `--target=aarch64-arm-none-eabi` to Clang you'd name it `aarch64`).

From there guard the new file with the architecture-specific preprocessor guards
and add the inverse to `device_generic.c` to prevent it from being used when the
source files are globbed.

To build the new bitcode file add a `make_arch_bc` call to [`bin/build.sh`](bin/build.sh).
The flags provided are passed directly to Clang and can be used to control the
compilation environment with the requirement being that the corresponding
selection logic is updated in `Device.cpp`.

Finally update the [`iree/compiler/Dialect/HAL/Target/LLVMCPU/Builtins/Device.cpp`](/iree/compiler/Dialect/HAL/Target/LLVMCPU/Builtins/Device.cpp)
file in the compiler to select the new bitcode file based on the
`llvm::TargetMachine` in the same way that it is produced with `make_arch_bc`.

Ergonomic improvements here would allow for function-level multi-versioning such
that bitcode files per architecture could be used instead of requiring
per-feature variants of each bitcode file.

## Engineering Requirements

As this library is directly merged into the compiler-generated code there are
specific restrictions as to what can be used inherited from the IREE executable
requirements:

* No mutable globals/static variables or thread-local storage
* No syscalls
* No libc calls outside of builtins (like memset/memcpy) - _no mallocs_!

Though the primary usage of the library is through the precompiled bitcode files
that only need to work with Clang the library may also be built on other
toolchains such as GCC and MSVC (or older version of Clang). When standard
intrinsics are used this will generally not be a problem however inline assembly
may need compiler-specific variants or at least exclusions that fall back to
generic paths.

### Compile-time Configuration

Preprocessor statements used to control behavior must only use information known
when the bitcode files are being compiled. This means that if the bitcode file
being produced is for AArch64 it is safe to use the `__aarch64__` macro.
Information that is only available after the bitcode file is produced - such as
in the IREE compiler pipelines - must use link-time configuration.

### Link-time Configuration

As we are producing bitcode files we cannot rely on the C preprocessor for
changing behavior based on some information only known during linking. In other
cases we may want to specialize code paths based on knowledge about the context
in which the kernels are used. To provide this link-time modification ability
there is support for flags by way of `extern` globals. These globals are either
specified by the IREE compiler when linking the bitcode or by the hosting
application when linked statically.

Each flag is defined in `device.h`; for example:
```c
extern int libdevice_platform_example_flag;
```

Any code may then use this flag to condition/control behavior:
```c
if (libdevice_platform_example_flag >= 1) {
  // Do something special.
}
```

When linking libdevice statically the flags can be provided by the hosting
application via compiler defines: `-DLIBDEVICE_PLATFORM_EXAMPLE_FLAG=123`.

When producing bitcode the flags are left symbolic and the IREE compiler
provides their values:
```c++
overridePlatformGlobal(*bitcodeModule, "libdevice_platform_example_flag", 123u);
```

What flags are useful and how to handle cases where flags are arch-dependent are
still TBD.

## Testing and Benchmarking

[`tools/libdevice_test.cc`](tools/libdevice_test.cc) provides a gtest runner
that compares the results of the optimized implementations for the target
architecture against a reference implementation for correctness.

[`tools/libdevice_benchmark.c`](tools/libdevice_benchmark.c) provides a
benchmark suite for the optimized implementations of the target architecture.

Both are compiled for the CMake target and can be used to develop
implementations without the need to rebuild/run the compiler.
