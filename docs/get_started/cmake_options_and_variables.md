---
layout: default
permalink: get-started/cmake-options-and-variables
title: "CMake Options and Variables"
nav_order: 13
parent: Getting Started
---

# CMake Options and Variables
{: .no_toc }

## Frequently-used CMake Variables

#### `CMAKE_BUILD_TYPE`:STRING

Sets the build type. Possible values are `Release`, `Debug`,
`RelWithDebInfo` and `MinSizeRel`. If unset, build type is set to `Release`.

#### `CMAKE_<LANG>_COMPILER`:STRING

This is the command that will be used as the `<LANG>` compiler, which are `C`
and `CXX` in IREE. These variables are set to compile IREE with `clang` or
rather `clang++`. Once set, these variables can not be changed.

## IREE-specific CMake Options and Variables

This gives a brief explanation of IREE specific CMake options and variables.

#### `IREE_ENABLE_RUNTIME_TRACING`:BOOL

Enables instrumented runtime tracing. Defaults to `OFF`.

#### `IREE_ENABLE_MLIR`:BOOL

Enables MLIR/LLVM dependencies. Defaults to `ON`. MLIR/LLVM dependencies are
required when building the IREE compiler components. Therefore, the option is
automatically set to `ON` if `IREE_BUILD_COMPILER` is set to `ON`.

#### `IREE_ENABLE_EMITC`:BOOL

Enables the build of the out-of-tree MLIR dialect EmitC. Defaults to `OFF`. To
build the EmitC dialect, `IREE_ENABLE_MLIR` must be set to `ON`.

#### `IREE_BUILD_COMPILER`:BOOL

Builds the IREE compiler. Defaults to `ON`.

#### `IREE_BUILD_TESTS`:BOOL

Builds IREE unit tests. Defaults to `ON`.

#### `IREE_BUILD_DOCS`:BOOL

Builds IREE documentation. Defaults to `OFF`.

#### `IREE_BUILD_SAMPLES`:BOOL

Builds IREE sample projects. Defaults to `ON`.

#### `IREE_BUILD_PYTHON_BINDINGS`:BOOL

Builds the IREE python bindings. Defaults to `OFF`.

#### `IREE_BUILD_EXPERIMENTAL_JAVA_BINDINGS`:BOOL

Builds the experimental java bindings. Defaults to `OFF`.

#### `IREE_BUILD_EXPERIMENTAL_MODEL_BUILDER`:BOOL

Builds the experimental model builder component. Defaults to `OFF`.

#### `IREE_BUILD_EXPERIMENTAL_REMOTING`:BOOL

Builds experimental remoting component. Defaults to `OFF`.

#### `IREE_HAL_DRIVERS_TO_BUILD`:STRING

Semicolon-separated list of HAL drivers to build, or `all` for building all HAL
drivers. Case-insensitive. If an empty list is provided, will build no HAL
drivers. Defaults to `all`. Example: `-DIREE_HAL_DRIVERS_TO_BUILD=Vulkan;VMLA`.

#### `IREE_TARGET_BACKENDS_TO_BUILD`:STRING

Semicolon-separated list of target backend to build, or `all` for building all
compiler target backends. Case-insensitive. If an empty list is provided, will
build no target backends. Defaults to `all`. Example:
`-DIREE_TARGET_BACKENDS_TO_BUILD=Vulkan-SPIRV;VMLA`.

#### `IREE_ENABLE_LLD`:BOOL

Use lld when linking. Defaults to `OFF`. This option is equivalent to
`-DIREE_USE_LINKER=lld`. The option `IREE_ENABLE_LLD` and `IREE_USE_LINKER` can
not be set at the same time.

#### `IREE_ENABLE_ASAN`:BOOL

Enable [address sanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) if
the current build type is Debug and the compiler supports it.

#### `IREE_ENABLE_MSAN`:BOOL

Enable [memory sanitizer](https://clang.llvm.org/docs/MemorySanitizer.html) if
the current build type is Debug and the compiler supports it.

#### `IREE_ENABLE_TSAN`:BOOL

Enable [thread sanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) if
the current build type is Debug and the compiler supports it.

#### `IREE_MLIR_DEP_MODE`:STRING

Defines the MLIR dependency mode. Case-sensitive. Can be `BUNDLED`, `DISABLED`
or `INSTALLED`. Defaults to `BUNDLED`. If set to `INSTALLED`, the variable
`MLIR_DIR` needs to be passed and that LLVM needs to be compiled with
`LLVM_ENABLE_RTTI` set to `ON`.

#### `IREE_BUILD_TENSORFLOW_COMPILER`:BOOL

Enables building of the TensorFlow to IREE compiler under
`integrations/tensorflow`, including some native binaries and Python packages.
Note that TensorFlow's build system is bazel and this will require having
previously built (or installed) the iree-tf-import at the path specified by
`IREE_TF_TOOLS_ROOT`.

#### `IREE_BUILD_TFLITE_COMPILER`:BOOL

Enables building of the TFLite to IREE compiler under `integrations/tensorflow`,
including some native binaries and Python packages. Note that TensorFlow's build
system is bazel and this will require having previously built (or installed) the
iree-tf-import at the path specified by `IREE_TF_TOOLS_ROOT`.

#### `IREE_BUILD_XLA_COMPILER`:BOOL

Enables building of the XLA to IREE compiler under `integrations/tensorflow`,
including some native binaries and Python packages. Note that TensorFlow's build
system is bazel and this will require having previously built (or installed) the
iree-tf-import at the path specified by `IREE_TF_TOOLS_ROOT`.

#### `IREE_TF_TOOLS_ROOT`:STRING

Path to prebuilt TensorFlow integration binaries to be used by the Python
bindings. Defaults to
"${CMAKE_SOURCE_DIR}/integrations/tensorflow/bazel-bin/iree_tf_compiler", which
is where they would be placed by a `bazel build` invocation.

## MLIR-specific CMake Options and Variables

#### `MLIR_DIR`:STRING

Specifies the path where to look for the installed MLIR/LLVM packages. Required
if `IREE_MLIR_DEP_MODE` is set to `INSTALLED`.

## Cross-compilation

When cross compiling (using a toolchain file like
[`android.toolchain.cmake`](https://android.googlesource.com/platform/ndk/+/master/build/cmake/android.toolchain.cmake)),
first build and install IREE's tools for your host configuration, then use the
`IREE_HOST_BINARY_ROOT` CMake option to point the cross compiled build at the
host tools.