# CMake Options and Variables

## Frequently-used CMake Variables

#### `CMAKE_BUILD_TYPE`:STRING

Sets the build type. Possible values are `Release`, `Debug`,
`RelWithDebInfo`/`FastBuild` and `MinSizeRel`. If unset, build type is set to
`Release`.

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

#### `IREE_BUILD_DEBUGGER`:BOOL

Builds the IREE debugger app. Defaults to `OFF`.

#### `IREE_BUILD_PYTHON_BINDINGS`:BOOL

Builds the IREE python bindings. Defaults to `OFF`.

#### `IREE_BUILD_EXPERIMENTAL`:BOOL

Builds experimental projects. Defaults to `OFF`.

#### `IREE_HAL_DRIVERS_TO_BUILD`:STRING

*This does not have any effect at the moment, but will be supported in the
future!* Semicolon-separated list of HAL drivers to build, or `all` for building
all HAL drivers. Case-insensitive. Defaults to `all`. Example:
`-DIREE_HAL_DRIVERS_TO_BUILD="Vulkan;VMLA"`.

#### `IREE_TARGET_BACKENDS_TO_BUILD`:STRING

*This does not have any effect at the moment, but will be supported in the
future!* Semicolon-separated list of HAL drivers to build, or `all` for building
all HAL drivers. Case-insensitive. Defaults to `all`. Example:
`-DIREE_HAL_DRIVERS_TO_BUILD="Vulkan_SPIRV;VMLA"`.

#### `IREE_ENABLE_LLD`:BOOL

Use lld when linking. Defaults to `OFF`. This option is equivalent to
`-DIREE_USE_LINKER=lld`. The option `IREE_ENABLE_LLD` and `IREE_USE_LINKER` can
not be set at the same time.

#### `IREE_MLIR_DEP_MODE`:STRING

Defines the MLIR dependency mode. Case-sensitive. Can be `BUNDLED`, `DISABLED`
or `INSTALLED`. Defaults to `BUNDLED`. If set to `INSTALLED`, the variable
`MLIR_DIR` needs to be passed and that LLVM needs to be compiled with
`LLVM_ENABLE_RTTI` set to `ON`.

## MLIR-specific CMake Options and Variables

#### `MLIR_DIR`:STRING

Specifies the path where to look for the installed MLIR/LLVM packages. Required
if `IREE_MLIR_DEP_MODE` is set to `INSTALLED`.

## Cross-compilation

[TODO(#2111): The following explanation is developer oriented. Move it to the
developer build doc once we've created that.]

Cross-compilation involves both a *host* platform and a *target* platform. One
invokes compiler toolchains on the host platform to generate libraries and
executables that can be run on the target platform.

IREE uses tools to programmatically generate C/C++ source code from some
domain-specific descriptions. For example, `flatc` is used to generate C/C++
code from FlatBuffer schemas. These tools should be compiled for the host
platform so that we can invoke them during build process. This requires
cross-compilation for IREE to (conceptually) happen in two stages: first compile
build tools under host platform, and then use these host tools together with
cross-compiling toolchains to generate artifacts for the target platform. (The
build system dependency graph may not have such clear two-stage separation.)

CMake cannot handle multiple compiler toolchains in one CMake invocation. So the
above conceptual two-stage compilation happens in two separate CMake
invocations.

When CMake is invoked with a toolchain file, e.g.,
[`android.toolchain.cmake`](https://android.googlesource.com/platform/ndk/+/master/build/cmake/android.toolchain.cmake),
we get the compiler toolchain for the target platform from the toolchain file.
IREE under the hood will
[create another CMake invocation](https://github.com/google/iree/blob/main/build_tools/cmake/iree_cross_compile.cmake)
using host platform's compiler toolchain. The inner CMake invocation will be
placed under `IREE_HOST_BINARY_ROOT`.

The tricky part is to set up target artifacts' dependencies on host executables.
Normally in CMake we define targets and `target_link_libraries` to express the
dependency relationship between targets. The optimal way is to
[`export`](https://cmake.org/cmake/help/latest/command/export.html) targets
under host configuration and then
[`import`](https://cmake.org/cmake/help/latest/command/add_executable.html?highlight=import#imported-executables)
into target configuration. But that is not playing well with LLVM's TableGen
configurations. LLVM's cross-compilation uses on files for dependencies. So here
we need to do the same. This results in reverting the relationship between IREE
artifact output names (e.g., `iree-translate`) and IREE package-prefixed CMake
target names (e.g., `iree_tools_iree-translate`).

When not cross-compiling, IREE uses the CMake target names as the real unique
identification inside CMake for defining the object and expressing dependencies;
the artifact output names are just used for naming the build artifacts. When
cross-compiling, it's the artifact names being load-bearing. The artifact names
are used to express dependencies across CMake invocation boundary (remember that
we cannot access targets defined in another CMake invocation); the
package-prefixed CMake target names are just custom targets depending on the
host artifact.

#### `IREE_HOST_BINARY_ROOT`:FILEPATH

Specifies the root directory for containing all host CMake invocation artifacts.
This defaults to `CMAKE_BINARY_DIR/host` if missing.

#### `IREE_HOST_C_COMPILER`:STRING

Specifies the C compiler for host compilation.

#### `IREE_HOST_CXX_COMPILER`:STRING

Specifies the C++ compiler for host compilation.

#### `IREE_HOST_<option>`:BOOL

For each option described in "IREE-specific CMake Options and Variables", you
can use the `IREE_HOST_<option>` counterpart to control the feature when
compiling under host configuration. For example, `IREE_HOST_BUILD_TESTS` will
enables all tests for the host configuration.
