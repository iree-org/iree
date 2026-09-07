# Out-of-tree compiler plugin example

An experimental dynamic IREE compiler plugin with its own dialect and pass.
The sample is built within IREE and demonstrates how to implement a plugin
that could be maintained in another repository.

```
src/ootex/IR/OotexOps.td            the ootex dialect, one op: ootex.mark
src/ootex/Transforms/Passes.td      the pass and its `tag` option
src/ootex/Transforms/               AnnotateMarkedFunctions
src/PluginRegistration.cpp          the plugin: dialect, pass, pipeline hook
test/annotate.mlir                  loads the plugin into iree-compile
```

## What it demonstrates

The pass erases `ootex.mark` operations directly inside a `util.func` and sets
`ootex.tag` on that function to the pass's `tag` option. Marks nested in other
operations' regions are not handled. This demonstrates how a plugin's dialect
and preprocessing pass can annotate IREE operations.

The compiler is given the plugin by path; the plugin reports its id:

```sh
iree-compile --iree-load-plugin=/path/to/libiree_compiler_plugin_ootex.so \
             --iree-plugin=ootex --ootex-tag=hello \
             --compile-to=preprocessing input.mlir
```

```mlir
util.func private @_marked() attributes {..., ootex.tag = "hello"} {
```

For `test/annotate.mlir`, the tag lands on the private function because IREE's
ABI pass has already moved the marked body there.

`--ootex-tag` is an ordinary compiler flag: plugins load before `llvm::cl`
parses.

## Building it

Run these commands from the IREE repository root. CMake includes this sample
through its built-in plugin paths when `IREE_BUILD_SAMPLES=ON`.
`bazel_to_cmake` generates `CMakeLists.txt` from `BUILD.bazel`.

```sh
# CMake
cmake -S . -B build -DIREE_BUILD_SAMPLES=ON \
  -DIREE_EXPERIMENTAL_COMPILER_DYNAMIC_PLUGINS=ON -DIREE_ENABLE_THIN_ARCHIVES=OFF
cmake --build build --target iree-compile iree_compiler_plugin_ootex

# Bazel
bazel build //tools:iree-compile \
  //samples/compiler_plugins/out_of_tree_example:iree_compiler_plugin_ootex
```

The Bazel rule packages the registration library and its dependencies:

```python
load("//build_tools/bazel:renamed_link.bzl", "iree_compiler_register_experimental_dynamic_plugin")

iree_compiler_register_experimental_dynamic_plugin(
    plugin_id = "ootex",
    target = ":registration",
    compiler = "//lib:IREECompilerShared",
)
```

## Build requirements

Dynamic plugins have no stable API or ABI. Use the host compiler's exact IREE
and LLVM/MLIR revisions, generated headers, toolchain, and ABI-affecting build
settings. Match the architecture, C++ standard library and ABI options, RTTI,
exceptions, assertions (`NDEBUG`), LLVM ABI-breaking checks, and sanitizer
configuration. See the [build requirements and compatibility checks](../../../compiler/src/iree/compiler/PluginAPI/README.md#build-requirements)
for details. The loader cannot verify all of these settings.

Use `IREE_DEFINE_COMPILER_PLUGIN` to supply compatibility metadata automatically.
Its version argument identifies the plugin's release, not API or ABI compatibility.
Do not edit the generated hash or symbol prefix to make
a plugin load; rebuild it against the matching compiler.

CMake requires a host built with
`IREE_EXPERIMENTAL_COMPILER_DYNAMIC_PLUGINS=ON`. Bazel exports compiler symbols
without a separate gate. Both builds require tools linked to the shared compiler
library and plugin archives built with position-independent code.

## Building against an install tree

To build a plugin in another repository, first install a compiler built with
dynamic plugin support:

```sh
cmake --install <build> --prefix <prefix> --component IREECMakeExports
cmake --install <build> --prefix <prefix> --component IREEDevLibraries-Compiler
cmake --install <build> --prefix <prefix> --component Compiler
```

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_iree_plugin LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(IREECompiler REQUIRED)
find_package(MLIR REQUIRED CONFIG)

add_library(registration STATIC "plugin.cpp")
set_target_properties(registration PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(registration PRIVATE iree_compiler_PluginAPI_build_options)
target_include_directories(registration PRIVATE
  ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

iree_compiler_register_dynamic_plugin(
  PLUGIN_ID my_plugin
  TARGET registration
)
```

`find_package(IREECompiler)` brings the plugin headers, the rename script and
`IREE_COMPILER_ABI_PREFIX`. `plugin.cpp` must define the `my_plugin` entry point
with `IREE_DEFINE_COMPILER_PLUGIN` and register the corresponding session.
IREE does not install LLVM/MLIR C++ headers. Point CMake at the packages from
the host compiler's build:

```sh
cmake -S <plugin-source> -B <plugin-build> \
  -DIREECompiler_DIR=<prefix>/lib/cmake/IREE \
  -DMLIR_DIR=<iree-build>/lib/cmake/mlir \
  -DLLVM_DIR=<iree-build>/llvm-project/lib/cmake/llvm
cmake --build <plugin-build> --target iree_compiler_plugin_my_plugin
```

The `iree_compiler_PluginAPI_build_options` target supplies the installed host's
configured C++ flags, assertion state, RTTI and exception options, and headers.
Link it to each plugin helper library as well. These options apply only to the
targets that use it. Use a compatible compiler and the host's toolchain settings;
the SDK does not select a compiler or sysroot for the enclosing project.

If the plugin has additional static libraries, list their archive paths under `EXTRA_ARCHIVES`,
for example `$<TARGET_FILE:helper>`; the installed rule does not collect
transitive dependencies automatically.

See the complete [install-tree test project](../../../build_tools/testing/plugin_from_install/CMakeLists.txt)
and [test script](../../../build_tools/testing/test_plugin_from_install.sh),
which build, activate, and incrementally rebuild a plugin against an install.
