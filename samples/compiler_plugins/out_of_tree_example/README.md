# Out-of-tree compiler plugin example

A dynamically loaded IREE compiler plugin with its own dialect and its own
pass, shaped the way a plugin living in someone else's repository would be.

```
src/ootex/IR/OotexOps.td            the ootex dialect, one op: ootex.mark
src/ootex/Transforms/Passes.td      the pass and its `tag` option
src/ootex/Transforms/               AnnotateMarkedFunctions
src/PluginRegistration.cpp          the plugin: dialect, pass, pipeline hook
test/annotate.mlir                  loads the plugin into iree-compile
```

## What it demonstrates

The pass reads the out-of-tree `ootex` dialect and writes to IREE's in-tree
`util` dialect: for every `util.func` holding an `ootex.mark`, it erases the
mark and sets `ootex.tag` to the value of a pass option. That is the shape of a
real plugin — its own ops decide what happens to IREE's.

Nothing here is registered at build time. The compiler is told about the plugin
by path, and the plugin reports its own id:

```sh
iree-compile --iree-load-plugin=/path/to/libiree_compiler_plugin_ootex.so \
             --iree-plugin=ootex --ootex-tag=hello \
             --compile-to=preprocessing input.mlir
```

```mlir
util.func private @_marked() attributes {..., ootex.tag = "hello"} {
```

The tag lands on the private function because IREE's ABI pass has already moved
the marked body there.

`--ootex-tag` is an ordinary compiler flag. The plugin registers it before
`llvm::cl` parses, which is why a plugin can add options at all.

## Building it

Both build systems are driven from `BUILD.bazel`; `bazel_to_cmake` generates
`CMakeLists.txt` from it.

```sh
# CMake
cmake -B build -DIREE_COMPILER_DYNAMIC_PLUGINS=ON -DIREE_ENABLE_THIN_ARCHIVES=OFF
ninja -C build iree_compiler_plugin_ootex

# Bazel
bazel build //samples/compiler_plugins/out_of_tree_example:iree_compiler_plugin_ootex
```

One rule does the integration:

```python
iree_compiler_register_dynamic_plugin(
    plugin_id = "ootex",
    target = ":registration",
    compiler = "//lib:IREECompilerShared",
)
```

## What a plugin has to agree with

The compiler renames every `llvm::`/`mlir::` symbol to an IREE-private
spelling so it can share a process with a foreign LLVM. A plugin is renamed the
same way and resolves against the compiler, which means it must match the
compiler's build:

- The host's RTTI and exception settings. A plugin compiled with RTTI against
  a `-fno-rtti` compiler emits typeinfo references nothing resolves. Building
  the plugin in the same tree as the compiler gets this right for free: CMake
  compiles both with `-fno-rtti -fno-exceptions`, Bazel compiles both with
  neither flag. An out-of-tree build has to match its host by hand.
- The same IREE revision. `IREE_COMPILER_PLUGIN_API_VERSION` catches a changed
  entry point, not a changed `Client.h`.
- Under CMake, a host built with `IREE_COMPILER_DYNAMIC_PLUGINS=ON`: without it
  `iree-compile` exports no compiler symbols, and a plugin touching MLIR has
  nothing to bind to. Bazel exports them either way.

## Building against an install tree

This sample lives in the IREE tree and is reached through
`IREE_CMAKE_PLUGIN_PATHS`. A plugin in another repository instead installs IREE
and finds it:

```sh
cmake --install <build> --prefix <prefix> --component IREECMakeExports
cmake --install <build> --prefix <prefix> --component IREEDevLibraries-Compiler
cmake --install <build> --prefix <prefix> --component Compiler
```

```cmake
find_package(IREECompiler REQUIRED)      # -DIREECompiler_DIR=<prefix>/lib/cmake/IREE
find_package(MLIR REQUIRED CONFIG)       # -DMLIR_DIR=..., -DLLVM_DIR=...

add_library(registration STATIC "plugin.cpp")
target_link_libraries(registration PRIVATE iree_compiler_PluginAPI_headers)

iree_compiler_register_dynamic_plugin(
  PLUGIN_ID my_plugin
  TARGET registration
)
```

`find_package(IREECompiler)` brings the plugin headers, the rename script and
`IREE_COMPILER_ABI_PREFIX`. IREE installs no llvm/mlir headers, so MLIR has to
come from wherever the compiler was built from — the build tree serves, as in
`build_tools/testing/test_plugin_from_install.sh`.
