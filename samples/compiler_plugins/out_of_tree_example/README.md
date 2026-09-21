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

The compiler renames every `llvm::` and `mlir::` symbol so it can share a
process with another LLVM. A plugin is renamed the same way and resolves against
the compiler, so it must match the compiler's build:

- Match ABI-affecting settings, including RTTI, exceptions, assertions, and
  the C++ standard library. RTTI mismatches can cause unresolved typeinfo
  symbols. In-tree builds inherit the settings; external builds must match
  them explicitly.
- Use the same IREE and LLVM/MLIR revisions. The API version and header hash
  check only part of the ABI; they do not establish full compatibility.
- Under CMake, a host built with `IREE_EXPERIMENTAL_COMPILER_DYNAMIC_PLUGINS=ON`;
  without it the compiler library does not export the C++ ABI these plugins
  need. Bazel's compiler shared library exports it without a separate gate.
  Both builds require tools linked to the shared compiler library.
