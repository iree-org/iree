# Out-of-tree compiler plugin example

A dynamic IREE compiler plugin with its own dialect and pass, as a plugin in
another repository would be.

```
src/ootex/IR/OotexOps.td            the ootex dialect, one op: ootex.mark
src/ootex/Transforms/Passes.td      the pass and its `tag` option
src/ootex/Transforms/               AnnotateMarkedFunctions
src/PluginRegistration.cpp          the plugin: dialect, pass, pipeline hook
test/annotate.mlir                  loads the plugin into iree-compile
```

## What it demonstrates

The pass erases every `ootex.mark` and sets `ootex.tag` on the enclosing
`util.func` to a pass option. Its own ops decide what happens to IREE's, as in
a real plugin.

The compiler is given the plugin by path; the plugin reports its id:

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

`--ootex-tag` is an ordinary compiler flag: plugins load before `llvm::cl`
parses.

## Building it

`bazel_to_cmake` generates `CMakeLists.txt` from `BUILD.bazel`.

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

## Build requirements

The compiler renames every `llvm::` and `mlir::` symbol so it can share a
process with another LLVM. A plugin is renamed the same way and resolves against
the compiler, so it must match the compiler's build:

- RTTI and exception settings. A plugin with RTTI against a `-fno-rtti`
  compiler references typeinfo nothing resolves. In-tree builds inherit the
  settings; out-of-tree builds set them by hand.
- The same IREE revision. The API version and header hash catch a changed
  plugin API, not a changed MLIR.
- Under CMake, a host built with `IREE_COMPILER_DYNAMIC_PLUGINS=ON`;
  without it `iree-compile` exports no compiler symbols. Bazel exports them
  either way.
