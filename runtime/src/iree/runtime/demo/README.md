# IREE C Runtime API Demo

This demonstrates how to use the higher-level IREE C API to load a compiled
module and call the functions within it.

The module used has a single exported function `@simple_mul` that multiplies two
tensors and returns the result:

```mlir
func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
```

The demo here sets up the shared `iree_runtime_instance_t`, loads the module
into an `iree_runtime_session_t`, and makes a call via `iree_runtime_call_t`.

[`hello_world_terse.c`](hello_world_terse.c) highlights the steps while
[`hello_world_explained.c`](hello_world_explained.c) has more discussion over
what is happening and things to watch out for.

Modules can be loaded from the file system or into memory by the application.
The `iree_runtime_demo_hello_world_file` target shows loading from a file
passed in as a command line argument and
`iree_runtime_demo_hello_world_embedded` shows loading from a blob of memory
where the test file has been built directly into the binary.

NOTE: for brevity the `_terse.c` example uses `IREE_CHECK_OK` to abort the
program on errors. Real applications - especially ones hosting IREE such as
Android apps - would want to follow the patterns in `_explained.c` for how to
propagate errors and clean up allocated resources.
