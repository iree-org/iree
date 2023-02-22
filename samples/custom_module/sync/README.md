# Synchronous tensor I/O custom module sample

This sample expects that you've already produced a working version of the
[basic sample](/samples/custom_module/basic/) (including compiler installation
and CMake setup).

This sample demonstrates adding custom modules callable from compiler-produced
programs that take and return `tensor` types. By default custom calls are
treated as blocking operations that synchronize with the underlying device to
ensure all passed `tensor` buffer views are host coherent and it's assumed that
any returned `tensor` buffer views are ready for use when the call returns.

This approach is the easiest to integrate and looks similar to classic ML
frameworks custom calls. There are many significant performance implications of
using this approach, though, and synchronous calls should only be used when
no asynchronous approach is possible. See the
[async tensor](/samples/custom_module/async/) sample for how to define
custom calls that work asynchronously.

## Instructions

1. Compile the [example module](./test/example.mlir) to a .vmfb file:

    ```
    iree-compile --iree-hal-target-backends=llvm-cpu samples/custom_module/sync/test/example.mlir -o=/tmp/example.vmfb
    ```

2. Build the `iree_samples_custom_module_sync_run` CMake target :

    ```
    cmake -B ../iree-build/ -DCMAKE_BUILD_TYPE=RelWithDebInfo . \
        -DCMAKE_C_FLAGS=-DIREE_VM_EXECUTION_TRACING_FORCE_ENABLE=1
    cmake --build ../iree-build/ --target iree_samples_custom_module_sync_run
    ```
    (here we force runtime execution tracing for demonstration purposes)

    [See here](https://openxla.github.io/iree/building-from-source/getting-started/)
    for general instructions on building using CMake.

3. Run the example program to call the main function:

   ```
   ../iree-build/samples/custom_module/sync/custom-module-sync-run \
       /tmp/example.vmfb example.main
   ```
