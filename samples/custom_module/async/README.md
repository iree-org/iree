# Asynchronous tensor I/O custom module sample

This sample expects that you've already produced a working version of the
[basic sample](/samples/custom_module/basic/) (including compiler installation
and CMake setup).

This sample demonstrates adding custom modules callable from compiler-produced
programs that take and return `tensor` types. Both the calls into the compiled
program and the custom call made from the compiled program are made
asynchronously using HAL fences for ordering work. This allows the entire
invocation - including the custom user call - to be scheduled without blocking
and enables pipelining and overlapping invocations. When embedded into a larger
user-level framework this lets IREE invocations be interleaved with other user
work.

## Instructions

1. Compile the [example module](./test/example.mlir) to a .vmfb file:

    ```
    iree-compile \
        --iree-execution-model=async-external \
        --iree-hal-target-backends=llvm-cpu \
        samples/custom_module/async/test/example.mlir \
        -o=/tmp/example.vmfb
    ```

2. Build the `iree_samples_custom_module_async_run` CMake target :

    ```
    cmake -B ../iree-build/ -DCMAKE_BUILD_TYPE=RelWithDebInfo . \
        -DCMAKE_C_FLAGS=-DIREE_VM_EXECUTION_TRACING_FORCE_ENABLE=1
    cmake --build ../iree-build/ --target iree_samples_custom_module_async_run
    ```
    (here we force runtime execution tracing for demonstration purposes)

    [See here](https://iree.dev/building-from-source/getting-started/)
    for general instructions on building using CMake.

3. Run the example program to call the main function:

   ```
   ../iree-build/samples/custom_module/async/custom-module-async-run \
       /tmp/example.vmfb example.main
   ```

## TBD

* Expose a way to tie call arguments and results for in-place operations.
* Expose a way to specify the lifetime of the I/O to allow for transient memory.
