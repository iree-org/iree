# Statically-linked custom module sample for IREE tooling

This sample derives from the [basic](./samples/custom_module/basic/) sample to
show how to build a custom C++ module that can be statically linked into the
IREE command line tools. This is only useful for debugging/developing/profiling
custom modules and is not intended as a deployment mechanism. Once a custom
module has been written it can be statically linked into the user's hosting
runtime application or library or dynamically linked and loaded as with the
[dynamic](./samples/custom_module/dynamic/) sample.

The custom module is implemented using a C++ module wrapper layer in
[`module.cc`](./module.cc) and loaded by the `iree-run-module` tool
automatically whenever any user module depends on it.

## Background

IREE's VM is used to dynamically link modules of various types together at
runtime (C, C++, IREE's VM bytecode, etc). Via this mechanism any number of
modules containing exported functions and types that can be used across modules
can extend IREE's base functionality. The IREE tooling (`iree-run-module`,
`iree-benchmark-module`, etc) is _not_ an ML runtime and not intended to be
deployed but it can be useful to test custom modules using them in order to
benchmark, profile, or debug outside of user applications. This sample
demonstrates how to take a generic custom module that's effectively identical to
both [basic](./samples/custom_module/basic/) and
[dynamic](./samples/custom_module/dynamic/) samples and link that into the IREE
tools for those purposes. Note that this is mostly about CMake configuration and
matching some tooling-specific function signatures and it's possible to share
the same module code between all various modes with minor differences.

## Instructions

1. Build or install the `iree-compile` binary:

    ```
    python -m pip install iree-compiler
    ```

    [See here](https://iree.dev/reference/bindings/python/)
    for general instructions on installing the compiler.

3. Compile the [example module](./test/example.mlir) to a .vmfb file:

    ```
    # This simple sample doesn't use tensors and can be compiled in host-only
    # mode to avoid the need for the HAL.
    iree-compile --iree-hal-target-backends=vmvx samples/custom_module/static/test/example.mlir -o=/tmp/example.vmfb
    ```

3. Configure the IREE tools to include the custom module:

    ```
    cmake -B ../iree-build/ -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo . \
        -DCMAKE_C_FLAGS=-DIREE_VM_EXECUTION_TRACING_FORCE_ENABLE=1 \
        -DIREE_EXTERNAL_TOOLING_MODULES=static_sample \
        -DIREE_EXTERNAL_TOOLING_MODULE_STATIC_SAMPLE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/samples/custom_module/static \
        -DIREE_EXTERNAL_TOOLING_MODULE_STATIC_SAMPLE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/samples/custom_module/static \
        -DIREE_EXTERNAL_TOOLING_MODULE_STATIC_SAMPLE_TARGET=iree_samples_custom_module_static_module \
        -DIREE_EXTERNAL_TOOLING_MODULE_STATIC_SAMPLE_NAME=custom \
        -DIREE_EXTERNAL_TOOLING_MODULE_STATIC_SAMPLE_REGISTER_TYPES=register_sample_module_types \
        -DIREE_EXTERNAL_TOOLING_MODULE_STATIC_SAMPLE_CREATE=create_sample_module
    cmake --build ../iree-build/ --target iree-run-module
    ```
    (here we force runtime execution tracing for demonstration purposes)

    [See here](https://iree.dev/building-from-source/getting-started/)
    for general instructions on building using CMake.

4. Run the example program using the main `iree-run-module` tool:

   ```
   ../iree-build/tools/iree-run-module \
      --module=/tmp/example.vmfb \
      --function=main
   ```

### Type registration

Custom types defined by the dynamic module must be registered with the
`iree_vm_instance_t` provided to the creation function.

Any externally defined types such as the builtin VM types
(`!vm.list`/`iree_vm_list_t`) or the HAL types
(`!hal.buffer_view`/`iree_hal_buffer_view_t`) must be resolved from the instance
prior to use. `iree_vm_resolve_builtin_types` and
`iree_hal_module_resolve_all_types` (or one of the more restricted sets like
`iree_hal_module_resolve_common_types`) can be used to perform the resolution.

The tooling has a quirk around dynamic module resolution that requires all types
be registered prior to loading modules. Here that is implemented with the
`register_sample_module_types` function that is provided to CMake via
`-DIREE_EXTERNAL_TOOLING_MODULE_STATIC_SAMPLE_REGISTER_TYPES=` to ensure that
the sample types get registered first. In a real application that manually
creates modules this would not be required as modules are expected to manage
type registration alongside their lifetime.
