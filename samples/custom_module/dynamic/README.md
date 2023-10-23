# Dynamically loaded custom module sample

This sample derives from the [basic](./samples/custom_module/basic/) sample to
show how to build a custom C++ module that can be dynamically loaded by the IREE
runtime.

The custom module is implemented using a C++ module wrapper layer in
[`module.cc`](./module.cc) and loaded by the `iree-run-module` tool using the
`--module=` flag. Any IREE tool can use the module and user applications hosting
the IREE runtime can programmatically load the module using
`iree_vm_dynamic_module_load_from_file`.

## Background

IREE's VM is used to dynamically link modules of various types together at
runtime (C, C++, IREE's VM bytecode, etc). Via this mechanism any number of
modules containing exported functions and types that can be used across modules
can extend IREE's base functionality. When possible it's preferred to statically
link modules to allow for smaller binaries and a better development experience
but it can be desirable to inject custom user modules into pre-built IREE
release tools or extend user applications with hermetically built modules from
others without fully rebuilding/synchronizing versions. It's easy to fall into
[dependency hell](https://en.wikipedia.org/wiki/Dependency_hell) and usage of
dynamic modules should be carefully considered.

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
    iree-compile --iree-hal-target-backends=vmvx samples/custom_module/dynamic/test/example.mlir -o=/tmp/example.vmfb
    ```

3. Build the `iree_samples_custom_module_dynamic_module` CMake target :

    ```
    cmake -B ../iree-build/ -DCMAKE_BUILD_TYPE=RelWithDebInfo . \
        -DCMAKE_C_FLAGS=-DIREE_VM_EXECUTION_TRACING_FORCE_ENABLE=1
    cmake --build ../iree-build/ \
        --target iree-run-module \
        --target iree_samples_custom_module_dynamic_module
    ```
    (here we force runtime execution tracing for demonstration purposes)

    [See here](https://iree.dev/building-from-source/getting-started/)
    for general instructions on building using CMake.

4. Run the example program using the main `iree-run-module` tool:

   ```
   ../iree-build/tools/iree-run-module \
      --module=../iree-build/samples/custom_module/dynamic/module.so@create_custom_module \
      --module=/tmp/example.vmfb \
      --function=main
   ```

## Limitations

Currently tracing with Tracy is not supported in dynamic modules. The hosting
application can be built with tracing enabled but the module shared libraries
must not have Tracy linked in.

## Exporting dynamic modules

Custom dynamic modules are identical to static ones as seen in the other samples
but contain a single exported function that can be used by the runtime to
instantiate the module and provide it with parameters.

By default the name of the creation function is `iree_vm_dynamic_module_create`
but users can override this to allow a single shared library to provide multiple
modules. In this sample the custom name is `create_custom_module` and the tools
are instructed to call that function via the `shared_library.so@fn_name` syntax.

### Versioning

The creation function must check the incoming `max_version` to ensure that the
dynamic module API is compatible. Today the API is unstable and it's expected
that dynamic modules are rebuilt alongside new runtime versions.

### Parameters

Module creation functions are provided with a list of key value parameters
either programmatically provided to `iree_vm_dynamic_module_load_from_file` or
passed as flags. To pass the flags use a URI query string:

```
iree-run-module --module=library.so?key0=value0&key1=value1
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
