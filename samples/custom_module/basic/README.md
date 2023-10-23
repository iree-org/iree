# Basic custom module sample

This sample shows how to

1. Create a custom module in C++ that can be used with the IREE runtime
2. Author an MLIR input that uses a custom module including a custom type
3. Compile that program to an IREE VM bytecode module
4. Load the compiled program using a low-level VM interface
5. Call exported functions on the loaded program to exercise the custom module

The custom module is declared in [`module.h`](./module.h), implemented using a
C++ module wrapper layer in [`module.cc`](./module.cc), and called by example in
[`main.c`](./main.c).

This document uses terminology that can be found in the documentation of
[IREE's execution model](https://github.com/openxla/iree/blob/main/docs/developers/design_docs/execution_model.md).
See [IREE's extensibility mechanisms](https://openxla.github.io/iree/reference/extensions/)
documentation for more information specific to extenting IREE and
alternative approaches to doing so.

## Background

IREE's VM is used to dynamically link modules of various types together at
runtime (C, C++, IREE's VM bytecode, etc). Via this mechanism any number of
modules containing exported functions and types that can be used across modules
can extend IREE's base functionality. In most IREE programs the HAL module is
used to provide a hardware abstraction layer for execution and both the HAL
module itself and the types it exposes (`!hal.buffer`, `!hal.executable`, etc)
are implemented using this mechanism.

## Instructions

1. Build or install the `iree-compile` binary:

    ```
    python -m pip install iree-compiler
    ```

    [See here](https://openxla.github.io/iree/reference/bindings/python/)
    for general instructions on installing the compiler.

3. Compile the [example module](./test/example.mlir) to a .vmfb file:

    ```
    # This simple sample doesn't use tensors and can be compiled in host-only
    # mode to avoid the need for the HAL.
    iree-compile --iree-execution-model=host-only samples/custom_module/basic/test/example.mlir -o=/tmp/example.vmfb
    ```

3. Build the `iree_samples_custom_module_run` CMake target :

    ```
    cmake -B ../iree-build/ -DCMAKE_BUILD_TYPE=RelWithDebInfo . \
        -DCMAKE_C_FLAGS=-DIREE_VM_EXECUTION_TRACING_FORCE_ENABLE=1
    cmake --build ../iree-build/ --target iree_samples_custom_module_basic_run
    ```
    (here we force runtime execution tracing for demonstration purposes)

    [See here](https://openxla.github.io/iree/building-from-source/getting-started/)
    for general instructions on building using CMake.

4. Run the example program to call the main function:

   ```
   ../iree-build/samples/custom_module/basic/custom-module-basic-run \
       /tmp/example.vmfb example.main
   ```

## Defining Custom Modules in C++

Modules are exposed to applications and the IREE VM via the `iree_vm_module_t`
interface. IREE canonically uses C headers to expose module and type functions
but the implementation of the module can be anything the user is able to work
with (C, C++, rust, etc).

A C++ wrapper is provided to ease implementation when minimal code size and overhead is not a focus and provides easy definition of exports and marshaling
of types. Utilities such as `iree::Status` and `iree::vm::ref<T>` add safety for
managing reference counted resources and can be used within the modules.

General flow:

1. Expose module via a C API ([`module.h`](./module.h)):

```c
// Ideally all allocations performed by the module should use |allocator|.
// The returned module in |out_module| should have a ref count of 1 to transfer
// ownership to the caller.
iree_status_t iree_table_module_create(iree_allocator_t allocator,
                                       iree_vm_module_t** out_module);
```

2. Implement the module using C/C++/etc ([`module.cc`](./module.cc)):

Modules have two parts: a shared module and instantiated state.

The `iree::vm::NativeModule` helper is used to handle the shared module
declaration and acts as a factory for per-context instantiated state and the
methods exported by the module:

```c++
// Any mutable state stored on the module may be accessed from multiple threads
// if the module is instantiated in multiple contexts and must be thread-safe.
struct TableModule final : public vm::NativeModule<TableModuleState> {
  // Each time the module is instantiated this will be called to allocate the
  // context-specific state. The returned state must only be thread-compatible
  // as invocations within a context will not be made from multiple threads but
  // the thread on which they are made may change over time; this means no TLS!
  StatusOr<std::unique_ptr<TableModuleState>> CreateState(
      iree_allocator_t allocator) override;
};
```

The module implementation is done on the state object so that methods may use
`this` to access context-local state:

```c++
struct TableModuleState final {
  // Local to the context the module was instantiated in and thread-compatible.
  std::unordered_map<std::string, std::string> mutable_state;

  // Exported functions must return Status or StatusOr. Failures will result in
  // program termination and will be propagated up to the top-level invoker.
  // If a module wants to provide non-fatal errors it can return results to the
  // program: here we return a 0/1 indicating whether the key was found as well
  // as the result or null.
  //
  // MLIR declaration:
  //   func.func private @table.lookup(!util.buffer) -> (i1, !util.buffer)
  StatusOr<std::tuple<int32_t, vm::ref<iree_vm_buffer_t>>> Lookup(
      const vm::ref<iree_vm_buffer_t> key);
};
```

Finally the exported methods are registered and marshaling code is expanded:

```c++
static const vm::NativeFunction<TableModuleState> kTableModuleFunctions[] = {
    vm::MakeNativeFunction("lookup", &TableModuleState::Lookup),
};
extern "C" iree_status_t iree_table_module_create(
    iree_allocator_t allocator, iree_vm_module_t** out_module) {
  auto module = std::make_unique<TableModule>(
      "table", /*version=*/0, allocator,
      iree::span<const vm::NativeFunction<CustomModuleState>>
      (kTableModuleFunctions));
  *out_module = module.release()->interface();
  return iree_ok_status();
}
```

## Registering Custom Modules at Runtime

Once a custom module is defined it needs to be provided to any context that it
is going to be used in. Each context may have its own unique mix of modules and
it's the hosting application's responsibility to inject the available modules.
See [`main.c`](./main.c) for an example showing the entire end-to-end lifetime
of loading a compiled bytecode module and providing a custom module for runtime
dynamic linking.

Since modules themselves can be reused across contexts it can be a way of
creating shared caches (requires thread-safety!) that span contexts while the
module state is context specific and isolated.

Import resolution happens in reverse registration order: the most recently
registered modules override previous ones. This combined with optional imports
allows overriding behavior and version compatibility shims (though there is
still some trickiness involved).

```c
// Ensure custom types are registered before loading modules that use them.
// This only needs to be done once per instance.
IREE_CHECK_OK(iree_basic_custom_module_register_types(instance));

// Create the custom module that can be reused across contexts.
iree_vm_module_t* custom_module = NULL;
IREE_CHECK_OK(iree_basic_custom_module_create(instance, allocator,
                                              &custom_module));

// Create the context for this invocation reusing the loaded modules.
// Contexts hold isolated state and can be reused for multiple calls.
// Note that the module order matters: the input user module is dependent on
// the custom module.
iree_vm_module_t* modules[] = {custom_module, bytecode_module};
iree_vm_context_t* context = NULL;
IREE_CHECK_OK(iree_vm_context_create_with_modules(
    instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), modules,
    allocator, &context));
```

## Calling Custom Modules from Compiled Programs

The IREE compiler allows for external functions that are resolved at runtime
using the [MLIR `func` dialect](https://mlir.llvm.org/docs/Dialects/Func/). Some
optional attributes are used to allow for customization where required but in
many cases no additional IREE-specific work is required in the compiled program.
A few advanced features of the VM FFI are not currently exposed via this
mechanism such as variadic arguments and tuples but the advantage is that users
need not customize the IREE compiler in order to use their modules.

Prior to passing input programs to the IREE compiler users can insert the
imported functions as external
[`func.func`](https://mlir.llvm.org/docs/Dialects/Func/#funcfunc-mlirfuncfuncop)
ops and calls to those functions using
[`func.call`](https://mlir.llvm.org/docs/Dialects/Func/#funccall-mlirfunccallop):

```mlir
// An external function declaration.
// `custom` is the runtime module and `string.create` is the exported method.
// This call uses both IREE types (`!util.buffer`) and custom ones not known to
// the compiler but available at runtime (`!custom.string`).
func.func private @custom.string.create(!util.buffer) -> !custom.string
```

```mlir
// Call the imported function.
%buffer = util.buffer.constant : !util.buffer = "hello world!"
%result = func.call @custom.string.create(%buffer) : (!util.buffer) -> !custom.string
```

Users with custom dialects and ops can use
[MLIR's dialect conversion](https://mlir.llvm.org/docs/DialectConversion/)
framework to rewrite their custom ops to this form and perform additional
marshaling logic. For example, the above could have started as this program
before the user ran their dialect conversion and passed it in to `iree-compile`:

```mlir
%result = custom.string.create "hello world!" : !custom.string
```

See this samples [`example.mlir`](./test/example.mlir) for examples of features
such as signature specification and optional import fallback support.
