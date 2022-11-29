# "Custom Module" samples

These samples demonstrate how to extend IREE with custom host code that can be
called from compiled modules. All modules regardless of type can call into each
other to allow for arbitrary module configurations.

## Basic sample

[samples/custom_module/basic/](/samples/custom_module/basic/README.md) shows how
to add a basic C++ custom module and use many of the more advanced features of
the module system.

* C++ VM wrappers for defining modules and using reference types
* Weak imports/fallback functions
* Custom types exposed to the compiler

## Tensor I/O

### Synchronous call sample

[samples/custom_module/sync/](/samples/custom_module/sync/README.md)
shows how to pass tensors to and from custom module imports with synchronous
execution. This approximates what a classic ML synchronous custom op may do by
presenting the tensor I/O as if they were host-synchronous buffers. This is the
lowest-performance way of running custom code and should be avoided when
possible.

* `tensor` types <-> HAL buffer views
* Host buffer mapping and manipulation

### Asynchronous call sample

[samples/custom_module/async/](/samples/custom_module/async/README.md)
shows how to pass tensors to and from custom module imports with asynchronous
execution. This shows how to move tensors across threads/frameworks in a
non-blocking way that allows IREE to overlap execution with custom user code.

* `tensor` types <-> HAL buffer views
* Fences for waiting on inputs and signaling readiness of outputs
* Side-effect annotations for wait-free imports
