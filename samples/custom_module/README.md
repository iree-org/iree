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
