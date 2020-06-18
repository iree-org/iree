# IREE-specific CMake Options and Variables

This gives a brief explanation of IREE specific CMake options and variables.

*   `IREE_ENABLE_RUNTIME_TRACING`:BOOL

    Enables instrumented runtime tracing. Defaults to `OFF`.

*   `IREE_ENABLE_MLIR`:BOOL

    Enables MLIR/LLVM dependencies. Defaults to `ON`.

*   `IREE_ENABLE_EMITC`:BOOL

    Enables MLIR EmitC dependencies. Defaults to OFF. Requires that
    `IREE_ENABLE_MLIR` is set to `ON`.

*   `IREE_BUILD_COMPILER`:BOOL

    Builds the IREE compiler. Defaults to `ON`.

*   `IREE_BUILD_TESTS`:BOOL

    Builds IREE unit tests. Defaults to `ON`.

*   `IREE_BUILD_DOCS`:BOOL

    Builds IREE documentation. Defaults to `OFF`.

*   `IREE_BUILD_SAMPLES`:BOOL

    Builds IREE sample projects. Defaults to `ON`.

*   `IREE_BUILD_DEBUGGER`:BOOL

    Builds the IREE debugger app. Defaults to `OFF`.

*   `IREE_BUILD_PYTHON_BINDINGS`:BOOL

    Builds the IREE python bindings Defaults to `OFF`.

*   `IREE_BUILD_EXPERIMENTAL`:BOOL

    Builds experimental projects. Defaults to `OFF`.

*   `IREE_HAL_DRIVERS_TO_BUILD`:STRING

    *This does not have any effect at the moment, but will be supported in the
    future!* Semicolon-separated list of HAL drivers to build, or `all` for
    building all HAL drivers. Case-insensitive. Defaults to `all`. Example:
    `-DIREE_HAL_DRIVERS_TO_BUILD="Vulkan;VMLA"`.

*   `IREE_TARGET_BACKENDS_TO_BUILD`:STRING

    *This does not have any effect at the moment, but will be supported in the
    future!* Semicolon-separated list of HAL drivers to build, or `all` for
    building all HAL drivers. Case-insensitive. Defaults to `all`. Example:
    `-DIREE_HAL_DRIVERS_TO_BUILD="Vulkan_SPIRV;VMLA"`.

*   `IREE_ENABLE_LLD`:BOOL

    Use lld when linking. Defaults to `OFF`.

*   `IREE_MLIR_DEP_MODE`:STRING

    Defines the MLIR dependency mode. Case-sensitive. Can be `BUNDLED`,
    `DISABLED` or `INSTALLED`. Defaults to `BUNDLED`. If set to `INSTALLED`, the
    variable `MLIR_DIR` needs to be passed and that LLVM needs to be compiled
    with `LLVM_ENABLE_RTTI` set to `ON`.

*   `MLIR_DIR`:STRING

    Specifies the path where to look for the installed MLIR/LLVM packages.
    Required if `IREE_MLIR_DEP_MODE` is set to `INSTALLED`.
