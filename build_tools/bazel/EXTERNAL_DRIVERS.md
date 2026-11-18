# External HAL Driver Registration System

## Overview

External HAL drivers can be added to the IREE runtime in Bazel builds without
modifying IREE source code. This matches `IREE_EXTERNAL_HAL_DRIVERS` in the
CMake build.

The feature is opt-in. Any Bazel module that depends on IREE can register
drivers from its own `MODULE.bazel`. Without any registration, the build is
unchanged.

## How to Opt In

1. Provide a registration library for your driver (see
   [Driver Registration Requirements](#driver-registration-requirements)).

2. In your `MODULE.bazel`, depend on IREE:

   ```starlark
   bazel_dep(name = "iree_core")
   ```

   If IREE is a submodule of your repository, point the dependency at it:

   ```starlark
   local_path_override(
       module_name = "iree_core",
       path = "third_party/iree",
   )
   ```

3. In the same `MODULE.bazel`, add one `external_hal_driver` tag per driver to
   IREE's module extension:

   ```starlark
   iree_ext = use_extension("@iree_core//build_tools/bazel:extensions.bzl", "iree_extension")

   iree_ext.external_hal_driver(
       target = "//my_driver/registration:my_driver_registration",
       register_fn = "iree_hal_my_driver_module_register",
   )
   ```

   You do not need a `use_repo()` call for this.

4. Build as usual. The drivers are then automatically:
   - compiled and linked into the IREE runtime (`//runtime/src/iree/hal/drivers`)
   - registered at runtime initialization
   - available without further configuration

To opt out again, remove the tags.

## Tag Attributes

- `target` (required): Label of the driver registration library.
- `register_fn` (required): Name of the C registration function in `target`.
- `enabled_by` (optional): Label of a `config_setting`. If set, the driver is
  only built and registered when it matches. If not set, the driver is always
  enabled.

Labels are resolved relative to the module that adds the tag. For example,
`//my_driver/...` refers to a package in your module, not in IREE.

Each `register_fn` may only be registered once across all modules.

## Enabling a Driver Conditionally

Use `enabled_by` to build a driver only for some configurations:

```starlark
iree_ext.external_hal_driver(
    target = "//my_driver/registration:my_driver_registration",
    register_fn = "iree_hal_my_driver_module_register",
    enabled_by = "//my_driver:enabled",
)
```

With a matching `config_setting` in `my_driver/BUILD.bazel`:

```starlark
config_setting(
    name = "enabled",
    define_values = {"my_driver": "1"},
)
```

The driver is then only included when building with `--define=my_driver=1`.

## Driver Registration Requirements

Each external driver must provide:

1. **Registration Library** - A Bazel target with:
   - Header: Registration function declaration
   - Source: Registration function implementation
   - Dependencies: Driver implementation + IREE HAL

2. **Registration Function** - Following this signature:

   ```c
   iree_status_t my_driver_module_register(iree_hal_driver_registry_t* registry);
   ```

## Which Modules Can Register Drivers

Tags from all modules that use `iree_extension` are collected, not only from
the root module. This means:

- A project that includes IREE as a submodule can register its drivers.
- A library module that bundles a driver can register it. Every project that
  depends on that library then gets the driver.

## How It Works

1. Modules add `external_hal_driver` tags to `iree_extension` in their
   `MODULE.bazel`.
2. `iree_extension` collects the tags of all modules and generates the
   `@iree_external_hal_drivers` repository. Its `drivers.bzl` holds the list
   of drivers. The list is empty if no module adds a tag.
3. `runtime/src/iree/hal/drivers/BUILD.bazel` loads this list. If it is not
   empty, the drivers library:
   - links the enabled driver targets,
   - compiles a generated `init_external.c` that calls each enabled
     registration function, and
   - defines `IREE_HAVE_HAL_EXTERNAL_DRIVERS`.
4. The runtime calls `iree_hal_register_external_drivers()` at initialization.
