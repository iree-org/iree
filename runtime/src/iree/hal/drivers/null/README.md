# Null HAL Driver (`null`)

This is a skeleton HAL driver that implements most interfaces needed by a HAL
driver. It contains notes that can be useful to implementers, stubs for
easy copy/paste/renaming, and some default behavior in places that many
implementations can use (such as file queue operations). It doesn't run anything
and nearly all methods return `UNIMPLEMENTED` errors: this can be used to
incrementally build a driver by running until you hit an UNIMPLEMENTED,
implementing it, and running again. Note however that the HAL is fairly regular
and starting with the files and running down the methods is often much easier
to do than the trial-and-error approach: if you can implement command buffer
fill (memset) you can often implement copy (memcpy) as well at the same time.

## Instructions for Cloning

1. Duplicate the entire directory in your own repository or the IREE
   `experimental/` folder if going in-tree.
1. Find/replace `{Null}` with the friendly name of your driver (e.g. `Vulkan`).
1. Find/replace `_null_` with the C name of your driver (e.g. `vulkan`).
1. Find/replace `// TODO(null):` with your github ID, your driver name, or a
   GitHub issue number tracking driver creation (e.g. `// TODO(#1234):`).

## Build Setup

HAL drivers are setup by adding some specially named cmake variables and then
pointing the IREE build at them by name. Projects embedding IREE runtime builds
as a submodule can use the `iree_register_external_hal_driver` cmake function
to do this, set the variables on the command line during cmake configure, or
via top-level project `CMakeLists.txt` before adding the IREE subdirectory.

See [runtime/src/iree/hal/drivers/CMakeLists.txt](runtime/src/iree/hal/drivers/CMakeLists.txt) for more information.

Example using the helper function:
```cmake
iree_register_external_hal_driver(
  NAME
    webgpu
  SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/experimental/webgpu"
  BINARY_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/experimental/webgpu"
  DRIVER_TARGET
    iree::experimental::webgpu::registration
  REGISTER_FN
    iree_hal_webgpu_driver_module_register
)
set(IREE_EXTERNAL_HAL_DRIVERS my_driver)
```

Example using the command line:
```sh
cmake ... \
    -DIREE_EXTERNAL_MY_DRIVER_HAL_DRIVER_TARGET=my_driver_static_library \
    -DIREE_EXTERNAL_MY_DRIVER_HAL_DRIVER_REGISTER=my_driver_register \
    -DIREE_EXTERNAL_HAL_DRIVERS=my_driver
```

## In-tree Drivers (`iree/hal/drivers/...`)

IREE is generally conservative about hosting in-tree HAL drivers unless authored
by the core team or an SLA is in-place and maintained. Any new HAL drivers
should expect to start in forks or external repositories and not be expected
to merge without deep involvement with the IREE maintainers. IREE is not a
monorepo and it's perfectly fine to be out-of-tree. If ergonomics issues are
encountered with being out of tree please file issues so that support can be
improved.
