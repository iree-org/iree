# IREE HAL: Vulkan and SPIR-V

[Vulkan](https://www.khronos.org/vulkan/) is a new generation graphics and
compute API that provides high-efficiency, cross-platform access to modern GPUs
used in a wide variety of devices from PCs and consoles to mobile phones and
embedded platforms.

IREE includes a Vulkan/[SPIR-V](https://www.khronos.org/spir/) HAL backend
designed for executing advanced ML models in a deeply pipelined and tightly
integrated fashion on accelerators like GPUs.

## Vulkan

### Minimum Requirements

TODO(benvanik): specify required extensions and caps. Cross reference with
gpuinfo.

### Supported Configurations

TODO(benvanik): fill out hardware and OS matrix.

### Vulkan ICDs/Loader

Vulkan applications interface with the Vulkan API through a series of systems
including the Vulkan Loader which dispatches Vulkan functions to various layers
and Installable Client Drivers (ICDs).

See also:

*   https://www.lunarg.com/tutorial-overview-of-vulkan-loader-layers/
*   https://vulkan.lunarg.com/doc/view/1.1.70.1/windows/loader_and_layer_interface.html
*   https://github.com/KhronosGroup/Vulkan-Loader

#### Choosing ICDs and SwiftShader

On systems with a GPU and modern drivers, the Vulkan loader should already be
present and configured to find and use the GPU's stable Vulkan driver. As
needed, you can configure the loader to use alternate ICDs like
[SwiftShader](https://swiftshader.googlesource.com/SwiftShader/) (a high
performance CPU-based implementation of Vulkan) or beta drivers.

Setting the `VK_ICD_FILENAMES` environment variable will instruct the Vulkan
loader to read an ICD manifest file at that location. See
[build_vk_swiftshader.sh](../build_tools/third_party/swiftshader/build_vk_swiftshader.sh)
for a working example of how to build and configure SwiftShader as an ICD for
IREE to use.

#### Debugging the Loader

Setting the environment variable `VK_LOADER_DEBUG=all` will enable verbose
logging during the loader initialization. This is especially useful when trying
to verify expected paths are being searched for layers or driver JSON manifests.

The simplest test for ensuring that The Vulkan loader and IREE are correctly
configured together is //iree/hal/vulkan:dynamic_symbols_test. Once that works,
you should also be able to run //iree/samples/hal:simple_compute_test and see
the Vulkan HAL in action.

#### Enabling Validation Layers

The `--vulkan_validation_layers=true` flag can be used to enable the standard
validation meta-layer (`VK_LAYER_LUNARG_standard_validation`) containing all the
common layers. Always run with this enabled unless you are profiling/
benchmarking as it's the
[ASAN-/TSAN-/MSAN](https://github.com/google/sanitizers)-equivalent for Vulkan
and will validate the usage and lifetime of API objects. If you're seeing weird
things the first step is to check the logs for validation warnings/errors.

#### Other Troubleshooting

You may need to preload your system Vulkan driver and define the driver shared
object path prior to loading an application. For example, on Linux with NVIDIA:

```
LD_PRELOAD=libGLX_nvidia.so.0 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/ \
    ./bazel-bin/my/app
```

Failing to preload may result in an error like this (when `VK_LOADER_DEBUG=all`
is defined):

```
DEBUG: Searching for ICD drivers named libGLX_nvidia.so.0, using default dir
ERROR: dlopen: cannot load any more object with static TLS
```

If you are running under an SSH session to Linux you may also need to set the
display if your drivers require X11:

```
DISPLAY=:0
```

### IREE's Vulkan API Usage

#### Optional Extensions

## SPIR-V

### Minimum Requirements

TODO(benvanik): minimum extensions and caps.

### Workgroups and Subgroups

TODO(benvanik): subgroups in 1.1
https://www.khronos.org/blog/vulkan-subgroup-tutorial - basic/vote are
supported: https://vulkan.gpuinfo.org/displayreport.php?id=6436#device

## Debugging and Profiling

### RenderDoc

TODO(benvanik): launching with renderdoc.
