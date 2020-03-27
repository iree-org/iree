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

The `VK_KHR_timeline_semaphore` extension is required. For systems where it is
not natively available, the `VK_LAYER_KHRONOS_timeline_semaphore` compatibility
layer may be used (see below).

### Supported Configurations

TODO(benvanik): fill out hardware and OS matrix.

### Vulkan ICDs/Loader

Vulkan applications interface with the Vulkan API through a series of systems
including the Vulkan Loader which dispatches Vulkan functions to various layers
and Installable Client Drivers (ICDs).

See also:

*   https://www.lunarg.com/tutorial-overview-of-vulkan-loader-layers/
*   https://vulkan.lunarg.com/doc/view/latest/windows/loader_and_layer_interface.html
*   https://github.com/KhronosGroup/Vulkan-Loader

#### Installing or Updating the Loader

The Vulkan loader should already be installed on Vulkan-capable systems, but you
might need to update it if the version that is installed is too old to use some
of the more recently added Vulkan features.

To get a more recent version of the loader, these are some options:

*   Update your system's GPU drivers
*   Install the Vulkan SDK: https://www.lunarg.com/vulkan-sdk/
*   Build the Vulkan loader from source:
    https://github.com/KhronosGroup/Vulkan-Loader

If you build the loader from source, you may also need to set
`LD_PRELOAD=/path/to/libvulkan.so.1` (or equivalent) to bypass the installed
loader.

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
configured together is //iree/hal/vulkan:dynamic_symbols_test.

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

TODO(benvanik): document optional extensions (e.g. VK_EXT_debug_utils)

### Timeline Semaphore Support

IREE uses Vulkan's timeline semaphores from the `VK_KHR_timeline_semaphore`
extension, which is core in Vulkan 1.2. For older Vulkan installs without native
support for this extension, the
[Vulkan-ExtensionLayer](https://github.com/KhronosGroup/Vulkan-ExtensionLayer)
project provides support through the `VK_LAYER_KHRONOS_timeline_semaphore`
layer.

Developers may build/install this layer on their own, or using the Bazel or
CMake build targets at
[build_tools/third_party/vulkan_extensionlayer](../build_tools/third_party/vulkan_extensionlayer).

After building the layer, set the `VK_LAYER_PATH` environment variable to
include the path to the associated manifest file (see the
[docs](https://vulkan.lunarg.com/doc/view/latest/windows/loader_and_layer_interface.html)
for details).

## SPIR-V

### Minimum Requirements

TODO(benvanik): minimum extensions and caps.

### Workgroups and Subgroups

TODO(benvanik): subgroups in 1.1
https://www.khronos.org/blog/vulkan-subgroup-tutorial - basic/vote are
supported: https://vulkan.gpuinfo.org/displayreport.php?id=6436#device

## Debugging and Profiling

### RenderDoc

RenderDoc captures of IREE Vulkan execution may be recorded through RenderDoc's
GUI and through IREE's headless (command line) programs.

For IREE's GUI programs like samples/vulkan/vulkan_inference_gui, you should be
able to launch the program through RenderDoc itself and control captures using
the UI overlay it injects.

For IREE's headless programs, set the `vulkan_enable_renderdoc` flag to tell
IREE to load RenderDoc, connect to it's in-application API, and trigger
capturing on its own. For example, this command runs `iree-run-mlir` on a simple
MLIR file with some sample input values and saves a RenderDoc capture to the
default location on your system (e.g. /tmp/RenderDoc/):

```shell
$ bazel build iree/tools:iree-run-mlir && LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RENDERDOC_LIB bazel-bin/iree/tools/iree-run-mlir $PWD/iree/samples/vulkan/simple_mul.mlir -iree-hal-target-backends=vulkan-spirv -input-value="4xf32=1,2,3,4" -input-value="4xf32=2,4,6,8" -run-arg="--vulkan_renderdoc"
```

You can also launch IREE's headless programs through RenderDoc itself, just be
sure to set the command line arguments appropriately. Saving capture settings in
RenderDoc can help if you find yourself doing this frequently.

Note: RenderDoc version 1.7 or higher is needed to record captures from IREE's
headless compute programs.
