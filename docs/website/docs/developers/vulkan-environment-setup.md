---
hide:
  - tags
tags:
  - GPU
  - Vulkan
icon: octicons/server-16
---

# Vulkan environment setup

[Vulkan](https://www.khronos.org/vulkan/) is a new generation graphics and
compute API that provides high-efficiency, cross-platform access to modern GPUs
used in a wide variety of devices from PCs and consoles to mobile phones and
embedded platforms.

This page lists steps and tips for setting up and troubleshooting a Vulkan
development environment. The information here is meant to be generic.

## Vulkan architecture

Vulkan adopts a layered architecture, which aims to better support extensiblity.
There are four components involved in this architecture:

* The Vulkan Application
* [The Vulkan Loader][VulkanLoader]
* [Vulkan Layers][VulkanLayer]
* [Installable Client Drivers (ICDs)][VulkanICD]

![High Level View of Loader][VulkanArchPicture]

The Vulkan loader sits between the Vulkan application, which calls Vulkan APIs,
and the ICDs, which implements these Vulkan APIs. Vulkan layers agument the
Vulkan system to provide optional features like validation and debugging. The
Vulkan loader composes a chain of requested layers, which processes the Vulkan
application's API calls one by one, and finally redirects the API calls made by
the Vulkan application to one or more ICDs.

It's highly recommned to read the
[Architecture of the Vulkan Loader Interfaces Overview][VulkanArchOverview] to
get a general understanding of what these components are and how they interact
with one another.

## Vulkan environment setup

### Windows

You need to install the [Vulkan SDK][VulkanSDK] from LunarG to get the Vulkan
loader.

Typically the Vulkan SDK will be installed at `C:\VulkanSDK\<version>\` and
there will be an environment variable `VULKAN_SDK` pointing to it. You can run
the `vulkancube` executable under the `Bin\` subdirectory of the Vulkan SDK to
make sure everything works properly. If not, you probably need to check whether
the graphics card is Vulkan capable or update the driver.

### Debian/Ubuntu

For Ubuntu 20.04/22.04, it's recommended to directly install the full
[Vulkan SDK][VulkanSDK] from LunarG's APT sources for the loader and various
developer tools.

If you want to have a minimal environment, the following packages should be
installed for a proper Vulkan runtime:

* [`libvulkan1`][PackageLibVulkan1] for the Vulkan loader `libvulkan.so`.
* For AMD GPUs, you can install
    * [`mesa-vulkan-drivers`][PackageMesaVulkan] for the Mesa AMD Vulkan ICD, or
    * AMD's official [VUlkan ICD][AMDVulkanDriver].
* For NVIDIA GPUs, you can install
    * [`nvidia-vulkan-icd`][PackageNvidiaVulkan] on Debian for NVIDIA Vulkan ICD.
    * the most recent `nvidia-driver-*` package on Ubuntu for NVIDIA Vulkan ICD.

The above packages provide the Vulkan loader and ICDs. With them a Vulkan
application should be able to run. You may additionally want to install

* [vulkan-tools][PackageVulkanTools] for command-line tools like `vulkaninfo`
  (dumping available ICDs and their capabilities) and GUI application like
  `vulkancube` (rendering a rotating cube).

In order to develop Vulkan applications, you additionally need the following
packages:

* [libvulkan-dev][PackageVulkanDev] for various Vulkan header files.
* [vulkan-validationlayers][PackageVulkanValidation] for Vulkan validation
    layers like `VkLayer_standard_validation`.

### Linux

For other Linux distros, please consult the corresponding package management
tools for the packages needed. (And please feel free to update this doc
regarding them.)

You can also download and install the [Vulkan SDK][VulkanSDK] tarball from
LunarG. It packages the loader with many useful layers and other shader tools.

You can also build the Vulkan SDK component projects like
[Vulkan-Loader][VulkanLoaderSource] and
[Vulkan-ValidationLayers][VulkanValidationLayersSource] from source. But note
that building these components separately you need to make sure they are
consistent with one another (e.g., using the same version of Vulkan headers) to
function together.

### Android

Please make sure your Android device is Vulkan capable. Vulkan is supported on
Android since 7, but we track newer Android versions (10+) closely and haven't
set a clear min version yet.

## Multiple Vulkan SDKs

If you have multiple versions of Vulkan loaders exist, you may also need to set
`LD_LIBRARY_PATH` and `LD_PRELOAD` to load the desired version of the loader.
For example:

```shell
LD_LIBRARY_PATH={PATH_TO_VULKAN_SDK}/x86_64/lib/
LD_PRELOAD=libvulkan.so.1
```

This can also be done by sourcing the proper `setup-env.sh` from one of the
downloaded Vulkan SDKs.

## Vulkan environment troubleshooting

### Useful environment variables

There are a few environment variables that can alter the default Vulkan loader
behavior and print verbose information, notably:

* `VK_LOADER_DEBUG`: enable loader debug messages. Setting it to `all` will
    enable the most verbose logging from the loader. This is especially useful
    when trying to see what layers/ICDs are searched and used.
* `VK_ICD_FILENAMES`: force the loader to use a specific ICD. This is
    especially useful when you have multiple Vulkan capable devices and want to
    select which one to use manually.
* `VK_INSTANCE_LAYERS`: force the loader to enable the given layers. For
    example, You can force enable `VK_LAYER_LUNARG_api_dump` to have a detailed
    dump of all Vulkan API calls made by the application. You can force enable
    `VK_LAYER_LUNARG_core_validation` to validate the API calls made by the
    application.
* `VK_LAYER_PATH`: override the loader's standard layer library search folders.

Please see the [Vulkan loader's documentation][VulkanLoaderEnvVars] for detailed
explanation for these variables.

### Setting environment variables for Bazel test

Bazel runs tests in a sandbox and environment variables must be passed through
to the test runner. Consider putting environment setup in a `user.bazelrc` to
save typing. For example:

```starlark
test --test_env="LD_LIBRARY_PATH=/absolute/path/to/vulkan/sdk/x86_64/lib/"
test --test_env="LD_PRELOAD=libvulkan.so.1"
test --test_env="VK_LAYER_PATH=/absolute/path/to/additional/layers/:$VK_LAYER_PATH"
```

### Vulkan function `vkCreateInstance` not available on Android

Since Android 8 Oreo, Android re-architected the OS framework with
[project Treble](https://source.android.com/devices/architecture#hidl).
Framework libraries and
[vendor libraries](https://source.android.com/devices/architecture/vndk) have a
more strict and clear separation. Their dependencies are carefully scrutinized
and only selected cases are allowed. This is enforced with
[linker namespaces](https://source.android.com/devices/architecture/vndk/linker-namespace).

`/data/local/tmp` is the preferred directory for automating native binary tests
built using NDK toolchain. They should be allowed to access libraries like
`libvulkan.so` for their functionality. However, there was an issue with fully
treblized Android 10 where `/data/local/tmp` did not have access to the linker
namespaces needed by `libvulkan.so`. This should be
[fixed](https://android.googlesource.com/platform/system/linkerconfig/+/296da5b1eb88a3527ee76352c2d987f82f3252eb)
now. But as typically in the Android system, it takes a long time to see the fix
getting propagated, if ever.

A known workaround is to symlink the vendor Vulkan implementation under
`/vendor/lib[64]` as `libvulkan.so` under `/data/local/tmp` and use
`LD_LIBRARY_PATH=/data/local/tmp` when invoking IREE executables.

For Qualcomm Adreno GPUs, the vendor Vulkan implementation is at
`/vendor/lib[64]/hw/vulkan.*.so`. So for example for Snapdragon 865:

```shell
adb shell ln -s /vendor/lib64/hw/vulkan.kona.so /data/local/tmp/libvulkan.so
```

For ARM Mali GPUs, there is only one monolithic driver
(`/vendor/lib[64]/libGLES_mali.so`) for OpenGL and Vulkan and the Vulkan vendor
driver (`/vendor/lib[64]/hw/vulkan.*.so`) is just a symlink to it. So for
example:

```shell
adb shell ln -s /vendor/lib64/libGLES_mali.so /data/local/tmp/libvulkan.so
```

### SSH on Linux and X forwarding

Physical devices enumerated on NVIDIA drivers can be affected by the `DISPLAY`
environment variable. If you are running under an SSH session to Linux or using
chrome remote desktop and have problems with physical device enumeration, you
probably want to check the `DISPLAY` environment and set it to point to a
display at the server side, for example:

```bash
export DISPLAY=:0
```

[VulkanArchOverview]: https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md#overview
[VulkanArchPicture]: https://raw.githubusercontent.com/KhronosGroup/Vulkan-Loader/main/docs/images/high_level_loader.png
[VulkanICD]: https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md#installable-client-drivers
[VulkanLayer]: https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md#layers
[VulkanLoader]: https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md#the-loader
[VulkanLoaderEnvVars]: https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md#table-of-debug-environment-variables
[VulkanLoaderSource]: https://github.com/KhronosGroup/Vulkan-Loader
[VulkanSDK]: https://vulkan.lunarg.com/sdk/home
[VulkanValidationLayersSource]: https://github.com/KhronosGroup/Vulkan-ValidationLayers
[PackageLibVulkan1]: https://packages.ubuntu.com/jammy/libvulkan1
[PackageMesaVulkan]: https://packages.ubuntu.com/jammy/mesa-vulkan-drivers
[PackageNvidiaVulkan]: https://packages.debian.org/bookworm/nvidia-vulkan-icd
[PackageVulkanDev]: https://packages.ubuntu.com/jammy/libvulkan-dev
[PackageVulkanTools]: https://packages.ubuntu.com/jammy/vulkan-tools
[PackageVulkanValidation]: https://packages.ubuntu.com/eoan/vulkan-validationlayers
[AMDVulkanDriver]: https://www.amd.com/en/support/linux-drivers
