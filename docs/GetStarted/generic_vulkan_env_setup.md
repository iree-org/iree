# Generic Vulkan Development Environment Setup and Troubleshooting

[Vulkan](https://www.khronos.org/vulkan/) is a new generation graphics and
compute API that provides high-efficiency, cross-platform access to modern GPUs
used in a wide variety of devices from PCs and consoles to mobile phones and
embedded platforms.

This page lists steps and tips for setting up and trouble shooting a Vulkan
development envirnoment. The information here is meant to be generic.

## Vulkan architecture

Vulkan adopts a layered architecture, which aims to better support extensiblity.
There are four components involved in this architecture:

*   The Vulkan Application
*   [The Vulkan Loader][VulkanLoader]
*   [Vulkan Layers][VulkanLayer]
*   [Installable Client Drivers (ICDs)][VulkanICD]

![High Level View of Loader][VulkanArchPicture]

The Vulkan loader sits between the Vulkan application, which calls Vulkan APIs,
and the ICDs, which implements these Vulkan APIs. Vulkan layers agument the
Vulkan system to provide optional features like validation and debugging. The
Vulkan loader composes a chain of requested layers, which processes the Vulkan
application's API calls one by one and finallly redirects the API calls made by
the Vulkan application to one or more ICDs.

It's highly recommned to read the
[Architecture of the Vulkan Loader Interfaces Overview][VulkanArchOverview] to
get a general understanding of what these components are and how they interact
with one another.

## Vulkan development environment setup

### Windows

You need to install the [Vulkan SDK][VulkanSDK] from LunarG to get the Vulkan
loader.

Typically the Vulkan SDK will be installed at ``C:\VulkanSDK\<version>\` and
there will be an environment variable``VULKAN_SDK`pointing to it. You can run
the`vulkancube`executable under the`Bin\` subdirectory of the Vulkan SDK to make
sure everything works properly. If not, you probably need to check whether the
graphics card is Vulkan capable or update the driver.

### Debian/Ubuntu

The following packages should be installed for a proper Vulkan runtime to test
the runtime functions properly:

*   [libvulkan1][PackageLibVulkan1] for the Vulkan loader `libvulkan.so`.
*   (AMD) [mesa-vulkan-drivers][PackageMesaVulkan] for Mesa AMD Vulkan ICD.
*   (NVIDIA) [nvidia-vulkan-icd][PackageNvidiaVulkan] for NVIDIA Vulkan ICD.

The above packages provide the Vulkan loader and ICDs. With them an Vulkan
application should be able to run. You may additionally want to install

*   [vulkan-tools][PackageVulkanTools] for command-line tools like `vulkaninfo`
    (dumping available ICDs and their capabilities) and GUI application like
    `vulkancube` (rendering a rotating cube).

In order to develop Vulkan applications, you additionally need the following
pacages:

*   [libvulkan-dev][PackageVulkanDev] for various Vulkan header files.
*   [vulkan-validationlayers][PackageVulkanValidation] for Vulkan validation
    layers like `VkLayer_standard_validation`.

### Linux

For other Linux distros, please consult the corresponding package managment
tools for the packages needed. (And please feel free to update this doc
regarding them.)

You can also download and install the [Vulkan SDK][VulkanSDK] from LunarG. It
packages the loader with many useful layers. The source code of the SDK
component projects are included, allowing you to recompile the artifacts if
needed.

You can also build the Vulkan SDK component projects like
[Vulkan-Loader][VulkanLoaderSource] and
[Vulkan-ValidationLayers][VulkanValidationLayersSource] from source. But note
that building these components separately you need to make sure they are
consistent with one another (e.g., using the same version of Vulkan headers) to
function together.

If you have multiple versions of Vulkan loaders exist, you may also need to set
`LD_LIBRARY_PATH` and `LD_PRELOAD` to load the desired version of the loader.
For example:

```shell
$ LD_LIBRARY_PATH={PATH_TO_VULKAN_SDK}/x86_64/lib/
$ LD_PRELOAD=libvulkan.so.1
```

## Vulkan development environment troubleshooting

### Useful environment variables

There are a few environmet variables that can alter the default Vulkan loader
behavior and print verbose information, notably:

*   `VK_LOADER_DEBUG`: enable loader debug messages.
*   `VK_ICD_FILENAMES`: force the loader to use a specific ICD.
*   `VK_INSTANCE_LAYERS`: force the loader to enable the given layers.
*   `VK_LAYER_PATH`: override the loader's standard layer libary search folders.

Please see the [Vulkan loader's documentation][VulkanLoaderEnvVars] for detailed
explanation for these variables.

[VulkanArchOverview]: https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#overview
[VulkanArchPicture]: https://raw.githubusercontent.com/KhronosGroup/Vulkan-Loader/master/loader/images/high_level_loader.png
[VulkanICD]: https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#installable-client-drivers
[VulkanLayer]: https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#layers
[VulkanLoader]: https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#the-loader
[VulkanLoaderEnvVars]: https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#table-of-debug-environment-variables
[VulkanLoaderSource]: https://github.com/KhronosGroup/Vulkan-Loader
[VulkanSDK]: https://www.lunarg.com/vulkan-sdk/
[VulkanValidationLayersSource]: https://github.com/KhronosGroup/Vulkan-ValidationLayers
[PackageLibVulkan1]: https://packages.ubuntu.com/focal/libvulkan1
[PackageMesaVulkan]: https://packages.ubuntu.com/focal/mesa-vulkan-drivers
[PackageNvidiaVulkan]: https://packages.debian.org/buster/nvidia-vulkan-icd
[PackageVulkanDev]: https://packages.ubuntu.com/focal/libvulkan-dev
[PackageVulkanTools]: https://packages.ubuntu.com/focal/vulkan-tools
[PackageVulkanValidation]: https://packages.ubuntu.com/eoan/vulkan-validationlayers
