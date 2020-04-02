# Getting Started on Windows with Vulkan

[Vulkan](https://www.khronos.org/vulkan/) is a new generation graphics and
compute API that provides high-efficiency, cross-platform access to modern GPUs
used in a wide variety of devices from PCs and consoles to mobile phones and
embedded platforms.

IREE includes a Vulkan/[SPIR-V](https://www.khronos.org/spir/) HAL backend
designed for executing advanced ML models in a deeply pipelined and tightly
integrated fashion on accelerators like GPUs.

This guide will walk you through using IREE's compiler and runtime Vulkan
components.

## Prerequisites

You should already have IREE cloned and building on your Windows machine. See
the [Getting Started on Windows with CMake](getting_started_windows_cmake.md)
guide for instructions.

You may have a physical GPU with drivers supporting Vulkan, but we also support
using [SwiftShader](https://swiftshader.googlesource.com/SwiftShader/) (a high
performance CPU-based implementation of Vulkan).

Vulkan API version > 1.2 is recommended, for the `VK_KHR_timeline_semaphore`
extension and other features, though the
[Vulkan-ExtensionLayer](https://github.com/KhronosGroup/Vulkan-ExtensionLayer)
project also provides a compatibility layer for implementations lacking native
support (such as SwiftShader).

## Vulkan Setup

### Background

TODO(scotttodd): high level overview on loader, ICDs, layers

### Quick Start

The
[dynamic_symbols_test](https://github.com/google/iree/blob/master/iree/hal/vulkan/dynamic_symbols_test.cc)
checks if the Vulkan loader and a valid ICD are accessible.

After building the `iree_hal_vulkan_dynamic_symbols_test` CMake target, run the
test:

```shell
$ .\build\iree\hal\vulkan\Debug\iree_hal_vulkan_dynamic_symbols_test.exe
```

If this test passes, you can skip down to the next section.

### Setting up the Vulkan Loader

If you see failures to find `vulkan-1.dll` (the Vulkan loader), install it by
either:

*   Updating your system's GPU drivers
*   Installing the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/)
*   Building the Vulkan loader
    [from source](https://github.com/KhronosGroup/Vulkan-Loader)

### Setting up SwiftShader

TODO(scotttodd): docs

### Setting up Vulkan-ExtensionLayer

TODO(scotttodd): docs

## Using IREE's Vulkan Compiler Target and Runtime Driver

### Compiling for the Vulkan HAL

```shell
$ .\build\iree\tools\Debug\iree-translate.exe -iree-mlir-to-vm-bytecode-module -iree-hal-target-backends=vulkan-spirv .\iree\tools\test\simple.mlir -o .\build\module.fb -print-ir-after-all
```

### Executing modules with the Vulkan driver

```shell
$ .\build\iree\tools\Debug\iree-run-module.exe -input_file=.\build\module.fb -driver=vulkan -entry_function=abs -inputs="i32=-2"
```

## Running IREE's Vulkan Samples

```shell
$ .\build\iree\samples\vulkan\Debug\vulkan_inference_gui.exe
```

## Debugging

TODO(scotttodd): Link to RenderDoc instructions, mention VK_LOADER_DEBUG, etc.
