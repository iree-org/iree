---
hide:
  - tags
tags:
  - GPU
  - Vulkan
icon: octicons/server-16
---

# GPU deployment using Vulkan

IREE can accelerate model execution on GPUs via
[Vulkan](https://www.khronos.org/vulkan/), a low-overhead graphics and compute
API. Vulkan is cross-platform: it is available on many operating systems,
including Android, Linux, and Windows. Vulkan is also cross-vendor: it is
supported by most GPU vendors, including AMD, ARM, Intel, NVIDIA, and Qualcomm.

## :octicons-project-roadmap-16: Support matrix

As IREE and the compiler ecosystem it operates within matures, more
target specific optimizations will be implemented. At this stage, expect
reasonable performance across all GPUs and for improvements to be
made over time for specific vendors and architectures.

GPU Vendor | Category | Performance | Focus Architecture
:--------: | :------: | :---------: | :----------------:
ARM Mali GPU | Mobile |  Good | Valhall+
Qualcomm Adreno GPU | Mobile | Reasonable | 640+
AMD GPU | Desktop/server | Good | RDNA+
NVIDIA GPU | Desktop/server | Reasonable | Turing+

## :octicons-download-16: Prerequisites

In order to use Vulkan to drive the GPU, you need to have a functional Vulkan
environment. IREE requires Vulkan 1.1 on Android and 1.2 elsewhere. It can be
verified by the following steps:

=== "Android"

    Android mandates Vulkan 1.1 support since Android 10. You just need to
    make sure the device's Android version is 10 or higher.

=== ":fontawesome-brands-linux: Linux"

    Run the following command in a shell:

    ``` shell
    vulkaninfo | grep apiVersion
    ```

    If `vulkaninfo` does not exist, you will need to [install the latest Vulkan
    SDK](https://vulkan.lunarg.com/sdk/home/). Installing via LunarG's package
    repository is recommended, as it places Vulkan libraries and tools under
    system paths so it's easy to discover.

    If the listed version is lower than Vulkan 1.2, you will need to update the
    driver for your GPU.

=== ":fontawesome-brands-windows: Windows"

    Run the following command in a shell:

    ``` shell
    vulkaninfo | grep apiVersion
    ```

    If `vulkaninfo` does not exist, you will need to [install the latest Vulkan
    SDK](https://vulkan.lunarg.com/sdk/home/).

    If the listed version is lower than Vulkan 1.2, you will need to update the
    driver for your GPU.

### Get the IREE compiler

Vulkan expects the program running on GPU to be expressed by the
[SPIR-V](https://www.khronos.org/registry/spir-v/) binary exchange format, which
the model must be compiled into.

#### :octicons-download-16: Download the compiler from a release

Python packages are distributed through multiple channels. See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core [`iree-base-compiler`](https://pypi.org/project/iree-base-compiler/)
package includes the SPIR-V compiler:

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-compiler-from-release.md"

#### :material-hammer-wrench: Build the compiler from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build
IREE for your host platform. The SPIR-V compiler backend is compiled in by
default on all platforms, though you should ensure that the
`IREE_TARGET_BACKEND_VULKAN_SPIRV` CMake option is `ON` when configuring.

!!! tip
    `iree-compile` will be built under the `iree-build/tools/` directory. You
    may want to include this path in your system's `PATH` environment variable.

### Get the IREE runtime

Next you will need to get an IREE runtime that supports the Vulkan HAL driver.

#### :octicons-download-16: Download the runtime from a release

Python packages are distributed through multiple channels. See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core [`iree-base-runtime`](https://pypi.org/project/iree-base-runtime/)
package includes the Vulkan HAL driver:

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-runtime-from-release.md"

#### :material-hammer-wrench: Build the runtime from source

Please make sure you have followed one of the
[Building from source](../../building-from-source/index.md) pages to build
IREE for your target platform. The Vulkan HAL driver is compiled in by default
on supported platforms, though you should ensure that the
`IREE_HAL_DRIVER_VULKAN` CMake option is `ON` when configuring.

#### :octicons-checklist-24: Check for Vulkan devices

You can check for Vulkan support by looking for a matching driver and device:

```console hl_lines="12"
--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-driver-list.md:2"
```

```console hl_lines="6"
--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-device-list-amd.md"
```

To see device details, including hints about what to use as a
[Vulkan target](#choosing-vulkan-targets) when
[compiling a program](#compile-and-run-a-program):

```console hl_lines="9-10"
$ iree-run-module --dump_devices

...
# ============================================================================
# Enumerated devices for driver 'vulkan'
# ============================================================================

# ===----------------------------------------------------------------------===
# --device=vulkan://00000000-1111-2222-3333-444444444444
#   AMD Radeon PRO W7900 Dual Slot  (RADV GFX1100)
# ===----------------------------------------------------------------------===
```

## Compile and run a program

With the requirements out of the way, we can now compile a model and run it.

### :octicons-file-code-16: Compile a program

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-import-onnx-mobilenet.md"

Then run the following command to compile with the `vulkan` target device:

``` shell hl_lines="2 3"
iree-compile \
    --iree-hal-target-device=vulkan \
    --iree-vulkan-target=<...> \
    mobilenetv2.mlir -o mobilenet_vulkan.vmfb
```

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-optimization-options.md"

#### Choosing Vulkan targets

The `--iree-vulkan-target` specifies the GPU architecture to target. It
accepts a few schemes:

* LLVM CodeGen backend style: this is using LLVM AMDGPU/NVPTX CodeGen targets
  like `gfx1100` for AMD RX 7900XTX and `sm_86` for NVIDIA RTX 3090 GPUs.
* Architecture code name style like `rdna3`/`valhall4`/`ampere`/`adreno`
  for AMD/ARM/NVIDIA/Qualcomm GPUs.
* Product name style: e.g., using `rx7900xtx`/`a100` for corresponding GPUs.

Here are a few examples showing how you can target various recent common GPUs:

| GPU                 | Product Name   | Target Architecture | Architecture Code Name |
| ------------------- | -------------- | ------------------- | ---------------------- |
| AMD RX 5000 series  |                |                     | `rdna1`                |
| AMD RX 6000 series  |                |                     | `rdna2`                |
| AMD RX 7700XT       | `rx7700xt`     | `gfx1101`           | `rdna3`                |
| AMD RX 7800XT       | `rx7800xt`     | `gfx1101`           | `rdna3`                |
| AMD RX 7900XT       | `rx7900xt`     | `gfx1100`           | `rdna3`                |
| AMD RX 7900XTX      | `rx7900xtx`    | `gfx1100`           | `rdna3`                |
| AMD RX 9060XT       | `rx9060xt`     | `gfx1200`           | `rdna4`                |
| AMD RX 9070         | `rx9070`       | `gfx1201`           | `rdna4`                |
| AMD RX 9070XT       | `rx9070xt`     | `gfx1201`           | `rdna4`                |
| ARM GPUs            |                |                     | `valhall`              |
| ARM Mali G510       | `mali-g510`    |                     | `valhall3`             |
| ARM Mali G715       | `mali-g715`    |                     | `valhall4`             |
| NVIDIA RTX20 series | `rtx2070super` | `sm_75`             | `turing`               |
| NVIDIA RTX30 series | `rtx3080ti`    | `sm_86`             | `ampere`               |
| NVIDIA RTX40 series | `rtx4090`      | `sm_89`             | `ada`                  |
| Qualcomm GPUs       |                |                     | `adreno`               |

If no target is specified, then a safe but more limited default will be used.

!!! note

    We don't support the full spectrum of GPUs here and it is impossible to
    capture all details of a Vulkan implementation with a target triple, given
    the allowed variances on extensions, properties, limits, etc. So the target
    triple is just an approximation for usage. This is more of a mechanism to
    help us develop IREE itself. In the long term we want to perform
    multi-targetting to generate code for multiple architectures if no explicit
    target is given.

### :octicons-terminal-16: Run a compiled program

To run the compiled program:

``` shell hl_lines="2"
iree-run-module \
    --device=vulkan \
    --module=mobilenet_vulkan.vmfb \
    --function=torch-jit-export \
    --input="1x3x224x224xf32=0"
```

The above assumes the exported function in the model is named `torch-jit-export`
and it expects one 224x224 RGB image. We are feeding in an image with all 0
values here for brevity, see `iree-run-module --help` for the format to specify
concrete values.

<!-- TODO(??): Vulkan profiles / API versions / extensions -->

<!-- TODO(??): measuring performance -->

<!-- TODO(??): troubleshooting -->
