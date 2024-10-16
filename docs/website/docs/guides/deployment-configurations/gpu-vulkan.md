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

#### :octicons-package-16: Download the compiler from a release

Python packages are regularly published to
[PyPI](https://pypi.org/user/google-iree-pypi-deploy/). See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core `iree-compiler` package includes the SPIR-V compiler:

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-compiler-from-release.md"

#### :material-hammer-wrench: Build the compiler from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build
IREE for your host platform and the
[Android cross-compilation](../../building-from-source/android.md) page if you
are cross compiling for Android. The SPIR-V compiler backend is compiled in by
default on all platforms.

Ensure that the `IREE_TARGET_BACKEND_VULKAN_SPIRV` CMake option is `ON` when
configuring for the host.

!!! tip
    `iree-compile` will be built under the `iree-build/tools/` directory. You
    may want to include this path in your system's `PATH` environment variable.

### Get the IREE runtime

Next you will need to get an IREE runtime that supports the Vulkan HAL driver.

You can check for Vulkan support by looking for a matching driver and device:

```console hl_lines="7"
--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-driver-list.md"
```

```console hl_lines="6"
$ iree-run-module --list_devices

  cuda://GPU-00000000-1111-2222-3333-444444444444
  local-sync://
  local-task://
  vulkan://00000000-1111-2222-3333-444444444444
```

#### :octicons-package-16: Download the runtime from a release

Python packages are regularly published to
[PyPI](https://pypi.org/user/google-iree-pypi-deploy/). See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core `iree-runtime` package includes the Vulkan HAL drivers:

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-runtime-from-release.md"

#### :material-hammer-wrench: Build the runtime from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build
IREE for Linux/Windows and the
[Android cross-compilation](../../building-from-source/android.md) page for
Android. The Vulkan HAL driver is compiled in by default on non-Apple platforms.

Ensure that the `IREE_HAL_DRIVER_VULKAN` CMake option is `ON` when configuring
for the target.

## Compile and run a program

With the SPIR-V compiler and Vulkan runtime, we can now compile programs and run
them on GPUs.

### :octicons-file-code-16: Compile a program

The IREE compiler transforms a model into its final deployable format in many
sequential steps. A model authored with Python in an ML framework should use the
corresponding framework's import tool to convert into a format (i.e.,
[MLIR](https://mlir.llvm.org/)) expected by the IREE compiler first.

Using MobileNet v2 as an example, you can download the SavedModel with trained
weights from
[TensorFlow Hub](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification)
and convert it using IREE's
[TensorFlow importer](../ml-frameworks/tensorflow.md). Then run the following
command to compile with the `vulkan-spirv` target:

``` shell hl_lines="2 3"
iree-compile \
    --iree-hal-target-backends=vulkan-spirv \
    --iree-vulkan-target=<...> \
    mobilenet_iree_input.mlir -o mobilenet_vulkan.vmfb
```

`iree-vulkan-target` specifies the GPU architecture to target. It accepts a few
schemes:

* LLVM CodeGen backend style: this is using LLVM AMDGPU/NVPTX CodeGen targets
  like `gfx1100` for AMD RX 7900XTX and `sm_86` for NVIDIA RTX 3090 GPUs.
* Architecture code name style: e.g., using `rdna3`/`valhall4`/`ampere`/`adreno`
  for AMD/ARM/NVIDIA/Qualcomm GPUs.
* Product name style(1): e.g., using `rx7900xtx`/`a100` for corresponding GPUs.

Here are a few examples showing how you can target various recent common GPUs:

| GPU                 | Target Architecture | Architecture Code Name | Product Name
| ------------------- | ------------------- | ---------------------- | ------------
| AMD RX7900XTX       | `gfx1100`           | `rdna3`                | `rx7900xtx`
| AMD RX7900XT        | `gfx1100`           | `rdna3`                | `rx7900xt`
| AMD RX7800XT        | `gfx1101`           | `rdna3`                | `rx7800xt`
| AMD RX7700XT        | `gfx1101`           | `rdna3`                | `rx7700xt`
| AMD RX6000 series   |                     | `rdna2`                |
| AMD RX5000 series   |                     | `rdna1`                |
| ARM Mali G715       |                     | `valhall4`             | e.g., `mali-g715`
| ARM Mali G510       |                     | `valhall3`             | e.g., `mali-g510`
| ARM GPUs            |                     | `valhall`              |
| NVIDIA RTX40 series | `sm_89`             | `ada`                  | e.g., `rtx4090`
| NVIDIA RTX30 series | `sm_86`             | `ampere`               | e.g., `rtx3080ti`
| NVIDIA RTX20 series | `sm_75`             | `turing`               | e.g., `rtx2070super`
| Qualcomm GPUs       |                     | `adreno`               |

If no target is specified, then a safe but more limited default will be used.

!!! note annotate
    Note that We don't support the full spectrum of GPUs here(2).
    This is more of a mechanism to help us develop IREE itself--in the long term
    we want to perform multiple targetting to generate to multiple architectures
    if no target is given.

1. Note that we only support very limited GPUs that we are actively developing
   against in this category, particularly for desktops.
2. It's also impossible to capture all details of a Vulkan implementation
   with a target triple, given the allowed variances on extensions, properties,
   limits, etc. So the target triple is just an approximation for usage.

### :octicons-terminal-16: Run a compiled program

In the build directory, run the following command:

``` shell hl_lines="2"
tools/iree-run-module \
    --device=vulkan \
    --module=mobilenet_vulkan.vmfb \
    --function=predict \
    --input="1x224x224x3xf32=0"
```

The above assumes the exported function in the model is named as `predict` and
it expects one 224x224 RGB image. We are feeding in an image with all 0 values
here for brevity, see `iree-run-module --help` for the format to specify
concrete values.

<!-- TODO(??): Vulkan profiles / API versions / extensions -->

<!-- TODO(??): measuring performance -->

<!-- TODO(??): troubleshooting -->
