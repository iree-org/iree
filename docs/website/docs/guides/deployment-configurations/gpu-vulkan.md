# Vulkan GPU HAL Driver

IREE can accelerate model execution on GPUs via
[Vulkan](https://www.khronos.org/vulkan/), a low-overhead graphics and compute
API. Vulkan is cross-platform: it is available on many operating systems,
including Android, Linux, and Windows. Vulkan is also cross-vendor: it is
supported by most GPU vendors, including AMD, ARM, Intel, NVIDIA, and Qualcomm.

<!-- TODO(??): when to use CPU vs GPU -->

## Support matrix

As IREE and the compiler ecosystem it operates within matures, more
target specific optimizations will be implemented. At this stage, expect
reasonable performance across all GPUs and for improvements to be
made over time for specific vendors and architectures.

GPU Vendor | Category | Performance | Focus Architecture
:--------: | :------: | :---------: | :----------------:
ARM Mali GPU | Mobile |  Good | Valhall
Qualcomm Adreno GPU | Mobile | Reasonable | 640+
AMD GPU | Desktop/server | Reasonable | -
NVIDIA GPU | Desktop/server | Reasonable | -

## Prerequisites

In order to use Vulkan to drive the GPU, you need to have a functional Vulkan
environment. IREE requires Vulkan 1.1 on Android and 1.2 elsewhere. It can be
verified by the following steps:

=== "Android"

    Android mandates Vulkan 1.1 support since Android 10. You just need to
    make sure the device's Android version is 10 or higher.

=== "Linux"

    Run the following command in a shell:

    ``` shell
    vulkaninfo | grep apiVersion
    ```

    If `vulkaninfo` does not exist, you will need to [install the latest Vulkan
    SDK](https://vulkan.lunarg.com/sdk/home/). For Ubuntu 18.04/20.04,
    installing via LunarG's package repository is recommended, as it places
    Vulkan libraries and tools under system paths so it's easy to discover.

    If the showed version is lower than Vulkan 1.2, you will need to update the
    driver for your GPU.

=== "Windows"

    Run the following command in a shell:

    ``` shell
    vulkaninfo | grep apiVersion
    ```

    If `vulkaninfo` does not exist, you will need to [install the latest Vulkan
    SDK](https://vulkan.lunarg.com/sdk/home/).

    If the showed version is lower than Vulkan 1.2, you will need to update the
    driver for your GPU.

## Get runtime and compiler

### Get IREE runtime with Vulkan HAL driver

Next you will need to get an IREE runtime that supports the Vulkan HAL driver
so it can execute the model on GPU via Vulkan.

<!-- TODO(??): vcpkg -->

#### Build runtime from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build IREE
for Linux/Windows and the
[Android cross-compilation](../../building-from-source/android.md) page for
Android. The Vulkan HAL driver is compiled in by default on non-Apple platforms.

<!-- TODO(??): a way to verify Vulkan is compiled in and supported -->

Ensure that the `IREE_HAL_DRIVER_VULKAN` CMake option is `ON` when configuring
for the target.

### Get compiler for SPIR-V exchange format

Vulkan expects the program running on GPU to be expressed by the
[SPIR-V](https://www.khronos.org/registry/spir-v/) binary exchange format, which
the model must be compiled into.

<!-- TODO(??): vcpkg -->

#### Download as Python package

Python packages for various IREE functionalities are regularly published
to [PyPI](https://pypi.org/user/google-iree-pypi-deploy/). See the
[Python Bindings](../../reference/bindings/python.md) page for more
details. The core `iree-compiler` package includes the SPIR-V compiler:

``` shell
python -m pip install iree-compiler
```

!!! tip
    `iree-compile` is installed to your python module installation path. If you
    pip install with the user mode, it is under `${HOME}/.local/bin`, or
    `%APPDATA%Python` on Windows. You may want to include the path in your
    system's `PATH` environment variable.

    ``` shell
    export PATH=${HOME}/.local/bin:${PATH}
    ```

#### Build compiler from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build IREE
for Linux/Windows and the
[Android cross-compilation](../../building-from-source/android.md) page for
Android. The SPIR-V compiler backend is compiled in by default on all platforms.

Ensure that the `IREE_TARGET_BACKEND_VULKAN_SPIRV` CMake option is `ON` when
configuring for the host.

## Compile and run the model

With the compiler for SPIR-V and runtime for Vulkan, we can now compile a model
and run it on the GPU.

### Compile the model

IREE compilers transform a model into its final deployable format in many
sequential steps. A model authored with Python in an ML framework should use the
corresponding framework's import tool to convert into a format (i.e.,
[MLIR](https://mlir.llvm.org/)) expected by main IREE compilers first.

Using MobileNet v2 as an example, you can download the SavedModel with trained
weights from
[TensorFlow Hub](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification)
and convert it using IREE's
[TensorFlow importer](../ml-frameworks/tensorflow.md). Then,

#### Compile using the command-line

Run the following command (passing `--iree-input-type=` as needed for your
import tool):

``` shell hl_lines="2 3"
iree-compile \
    --iree-hal-target-backends=vulkan-spirv \
    --iree-vulkan-target-triple=<...> \
    --iree-input-type=stablehlo \
    iree_input.mlir -o mobilenet-vulkan.vmfb
```

where `iree_input.mlir` is the imported program.

Note that a target triple of the form `<vendor/arch>-<product>-<os>` is needed
to compile towards each GPU architecture. If no triple is specified then a safe
but more limited default will be used. We don't support the full spectrum
here[^1]; the following table summarizes the
currently recognized ones:

GPU Vendor | Target Triple
:--------: | :-----------:
ARM Mali GPU | `valhall-g78-android30`
Qualcomm Adreno GPU | `adreno-unknown-android30`
AMD GPU | e.g., `rdna1-5700xt-linux`
NVIDIA GPU | e..g, `ampere-rtx3080-windows`
SwiftShader CPU | `cpu-swiftshader-unknown`

### Run the model

#### Run using the command-line

In the build directory, run the following command:

``` shell hl_lines="2"
tools/iree-run-module \
    --device=vulkan \
    --module=mobilenet-vulkan.vmfb \
    --function=predict \
    --input="1x224x224x3xf32=0"
```

The above assumes the exported function in the model is named as `predict` and
it expects one 224x224 RGB image. We are feeding in an image with all 0 values
here for brevity, see `iree-run-module --help` for the format to specify
concrete values.

<!-- TODO(??): Vulkan profiles / API versions / extensions -->

<!-- TODO(??): deployment options -->

<!-- TODO(??): measuring performance -->

<!-- TODO(??): troubleshooting -->

[^1]: It's also impossible to capture all details of a Vulkan implementation
with a target triple, given the allowed variances on extensions, properties,
limits, etc. So the target triple is just an approximation for usage.
