---
hide:
  - tags
tags:
  - GPU
# TODO(scotttodd): use a square icon (mark, no text?) instead of this wide one?
icon: simple/amd
---

# GPU deployment using ROCm

IREE can accelerate model execution on AMD GPUs using
[ROCm](https://www.amd.com/en/graphics/servers-solutions-rocm).

## :octicons-download-16: Prerequisites

In order to use ROCm to drive the GPU, you need to have a functional ROCm
environment. It can be verified by the following steps:

``` shell
rocm-smi | grep rocm
```

If `rocm-smi` does not exist, you will need to install the latest ROCm Toolkit
SDK for
[Windows](https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html)
or [Linux](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html).

### Get the IREE compiler

#### :octicons-package-16: Download the compiler from a release

!!! note "Currently ROCm is **NOT supported** for the Python interface."

#### :material-hammer-wrench: Build the compiler from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build
the IREE compiler, then enable the ROCm compiler target with the
`IREE_TARGET_BACKEND_ROCM` option.

!!! tip
    `iree-compile` will be built under the `iree-build/tools/` directory. You
    may want to include this path in your system's `PATH` environment variable.

### Get the IREE runtime

Next you will need to get an IREE runtime that includes the ROCm HAL driver.

#### :material-hammer-wrench: Build the runtime from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build
IREE from source, then enable the experimental ROCm HAL driver with the
`IREE_EXTERNAL_HAL_DRIVERS=rocm` option.

## Compile and run a program model

With the compiler and runtime ready, we can now compile programs and run them
on GPUs.

### :octicons-file-code-16: Compile a program

The IREE compiler transforms a model into its final deployable format in many
sequential steps. A model authored with Python in an ML framework should use the
corresponding framework's import tool to convert into a format (i.e.,
[MLIR](https://mlir.llvm.org/)) expected by the IREE compiler first.

Using MobileNet v2 as an example, you can download the SavedModel with trained
weights from
[TensorFlow Hub](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification)
and convert it using IREE's
[TensorFlow importer](../ml-frameworks/tensorflow.md). Then run one of the
following commands to compile:

```shell hl_lines="2-5"
iree-compile \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target-chip=<...> \
    --iree-rocm-link-bc=true \
    --iree-rocm-bc-dir=<...> \
    mobilenet_iree_input.mlir -o mobilenet_rocm.vmfb
```

Note ROCm Bitcode Dir (`iree-rocm-bc-dir`) path is required. If the system
you are compiling IREE in has ROCm installed, then the default value of
`/opt/rocm/amdgcn/bitcode` will usually suffice. If you intend on building
ROCm compiler in a non-ROCm capable system, please set `iree-rocm-bc-dir`
to the absolute path where you might have saved the amdgcn bitcode.

Note that a ROCm target chip (`iree-rocm-target-chip`) of the form
`gfx<arch_number>` is needed to compile towards each GPU architecture. If
no architecture is specified then we will default to `gfx908`.

Here is a table of commonly used architectures:

| AMD GPU   | Target Chip |
| --------- | ----------- |
| AMD MI25  | `gfx900`    |
| AMD MI50  | `gfx906`    |
| AMD MI60  | `gfx906`    |
| AMD MI100 | `gfx908`    |

### :octicons-terminal-16: Run a compiled program

Run the following command:

``` shell hl_lines="2"
iree-run-module \
    --device=rocm \
    --module=mobilenet_rocm.vmfb \
    --function=predict \
    --input="1x224x224x3xf32=0"
```

The above assumes the exported function in the model is named as `predict` and
it expects one 224x224 RGB image. We are feeding in an image with all 0 values
here for brevity, see `iree-run-module --help` for the format to specify
concrete values.
