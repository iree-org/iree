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

Next you will need to get an IREE runtime that includes the HIP HAL driver.

#### :material-hammer-wrench: Build the runtime from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build
IREE from source, then enable the HIP HAL driver with the `IREE_HAL_DRIVER_HIP`
option.

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
    --iree-hip-target=<...> \
    mobilenet_iree_input.mlir -o mobilenet_rocm.vmfb
```

Note that IREE comes with bundled bitcode files, which are used for linking
certain intrinsics on AMD GPUs. These will be used automatically or if the
`--iree-hip-bc-dir` is empty. As additional support may be needed for
different chips, users can use this flag to point to an explicit directory.
For example, in ROCm installations on Linux, this is often found under
`/opt/rocm/amdgcn/bitcode`.

Canonically a HIP target (`iree-hip-target`) matching the LLVM AMDGPU backend
of the form `gfx<arch_number>` is needed to compile towards each GPU chip.
If no target is specified then we will default to `gfx908`.

Here is a table of commonly used architectures:

| AMD GPU                  | Target Chip | Architecture Code Name
| ------------------------ | ----------- | ----------------------
| AMD MI100                | `gfx908`    | `cdna1`
| AMD MI210                | `gfx90a`    | `cdna2`
| AMD MI250                | `gfx90a`    | `cdna2`
| AMD MI300X (early units) | `gfx940`    | `cdna3`
| AMD MI300A (early units) | `gfx941`    | `cdna3`
| AMD MI300A               | `gfx942`    | `cdna3`
| AMD MI300X               | `gfx942`    | `cdna3`
| AMD RX7900XTX            | `gfx1100`   | `rdna3`
| AMD RX7900XT             | `gfx1100`   | `rdna3`
| AMD RX7800XT             | `gfx1101`   | `rdna3`
| AMD RX7700XT             | `gfx1101`   | `rdna3`

For a more comprehensive list of prior GPU generations, you can refer to the
[LLVM AMDGPU backend](https://llvm.org/docs/AMDGPUUsage.html#processors).

In addition to the canonical `gfx<arch_number>` scheme, `iree-hip-target` also
supports two additonal schemes to make a better developer experience:

* Architecture code names like `cdna3` or `rdna3`
* GPU product names like `mi300x` or `rx7900xtx`

These two schemes are translated into the canonical form under the hood.
We add support for common code/product names without aiming to be exhaustive.
If the ones you want are missing, please use the canonical form.

### :octicons-terminal-16: Run a compiled program

Run the following command:

``` shell hl_lines="2"
iree-run-module \
    --device=hip \
    --module=mobilenet_rocm.vmfb \
    --function=predict \
    --input="1x224x224x3xf32=0"
```

The above assumes the exported function in the model is named as `predict` and
it expects one 224x224 RGB image. We are feeding in an image with all 0 values
here for brevity, see `iree-run-module --help` for the format to specify
concrete values.
