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
[ROCm](https://www.amd.com/en/products/software/rocm.html).

## :octicons-download-16: Prerequisites

In order to use ROCm to drive the GPU, you need to have a functional ROCm
environment. It can be verified by the following steps:

=== ":fontawesome-brands-linux: Linux"

    Linux users can run the
    [ROCmInfo](https://rocm.docs.amd.com/projects/rocminfo/en/latest/how-to/use-rocminfo.html)
    tool:

    ``` console
    $ rocminfo

    ROCk module is loaded
    =====================
    HSA System Attributes
    =====================
    Runtime Version:         1.13
    Runtime Ext Version:     1.4

    ...
    ```

    If `rocminfo` does not exist, you will need to install the latest
    [ROCm for Linux](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html).

    !!! tip

        The
        [`rocm-smi`](https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/)
        tool and the more recent
        [`amd-smi`](https://rocm.docs.amd.com/projects/amdsmi/en/latest/) tool
        can also show information about your ROCm installation.

=== ":fontawesome-brands-windows: Windows"

    Windows users can run `hipInfo.exe` to check their environment:

    ``` console
    $ hipInfo.exe

    --------------------------------------------------------------------------------
    device#                           0
    Name:                             AMD Radeon PRO W7900 Dual Slot
    pciBusID:                         35
    pciDeviceID:                      0
    pciDomainID:                      0
    multiProcessorCount:              48

    ...
    ```

    If `hipInfo.exe` does not exist, you will need to install the latest
    [HIP SDK for Windows](https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html).

### Get the IREE compiler

#### :octicons-download-16: Download the compiler from a release

Python packages are distributed through multiple channels. See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core [`iree-base-compiler`](https://pypi.org/project/iree-base-compiler/)
package includes the ROCm compiler:

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-compiler-from-release.md"

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

#### :octicons-download-16: Download the runtime from a release

Python packages are distributed through multiple channels. See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core [`iree-base-runtime`](https://pypi.org/project/iree-base-runtime/)
package includes the HIP HAL driver:

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-runtime-from-release.md"

#### :material-hammer-wrench: Build the runtime from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build
IREE from source, then enable the HIP HAL driver with the `IREE_HAL_DRIVER_HIP`
option.

#### :octicons-checklist-24: Check for HIP devices

You can check for HIP support by looking for a matching driver and device:

```console hl_lines="9"
--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-driver-list.md:2"
```

```console hl_lines="3"
--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-device-list-amd.md"
```

To see device details, including the architecture to use as a
[HIP target](#choosing-hip-targets) when
[compiling a program](#compile-and-run-a-program):

```console hl_lines="9-10 15"
$ iree-run-module --dump_devices

...
# ============================================================================
# Enumerated devices for driver 'hip'
# ============================================================================

# ===----------------------------------------------------------------------===
# --device=hip://GPU-00000000-1111-2222-3333-444444444444
#   AMD Radeon PRO W7900 Dual Slot
# ===----------------------------------------------------------------------===

- amdhip64_dylib_path: /opt/rocm-6.1.3/lib/libamdhip64.so
- gpu-compute-capability: 11.0
- gpu-arch-name: gfx1100
```

## Compile and run a program

With the compiler and runtime ready, we can now compile programs and run them
on GPUs.

### :octicons-file-code-16: Compile a program

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-import-onnx-mobilenet.md"

Then run the following command to compile with the `rocm` target backend:

```shell hl_lines="7-8"
# You need to specify a HIP target for your GPU. For this example we extract
# the architecture name for the first device using rocm_agent_enumerator.
# See the section below for more context and alternatives.
IREE_HIP_TARGET=$(rocm_agent_enumerator | sed -n '2 p')  # e.g. gfx1100

iree-compile \
    --iree-hal-target-device=hip \
    --iree-hip-target=${IREE_HIP_TARGET} \
    mobilenetv2.mlir -o mobilenet_hip.vmfb
```

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-optimization-options.md"

???+ tip "Tip - HIP bitcode files"

    That IREE comes with bundled bitcode files, which are used for linking
    certain intrinsics on AMD GPUs. These will be used automatically or if the
    `--iree-hip-bc-dir` is empty. As additional support may be needed for
    different chips, users can use this flag to point to an explicit directory.
    For example, in ROCm installations on Linux, this is often found under
    `/opt/rocm/amdgcn/bitcode`.

#### Choosing HIP targets

A HIP target (`iree-hip-target`) matching the LLVM AMDGPU backend is needed
to compile towards each GPU chip. Here is a table of commonly used
architectures:

| AMD GPU                  | SKU Name    | Target Architecture | Architecture Code Name |
| ------------------------ | ----------- | ------------------- | ---------------------- |
| AMD MI100                | `mi100`     | `gfx908`            | `cdna1`                |
| AMD MI210                | `mi210`     | `gfx90a`            | `cdna2`                |
| AMD MI250                | `mi250`     | `gfx90a`            | `cdna2`                |
| AMD MI300A               | `mi300a`    | `gfx942`            | `cdna3`                |
| AMD MI300X               | `mi300x`    | `gfx942`            | `cdna3`                |
| AMD MI308X               | `mi308x`    | `gfx942`            | `cdna3`                |
| AMD MI325X               | `mi325x`    | `gfx942`            | `cdna3`                |
| AMD PRO V710             | `v710`      | `gfx1101`           | `rdna3`                |
| AMD PRO W7700            | `w7700`     | `gfx1101`           | `rdna3`                |
| AMD PRO W7800            | `w7800`     | `gfx1100`           | `rdna3`                |
| AMD PRO W7900            | `w7900`     | `gfx1100`           | `rdna3`                |
| AMD RX 7700XT            | `rx7700xt`  | `gfx1101`           | `rdna3`                |
| AMD RX 7800XT            | `rx7800xt`  | `gfx1101`           | `rdna3`                |
| AMD RX 7900XT            | `rx7900xt`  | `gfx1100`           | `rdna3`                |
| AMD RX 7900XTX           | `rx7900xtx` | `gfx1100`           | `rdna3`                |
| AMD RX 9060XT            | `rx9060xt`  | `gfx1200`           | `rdna4`                |
| AMD RX 9070              | `rx9070`    | `gfx1201`           | `rdna4`                |
| AMD RX 9070XT            | `rx9070xt`  | `gfx1201`           | `rdna4`                |
| AMD AI PRO R9070         | `r9070`     | `gfx1201`           | `rdna4`                |

For a more comprehensive list of prior GPU generations, you can refer to the
[LLVM AMDGPU backend](https://llvm.org/docs/AMDGPUUsage.html#processors).

The `iree-hip-target` option support three schemes:

1. The exact GPU product name (SKU), e.g., `--iree-hip-target=mi300x`. This
    allows the compiler to know about both the target architecture and about
    additional hardware details like the number of compute units. This extra
    information guides some compiler heuristics and allows for SKU-specific
    [tuning specs](../../reference/tuning.md).
2. The GPU architecture, as defined by LLVM, e.g.,
    `--iree-hip-target=gfx942`. This scheme allows for architecture-specific
    [tuning specs](../../reference/tuning.md) only.
3. The architecture code name, e.g., `--iree-hip-target=cdna3`. This scheme
    gets translated to closes matching GPU architecture under the hood.

We support for common code/SKU names without aiming to be exhaustive. If the
ones you want are missing, please use the GPU architecture scheme (2.) as it
is the most general.

!!! tip "Tip - querying target information from devices"

    In the example above we used
    [`rocm_agent_enumerator`](https://rocm.docs.amd.com/projects/rocminfo/en/latest/how-to/use-rocm-agent-enumerator.html)
    to get target chip information for the first GPU device on our system. This
    information can be queried in several ways:

    * The `amdgpu-arch` program on Linux and Windows:

        ```bash
        amdgpu-arch

        # gfx1100
        # gfx1100
        ```

    * The
      [`rocm_agent_enumerator`](https://rocm.docs.amd.com/projects/rocminfo/en/latest/how-to/use-rocm-agent-enumerator.html)
      program on Linux:

        ```bash
        rocm_agent_enumerator

        # gfx000
        # gfx1100
        # gfx1100
        ```

    * IREE's [`iree-run-module`](../../developers/general/developer-overview.md#iree-run-module)
      tool:

        ```bash
        # See information about all devices:
        iree-run-module --dump_devices

        # Grep for the first device target:
        iree-run-module --dump_devices | grep -m 1 -oP "gpu-arch-name: \K.*"
        # gfx1100
        ```

    * The
      [ROCmInfo](https://rocm.docs.amd.com/projects/rocminfo/en/latest/how-to/use-rocminfo.html)
      program on Linux:

        ```bash
        rocminfo

        # Look for e.g. gfx1100 in the output:

        # ...
        # *******
        # Agent 2
        # *******
        #   Name:                    gfx1100
        #   Uuid:                    GPU-1111111111111111
        #   Marketing Name:          AMD Radeon PRO W7900 Dual Slot
        # ...
        ```

### :octicons-terminal-16: Run a compiled program

To run the compiled program:

``` shell hl_lines="2"
iree-run-module \
    --device=hip \
    --module=mobilenet_rocm.vmfb \
    --function=torch-jit-export \
    --input="1x3x224x224xf32=0"
```

The above assumes the exported function in the model is named `torch-jit-export`
and it expects one 224x224 RGB image. We are feeding in an image with all 0
values here for brevity, see `iree-run-module --help` for the format to specify
concrete values.
