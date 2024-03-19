# Deployment configurations

IREE provides a flexible set of tools for various deployment scenarios.
Fully featured environments can use IREE to load programs on demand and to take
advantage of multi-threaded hardware, while embedded systems can bypass IREE's
runtime entirely or interface with custom accelerators.

## Stable configurations

* [:octicons-cpu-16: CPU](./cpu.md) for general
  purpose CPU deployment
* [:octicons-cpu-16: CPU - Bare-Metal](./bare-metal.md)
  with minimal platform dependencies
* [:octicons-server-16: GPU - Vulkan](./gpu-vulkan.md)
  for cross-platform usage and interop with graphics applications
* [:simple-nvidia: GPU - CUDA](./gpu-cuda.md)
  for NVIDIA-specific solutions
* [:simple-amd: GPU - ROCm](./gpu-rocm.md)
  for AMD-specific solutions
* [:simple-apple: GPU - Metal](./gpu-metal.md)
  for running on Apple hardware

These are just the most stable configurations IREE supports. Feel free to reach
out on any of IREE's
[communication channels](../../index.md#communication-channels) if you have
questions about a specific platform, hardware accelerator, or set of system
features.

## Compiler target backends

Compiler target backends are used to generate executable code for hardware APIs
and device architectures. Compiler targets may implement special optimizations
or generate distinct code for certain device/architecture/performance profiles.

When compiling programs, a list of target backends must be specified via

* `--iree-hal-target-backends=` (command-line)
* `target_backends=[...]` (Python)

| Target backend | Description | Compatible HAL devices |
| -------------- | ----------- | ---------------------- |
| `llvm-cpu` | Code generation for CPU-like devices supported by LLVM | `local-sync`, `local-task` |
| `vmvx` | Portable interpreter powered by a microkernel library | `local-sync`, `local-task` |
| `vulkan-spirv` | Portable GPU support via SPIR-V for Vulkan | `vulkan` |
| `cuda` | NVIDIA GPU support via PTX for CUDA | `cuda` |
| `metal-spirv` | GPU support on Apple platforms via MSL for Metal | `metal` |
| `rocm` | **Experimental** <br> AMD GPU support via HSACO for ROCm | `rocm` |
| `webgpu-spirv` | **Experimental** <br> GPU support on the Web via WGSL for WebGPU | `webgpu` |

!!! tip "Tip - listing available backends"
    The list of compiler target backends can be queried:

    === "Command-line"

        ```console
        $ iree-compile --iree-hal-list-target-backends

        Registered target backends:
            cuda
            llvm-cpu
            metal
            metal-spirv
            rocm
            vmvx
            vmvx-inline
            vulkan
            vulkan-spirv
        ```

    === "Python bindings"

        ```python
        iree.compiler.query_available_targets()

        ['cuda',
         'llvm-cpu',
         'metal',
         'metal-spirv',
         'rocm',
         'vmvx',
         'vmvx-inline',
         'vulkan',
         'vulkan-spirv']
        ```

## Runtime HAL drivers/devices

Runtime HAL devices call into hardware APIs to load and run executable code.
Devices may use multithreading or other system resources, depending on their
focus and the build configuration.

| HAL device   | Description |
| ------------ | ----------- |
| `local-sync` | Synchronous local CPU device with inline execution |
| `local-task` | Multithreaded local CPU device using a 'task' executor |
| `vulkan`     | Portable GPU execution using the Vulkan API |
| `cuda`       | NVIDIA GPU execution using CUDA |
| `metal`      | GPU execution on Apple platforms using Metal |
| `rocm`       | **Experimental** <br> AMD GPU execution using ROCm |
| `webgpu`     | **Experimental** <br> GPU execution on the web using WebGPU |

Additional HAL drivers can also be defined external to the core project via
`IREE_EXTERNAL_HAL_DRIVERS`.
