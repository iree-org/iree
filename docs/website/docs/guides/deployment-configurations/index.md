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
* [:simple-amd: GPU - ROCm](./gpu-rocm.md)
  for AMD-specific solutions
* [:simple-nvidia: GPU - CUDA](./gpu-cuda.md)
  for NVIDIA-specific solutions
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

When compiling programs, a list of target backends can be specified via

* `--iree-hal-target-backends=` (command-line)
* `target_backends=[...]` (Python)

| Target backend | Description | Compatible HAL devices |
| -------------- | ----------- | ---------------------- |
| `llvm-cpu` | Code generation for CPU-like devices supported by LLVM | `local-sync`, `local-task` |
| `vmvx` | Portable interpreter powered by a microkernel library | `local-sync`, `local-task` |
| `vulkan-spirv` | Portable GPU support via SPIR-V for Vulkan | `vulkan` |
| `rocm` | AMD GPU support via HSACO for HIP | `hip` |
| `cuda` | NVIDIA GPU support via PTX for CUDA | `cuda` |
| `metal-spirv` | GPU support on Apple platforms via MSL for Metal | `metal` |
| `webgpu-spirv` | **Experimental** <br> GPU support on the Web via WGSL for WebGPU | `webgpu` |

### Listing available backends

The list of compiler target backends can be queried:

=== "Command-line"

    ```console
    $ iree-compile --iree-hal-list-target-backends

    Registered target backends:
        cuda
        llvm-cpu
        metal-spirv
        rocm
        vmvx
        vmvx-inline
        vulkan-spirv
    ```

=== "Python bindings"

    ```python
    import iree.compiler as ireec

    ireec.query_available_targets()
    # ['cuda', 'llvm-cpu', 'metal-spirv', 'rocm', 'vmvx', 'vmvx-inline', 'vulkan-spirv']
    ```

## Runtime HAL drivers and devices

Runtime HAL drivers can be used to enumerate and create HAL devices.

Runtime HAL devices call into hardware APIs to load and run executable code.
Devices may use multithreading or other system resources, depending on their
focus and the build configuration.

| HAL device   | Description |
| ------------ | ----------- |
| `local-sync` | Synchronous local CPU device with inline execution |
| `local-task` | Multithreaded local CPU device using a 'task' executor |
| `amdgpu`     | **Experimental** <br> AMD GPU execution using HSA |
| `cuda`       | NVIDIA GPU execution using CUDA |
| `metal`      | GPU execution on Apple platforms using Metal |
| `hip`        | AMD GPU execution using HIP |
| `vulkan`     | Portable GPU execution using the Vulkan API |
| `webgpu`     | **Experimental** <br> GPU execution on the web using WebGPU |

!!! tip "Tip - External HAL drivers"

    Additional HAL drivers can also be defined out of tree via the
    `IREE_EXTERNAL_HAL_DRIVERS` CMake option.

### Listing available drivers and devices

The list of runtime HAL drivers and devices can be queried:

=== "Command-line"

    List drivers:

    ```console
    --8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-driver-list.md:2"
    ```

    List devices:

    ```console
    --8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-device-list-amd.md"
    ```

    Dump information about devices:

    ```console
    --8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-device-dump-amd.md:2"
    ```

=== "Python bindings"

    List drivers:

    ```python
    import iree.runtime as ireert

    ireert.system_setup.query_available_drivers()
    # ['cuda', 'hip', 'local-sync', 'local-task', 'vulkan']
    ```

    List devices:

    ```python
    import iree.runtime as ireert

    for driver_name in ireert.system_setup.query_available_drivers():
        print(driver_name)
        try:
            driver = ireert.get_driver(driver_name)
            device_infos = driver.query_available_devices()
            for device_info in device_infos:
                print(f"  {device_info}")
        except:
            print(f"  (failed to initialize)")

    # cuda
    #   (failed to initialize)
    # hip
    #   {'device_id': 1, 'path': 'GPU-00000000-1111-2222-3333-444444444444', 'name': 'AMD Radeon ...'}
    # local-sync
    #   {'device_id': 0, 'path': '', 'name': 'default'}
    # local-task
    #   {'device_id': 0, 'path': '', 'name': 'default'}
    # vulkan
    #   {'device_id': 1234, 'path': '00000000-1111-2222-3333-444444444444', 'name': 'AMD Radeon ...'}
    ```
