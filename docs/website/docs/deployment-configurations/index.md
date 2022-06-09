# Deployment configurations

IREE provides a flexible set of tools for various deployment scenarios.
Fully featured environments can use IREE to load programs on demand and to take
advantage of multi-threaded hardware, while embedded systems can bypass IREE's
runtime entirely or interface with custom accelerators.

## Stable configurations

* [CPU](./cpu.md) for general purpose CPU deployment
* [CPU - Bare-Metal](./bare-metal.md) with minimal platform dependencies
* [GPU - Vulkan](./gpu-vulkan.md)
* [GPU - CUDA/ROCm](./gpu-cuda-rocm.md)

These are just the most stable configurations IREE supports. Feel free to reach
out on any of IREE's
[communication channels](../index.md#communication-channels) if you have
questions about a specific platform, hardware accelerator, or set of system
features.
