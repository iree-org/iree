# HAL Driver Features

Heterogeneity is one of IREE's core design principles. IREE aims to support
various accelerators for compute, ranging from general purpose CPUs, GPUs, to
other special purpose accelerators. IREE provides a
[Hardware Abstraction Layer (HAL)][iree-hal] as a common interface to these
accelerators. IREE exposes it via an [C API][iree-hal-c-api] for programmers and
an MLIR [dialect][iree-hal-dialect] for compilers.

Heterogeneity inevitably means IREE needs to provide a solution for managing
different features on different accelerators and their availability. This doc
describes the designs and mechanisms.

## General HAL Driver Features

IREE uses compilers to generate native code for each accelerator, serialize the
native code, and embed the code in one flat byte code following FlatBuffer
encoding format. The native code embedded in the final FlatBuffer file will
indicate the target architecture and required feature sets. At runtime IREE
selects a HAL driver meeting all the requirements to dispatch the workload to.

[TODO: describe the HAL functionality, C API, and dialect abstraction]

## Vulkan HAL Driver Features

Vulkan has many mechanisms for supporting different hardware implementations:
versions, extensions, features, limits. Vulkan uses SPIR-V to express the GPU
program but Vulkan is just one client SPIR-V supports. So SPIR-V has its own
mechanisms for supporting different clients: versions, capabilities, extensions.
The mechanims in these two domains bear lots of similarity, but they are not
exactly the same. We need to bridge these two worlds inside IREE.

IREE has its own [Vulkan dialect][iree-vulkan-dialect], which defines the Vulkan
target environment, including [versions][iree-vulkan-base],
[extensions][iree-vulkan-base], [features][iree-vulkan-cap-td]. These
definitions leverage MLIR attribute for storage, parsing/printing, and
validation. For example, we can have the following Vulkan target environment:

```
target_env = #vk.target_env<
  v1.1, r(120),
  [VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class],
  {
    maxComputeWorkGroupInvocations = 1024: i32,
    maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>
  }
>
```

The above describes a Vulkan implementation that supports specification version
1.1.120, supports `VK_KHR_spirv_1_4` and `VK_KHR_storage_buffer_storage_classs`
extensions, has a max compute workgroup invocation of 1024, and so on.

The above bears lots of similarity with the output of the
[`vulkaninfo`][vulkaninfo] utility. That's intended: `vulkaninfo` gives a
detailed dump of a Vulkan implementation by following the structures of all the
registered extensions to the specification. We pick relevant fields from it to
compose the list in the above to drive code generation. These are just different
formats for expressing the Vulkan implementation; one can image having a tool to
directly dump the MLIR attribute form used by IREE from the `vulkaninfo`'s JSON
dump.

When compiling ML models towards Vulkan, one specifies the target environment as
a `#vk.target_env` attribute assembly via the
[`iree-vulkan-target-env`][iree-vulkan-target-cl] command line option. At the
moment only one target environment is supported; in the future this is expected
to support multiple ones so that one can compile towards different Vulkan
implementations at once and embed all of them in the final FlatBuffer and select
at runtime.

Under the hood, this Vulkan target environment is then converted to the SPIR-V
target environment counterpart to drive code generation. The conversion happens
in one of Vulkan dialect's [utility function][iree-vulkan-target-conv]. The
converted SPIR-V target environment is [attached][iree-spirv-target-attach] to
the dispatch region's module for SPIR-V passes to use.

SPIR-V's target environment is very similar to the Vulkan target environment in
the above; it lives in upstream MLIR repo and is documented
[here][mlir-spirv-target] and implemented in SPIR-V dialect's
[`SPIRVAttribues.h`][mlir-spirv-attr] and
[`TargetAndABI.td`][mlir-spirv-target-td].

[iree-hal]: https://github.com/google/iree/tree/main/iree/hal
[iree-hal-c-api]: https://github.com/google/iree/blob/main/iree/hal/api.h
[iree-hal-dialect]: https://google.github.io/iree/Dialects/HALDialect
[iree-vulkan-dialect]: https://github.com/google/iree/tree/main/iree/compiler/Dialect/Vulkan
[iree-vulkan-base-td]: https://github.com/google/iree/blob/main/iree/compiler/Dialect/Vulkan/IR/VulkanBase.td
[iree-vulkan-cap-td]: https://github.com/google/iree/blob/main/iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.td
[iree-vulkan-target-cl]: https://github.com/google/iree/blob/b4739d704de15029cd671e53e7d7e743f4ca2e35/iree/compiler/Dialect/HAL/Target/VulkanSPIRV/VulkanSPIRVTarget.cpp#L66-L70
[iree-vulkan-target-conv]: https://github.com/google/iree/blob/b4739d704de15029cd671e53e7d7e743f4ca2e35/iree/compiler/Dialect/Vulkan/Utils/TargetEnvUtils.h#L29-L42
[iree-spirv-target-attach]: https://github.com/google/iree/blob/b4739d704de15029cd671e53e7d7e743f4ca2e35/iree/compiler/Dialect/HAL/Target/VulkanSPIRV/VulkanSPIRVTarget.cpp#L228-L240
[mlir-spirv-target]: https://mlir.llvm.org/docs/Dialects/SPIR-V/#target-environment
[mlir-spirv-attr]: https://github.com/llvm/llvm-project/blob/076305568cd6c7c02ceb9cfc35e1543153406d19/mlir/include/mlir/Dialect/SPIRV/SPIRVAttributes.h
[mlir-spirv-target-td]: https://github.com/llvm/llvm-project/blob/076305568cd6c7c02ceb9cfc35e1543153406d19/mlir/include/mlir/Dialect/SPIRV/TargetAndABI.td
[vulkaninfo]: https://vulkan.lunarg.com/doc/view/latest/linux/vulkaninfo.html
