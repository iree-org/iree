---
hide:
  - tags
tags:
  - GPU
  - CUDA
  - Metal
  - ROCm
  - Vulkan
icon: octicons/bug-16
---

# GPU debugging playbook

This page aims to provide general approaches and practical tips for debugging
GPU compiler/runtime correctness/performance issues in IREE.

GPUs fundamentally have similar architectures and software stacks.
We target GPUs from various vendors using different GPU APIs, but they share
quite a lot common infrastructure in IREE.
So the approaches and tips here should be widely applicable.

For the ones that are specific to a particular kind of problem/component/GPU,
they are prefixed with proper icons to be clear.
Here are what those icons represents--

Icon | Category
:--: | :------:
:material-check-bold: | Correctness
:material-trending-up: | Performance
:simple-amd: | AMD HIP/ROCm
:simple-apple: | Apple Metal
:simple-microsoft: | Microsoft DirectX
:simple-nvidia: | NVIDIA CUDA
:simple-vulkan: | Vulkan

## General methodology

The difficulties associated with debugging typically arise from **isolating the
problematic component and pinpointing the culprit**.
Once done, the solution typically derives naturally.

There are many components in the IREE stack; hierarchically we can categorize
them into either the compiler or runtime bucket:

* For compilers, there are multiple layers from the top to the bottom--frontend
  input importers, IREE flow/stream compilation, IREE host/device compilation,
  GPU backend in LLVM proper or GPU driver compiler for SPIR-V.
* For runtime, we have fewer layers--IREE HAL drivers, and GPU driver.

Any of the above components/layers can have bugs.
It's important to reduce the potential surface area to make the problem more
tractable.

Once we have a more isolated case, the general methodology to pinpoint the
exact culprit is to

1. **collect and inspect the symptoms,**
2. **form hypothesis and run experiments to prove/refute the hypothesis, and**
3. **iterate**.

### .. with shortcuts

The above procedure is for facing a large problem with no clue, for example,
when bringing up a new model end-to-end via IREE.

Though most of the time, we can leverage existing facilities to avoid going down
the full top-down hiearchical debugging procedure.
For example, for regression happening on an existing model, CI or `git bitsect`
might tell us directly the culprit commit.

### .. using tools

For issues with strong signals like crashing, it's also easier to pinpoint the
exact culprit with dedicated tools--we can leverage various
[sanitizers](./sanitizers.md) or debuggers.

## Isolating the problematic component

If we are facing a large problem without a clear clue, we need to isolate the
problematic compiler or runtime layer first, typically by comparing with a
working solution:

!!! tip "[correctness/performance]"

    Sanitize the environment first.
    Asking these questions and making sure the environment is proper can save
    you hours of debugging sometimes:

    * Did you recently updated the GPU SDK or driver?
    * Are others able to reproduce the issue?
    * If not what SDK / driver versions they are using?
    * Is your machine drawing enough power when benchmarking?
    * Is your machine connected with a mointor (e.g., for Vulkan)?
    * How long since you last rebooted your machine? 👻

!!! tip "[correctness/performance]"

    We have multiple GPU targets/drivers in IREE--LLVMGPU/CUDA, LLVMGPU/HIP,
    SPIR-V/Vulkan, SPIR-V/Metal.

    For the _same_ GPU, we typically have two paths to target, e.g., CUDA/HIP
    or Vulkan for NVIDIA/AMD GPUs, Metal or Vulkan for Apple GPUs.

    If one path is correct/performant, we can diff against it to try isolate
    the problem--the common/shared compiler/runtime code is likely okay; what
    differs between paths is likely problematic.

!!! tip "[correctness/performance] [vulkan]"

    Vulkan supports different GPUs.
    Similarly, if one GPU gives correct/performant result, we diff against it
    to find clues.

    Even more code in compiler/runtime are shared here; what's problematic is
    likely different capabilities triggering different CodeGen pipelines so
    revealing bugs in a particular CodeGen pipeline.
    Or there are driver issues from a particular vendor.

!!! tip "[correctness]"

    If the CPU is working properly, we can use the same dispatch region formation
    and diff against the CPU dispatches one by one to isolate the problem. See
    [this issue](https://github.com/openxla/iree/issues/14739) as an example.

!!! tip "[correctness]"

    `--iree-flow-trace-dispatch-tensors` and/or `--iree-flow-break-dispatch=` to
    `iree-compile` is quite helpful to inspect the output after all/each
    dispatch(es).

!!! tip "[correctness]"

    `iree-reduce` is a great tool to reduce and isolate issues programmatically.
    See more details [here](https://github.com/openxla/iree/blob/main/samples/reducer/README.md).

## Pinpointing compiler issues

Once we identified that the problem is due to some compiler issue, we can
investigate by comparing with different paths and inputs:

!!! tip "[correctness]"

    For the same dispatch, we may have different CodeGen pipelines, e.g.,
    for matmul we can have simple SIMT pipeline or using tensor/matrix cores.
    We can try to switch between different pipelines to isolate the problem.

!!! tip "[correctness]"

    Assuming we have a small repro, we can also try to see if there are
    "patterns" in the wrong result (e.g.,
    [this issue](https://github.com/openxla/iree/issues/14739#issuecomment-1685149869)).
    Or mutate the input to see if the failure has some "consistency".

!!! tip "[correctness/performance]"

    `--mlir-print-ir-*` and `--debug*` to `iree-opt` is our best friend.
    Sometimes it just takes eyeballing the IRs between stages to find clues.

!!! tip "[performance]"

    For identifying performance issues, we typically need to use:

    * [Tracy profiling](../performance/profiling-with-tracy.md) to get a
      course-grained command-buffer timing to understand what's the most
      time-consuming kernels.
      Typical big performance issues include but not limit to going down a
      incorrect CodeGen pipeline, missing tiling/vectorization, having an
      improper tiling/vectorization configuration, and so on.
      If the course-grained information is not enough, then we need to
    * [vendor-specific tools](../performance/profiling-gpu-vulkan.md) to
      understand kernel internal counters to identify the bottleneck.

## Pinpointing runtime issues

On the other side, if we suspect that it's a runtime issue, here are some
useful approachs and tips:

!!! tip "[correctness/performance]"

    [Tracy profiling](../performance/profiling-with-tracy.md) is a great way to
    view how the application runs dynamically.
    It can help to show problematic GPU API call sequences and performance
    bottlenecks.

    * It requires adding `-DIREE_ENABLE_RUNTIME_TRACING=ON` during CMake
      configuration, or use the `IREE_PY_RUNTIME=tracy` environment variable
      when invoking IREE runtime installed via Python packages.

!!! tip "[correctness]"

    GPU validation can sometimes give us hints:

    * [:simple-apple:] Enable validation via `export METAL_DEVICE_WRAPPER_TYPE=1`.
    * [:simple-vulkan:] Use `--vulkan_validation_layers=true` to `iree-run-module`, or
    * [:simple-vulkan:] Force enable via environment variables to the Vulkan loader:
      `export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_standard_validation`
      (may additionally need
      `export VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d` and
      `export LD_LIBRARY_PATH=$VULKAN_SDK/lib` if Vulkan SDK is not installed
      to a system path).

!!! tip "[correctness]"

    Turning on verbose output can give us more information:

    * When compiling IREE runtime, add
      `-DCMAKE_C_FLAGS=-DIREE_VM_EXECUTION_TRACING_FORCE_ENABLE=1` in CMake
      configuration to enable VM op tracing.
    * [:simple-vulkan:] Use `--vulkan_debug_verbosity=4` to `iree-run-module`.
    * [:simple-vulkan:] Print all Vulkan APIs calls with detailed arguments:
      `export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_api_dump`
      (may additionally need
      `export VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d` and
      `export LD_LIBRARY_PATH=$VULKAN_SDK/lib` if Vulkan SDK is not installed
      to a system path).

!!! tip "[correctness]"

    Try different "debugging modes" provided by HAL drivers:

    * [:simple-nvidia:] Switch `--cuda_use_streams=` between `true` and `false` to
      `iree-run-module` to see whether the issue comes from the stream/graph
      command buffer implementation.
    * [:simple-nvidia:] Switch `--cuda_async_allocations=false` to `iree-run-module` to
      see if the issue comes from async allocation.
    * [:simple-apple:] Use `--metal_serial_command_dispatch=true`,
      `--metal_command_buffer_retain_resources=true`, or
      `--metal_resource_hazard_tracking=true` to `iree-run-module` to see
      if any of the above "fixes" the issue.
      It can help to isolate the pontential problem.
    * [:simple-vulkan:] Use `--vulkan_robust_buffer_access=true` to `iree-run-module`
      especially when seeing undeterministic/corrupted contents in buffers and
      suspecting there are buffer allocation/indexing issues.
