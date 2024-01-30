---
hide:
  - tags
tags:
  - GPU
  - Metal
---

# Metal HAL Driver

This document lists technical details regarding the Metal implemenation of
IREE's Hardware Abstraction Layer, called a Metal HAL driver.

IREE provides a [Hardware Abstraction Layer (HAL)][iree-hal] as a common
interface to different compute accelerators. IREE HAL's design draws inspiration
from modern GPU architecture and APIs; so implementing a HAL driver using modern
GPU APIs is generally straightforward. This applies to the Metal HAL driver.

## Overall Design Choices

### Metal Versions

Currently the Metal HAL driver expects Metal 3 capabilities. Metal 3 was
released late 2022 and are supported since macOS Ventura and iOS 16.
It [covers][metal-feature-set] recent Apple silicon GPUs including A13+ and M1+
chips and others.

In the future, we expect to increase the support to cover Metal 2 capabilities.
Metal 2 introduces useful features like argument buffer and others that are
necessary for performance and make IREE HAL implementation simpler. Metal 2 was
released late 2017 and are supported since macOS High Sierra and iOS 11. It is
already dominant ([macOS][macos-version-share], [iOS][ios-version-share]).

### Programming Languages and Libraries

The Metal framework only exposes Objective-C or Swift programming language APIs.
Metal HAL driver needs to inherit from common HAL abstraction definitions, which
are in C. To minimize dependency and binary size and increase performance, we
use Metal's Objective-C API for implementing the Metal HAL driver.
Header (`.h`) and implementation (`.m`) files are put adjacent to each other.

### Object Lifetime Management

Objective-C uses refcount for tracking object lifetime and managing memory. This
is traditionally done manually by sending `retain` and `release` messages to
Objective-C objects. Modern Objective-C allows developers to opt in to use
[Automatic Reference Counting][objc-arc] to let the compiler to automatically
deduce and insert `retain`/`release` where possible to simplify the burdern of
manual management.

We don't use ARC in the Metal HAL driver given that IREE has its own object
refcount and lifetime management mechanism. Metal HAL GPU objects are tracked
with that to be consistent with others. Each Metal HAL GPU object `retain`s
the underlying Metal `id<MTL*>` object on construction and `release`s on
destruction.

## GPU Objects

Metal is one of the main modern GPU APIs that provide more explicit control over
the hardware. The mapping between IREE HAL classes and Metal protocols are
relatively straightforward:

IREE HAL Class                                                  | Metal Protocol
:-------------------------------------------------------------: | :--------------------------------------:
[`iree_hal_driver_t`][hal-driver]                               | N/A
[`iree_hal_device_t`][hal-device]                               | [`MTLDevice`][mtl-device]
[`iree_hal_command_buffer_t`][hal-command-buffer]               | [`MTLCommandBuffer`][mtl-command-buffer]
[`iree_hal_semaphore_t`][hal-semaphore]                         | [`MTLSharedEvent`][mtl-shared-event]
[`iree_hal_allocator_t`][hal-allocator]                         | N/A
[`iree_hal_buffer_t`][hal-buffer]                               | [`MTLBuffer`][mtl-buffer]
[`iree_hal_executable_t`][hal-executable]                       | [`MTLLibrary`][mtl-library]
[`iree_hal_executable_cache_t`][hal-executable-cache]           | N/A
[`iree_hal_descriptor_set_layout_t`][hal-descriptor-set-layout] | N/A
[`iree_hal_pipeline_layout_t`][hal-pipeline-layout]             | N/A

In the following subsections, we go over each pair to provide more details.

### Driver

There is no native driver abstraction in Metal. IREE's Metal HAL driver still
provides a [`iree_hal_metal_driver_t`][metal-driver] struct to implement the
common [`iree_hal_driver_t`][hal-driver] struct. `iree_hal_metal_driver_t` just
`retain`s all available Metal devices in the system during its lifetime, to
guarantee that we have the same `id<MTLDevice>` for device querying and
creation.

### Device

[`iree_hal_metal_device_t`][metal-device] implements [`iree_hal_device_t`][hal-device]
to provide the interface to Metal GPU device by wrapping a `id<MTLDevice>`. Upon
construction, `iree_hal_metal_device_t` creates and retains one queue for both
dispatch and transfer during its lifetime. In the future we expect to spport
multiple queues for better concurrency.

#### Command buffer submission

In IREE HAL, command buffers are directly created from the `iree_hal_device_t`.
It's also directly submitted there via `iree_hal_device_queue_execute()`.
Each execution takes a batch of command buffers, together with a list of waiting
`iree_hal_semaphore_t`s and a list signaling `iree_hal_semaphore_t`s.
There is no direct mapping of such structure in Metal; so we performs the submission
in three steps:

1.  Create a new `MTLCommandBuffer` to `encodeWaitForEvent:value` for all
    waiting `iree_hal_semaphore_t`s and commit this command buffer.
1.  Commit all command buffers in the submmision batch.
1.  Create a new `MTLCommandBuffer` to `encodeSignalEvent:value` for all
    signaling `iree_hal_semaphore_t`s and commit this command buffer.

Such submission enables asynchronous execution of the workload on the GPU.

#### Queue-ordered allocation

Queue-ordered asynchronous allocations via `iree_hal_device_queue_alloc` is not fully
supported yet; it just translates to blocking wait and allocation.

#### Collectives

Collectives suppport is not yet implemented.

#### Profiling

The Metal HAL driver supports profiling via `MTLCaptureManager`. We can either
capture to a trace file or XCode.

To perform profiling in the command line, attach `--device_profiling_mode=queue
--device_profiling_file=/path/to/metal.gputrace` to IREE binaries.

### Command buffer

Command buffers are where IREE HAL and Metal API have a major difference.

IREE HAL command buffers follow the flat Vulkan recording model, where all memory
or dispatch commands are recorded into a command buffer directly.
Unlike Vulkan, Metal adopts a multi-level command recording model--memory/dispatch
commands are not directly recorded into a command buffer; rather, they must go
through the additional level of blit/compute encoders.
Implementing IREE's HAL using Metal would require switching encoders for
interleaved memory and dispatch commands.
Additionally, certain IREE HAL API features do not have direct mapping in Metal
APIs, e.g., various forms of IREE HAL execution/memory barriers. Translating
them would require looking at both previous and next commands to decide the
proper mapping.

Due to these reasons, it's beneficial to have a complete view of the full
command buffer and extra flexibility during recording, in order to fixup past
commands, or inspect future commands.

Therefore, to implement IREE HAL command buffers using Metal, we perform two
steps using a linked list of command segments:
First we create segments to keep track of all IREE HAL commands and the
associated data. And then, when finalizing the command buffer, we iterate
through all the segments and record their contents into a proper
`MTLCommandBuffer`. A linked list gives us the flexibility to organize
command sequence in low overhead; and a deferred recording gives us the
complete picture of the command buffer when really started recording.

The Metal HAL driver right now only support one-shot command buffers, by mapping
to `MTLCommandBuffer`s.

#### Fill/copy/update buffer

Metal APIs for fill and copy buffers have alignment restrictions on the offset
and length. `iree_hal_command_buffer_{fill|copy|update}_buffer()` is more
flexible regarding that. So for cases aren't directly supported by Metal APIs,
we use [polyfill compute kernels][metal-builtin-kernels] to perform the memory
operation using GPU threads.

### Semaphore

[`iree_hal_semaphore_t`][hal-semaphore] allows host->device, device->host, host->host,
and device->device synchronization. It maps to Vulkan timeline semaphore. In
Metal world, the counterpart would be [`MTLSharedEvent`][mtl-shared-event]. Most
of the `iree_hal_semaphore_t` APIs are simple to implement in
[`MetalSharedEvent`][metal-shared-event], with `iree_hal_semaphore_wait()` as an
exception. A listener is registered on the `MTLSharedEvent` with
`notifyListener:atValue:block:` to singal a semaphore to wake the current
thread, which is put into sleep by waiting on the semaphore.

### Allocator

At the moment the Metal HAL driver just has a very simple
[`iree_hal_allocator_t`][hal-allocator] implementation. It just wraps a `MTLDevice`
and redirects all allocation requests to the `MTLDevice`. No page/pool/slab or
whatever. This is meant to be used together with common allocator layers like the
caching allocator.

### Buffer

IREE [`iree_hal_buffer_t`][hal-buffer] maps Metal `MTLBuffer`. See
[Memory Management](#memory-management) for more details.

### Executable

IREE [`iree_hal_executable_t`][hal-executable] represents a GPU program archive with
a driver-defined format. It maps naturally to Metal [`MTLLibrary`][mtl-library].
An entry point in a `MTLLibrary` is a [`MTLFunction`][mtl-function]. We define
[`iree_hal_metal_kernel_params_t`][metal-kernel-library] to wrap around a
`MTLLibrary`, its `MTLFunction`s, and also `MTLComputePipelineState` objects
constructed from `MTLFunction`s.

### Executable cache

IREE [`iree_hal_executable_cache_t`][hal-executable-cache] is modeling a cache of
preprared GPU executables for a particular device. At the moment the Metal
HAL driver does not peforming any caching on GPU programs; it simply reads the
program from the FlatBuffer and hands it over to Metal driver.

### Descriptor set / pipeline layout

See [Resource descriptors](#resource-descriptors) for more details.

## Compute Pipeline

### Shader/kernel compilation

Metal has [Metal Shading Language (MSL)][msl-spec] for authoring graphics
shaders and compute kernels. MSL source code can be directly consumed by the
Metal framework at run-time; it can also be compiled first into an opaque
library using [command-line tools][msl-cl-library] at build-time.

IREE uses compilers to compile ML models expressed with high-level op semantics
down to GPU native source format. This is also the case for the Metal HAL
driver. Metal does not provide an open intermediate language; we reuse the
SPIR-V code generation pipeline and then cross compile the generated SPIR-V into
MSL source with [SPIRV-Cross][spirv-cross]. This is actually a fair common
practice for targeting multiple GPU APIs in graphics programming world. For
example, the Vulkan implmenation in macOS/iOS, [MoltenVK][moltenvk], is also
doing the same for shaders/kernels. The path is quite robust, as demonstrated
by various games on top of MoltenVK.

Therefore, in IREE, we have a [`MetalSPIRVTargetBackend`][metal-spirv-target],
which pulls in the common SPIR-V passes to form the compilation pipeline.
The difference would be to provide a suitable SPIR-V target environment to drive
the compilation, which one can derive from the Metal GPU families to target.
The serialization step differs from
[`VulkanSPIRVTargetBackend`][vulkan-spirv-target] too: following the normal
SPIR-V serialization step, we additionally need to invoke SPRIV-Cross to
cross compile the generated SPIR-V into MSL, and then compile and/or serialize
the MSL source/library.

IREE uses [FlatBuffer][flatbuffer] to encode the whole workload module,
including both GPU shader/kernel (called executable in IREE terminology) and
CPU scheduling logic. The GPU executables are embedded as part of the module's
FlatBuffer, which are [`mmap`][mmap]ped when IREE runs.

For the Metal HAL driver, it means we need to embed the MSL kernels inside the
module FlatBuffer. Right now we can either encode the MSL source strings and
compile them at Metal run-time, or directly encoding the library instead.

### Workgroup/threadgroup size

When dispatching a compute kernel in Metal, we need to specify the number of
thread groups in grid and the number of threads in thread group. Both are
3-D vectors. IREE HAL, which follows Vulkan, calls them workgroup count and
workgroup size, respectively.

In Vulkan programming model, workgroup count and workgroup size are specified at
different places: the former is given when invoking
[`vkCmdDispatch()`][vulkan-cmd-dispatch], while the later is encoded in the
dispatched SPIR-V code. This split does not match the Metal model, where we
specify both in the API with `dispatchThreads:threadsPerThreadgroup:`.

As said in [shader/kernel compilation](#shader-kernel-compilation), MSL kernels
are cross compiled from SPIR-V code and then embeded in the module FlatBuffer.
The module FlatBuffer provides us a way to convey the threadgroup/workgroup size
information extracted from the SPIR-V code. We encode an additional 3-D vector
for each entry point and use it as the threadgroup size when later dispatching
the `MTLFunction` corresponding to the entry point.

### Resource descriptors

A descriptor is an opaque handle pointing to a resource that is accessed in
the compute kernel. IREE's HAL models several concepts related to GPU resource
management explicitly:

* [`iree_hal_descriptor_set_layout_t`][hal-descriptor-set-layout]: a schema for
  describing an array of descriptor bindings. Each descriptor binding specifies
  the resource type, access mode and other information.
* [`iree_hal_pipeline_layout_t`][hal-pipeline-layout]: a schema for describing all
  the resources accessed by a compute pipeline. It includes zero or more
  `DescriptorSetLayout`s and (optional) push constants.

However, this isn't totally matching Metal's paradigm.
In the Metal framework, the closest concept to descriptor sets would be [argument
buffer][mtl-argument-buffer]. There is no direct correspondence to
descriptor set layout and pipeline layout. Rather, the layout is implicitly
encoded in Metal shaders as MSL structs. The APIs for creating argument buffers
do not encourage early creation without pipelines: one typically creates them
for each `MTLFunction`.

All of this means it's better to defer the creation of the argument buffer
until the point of compute pipeline creation and dispatch. Therefore, the Metal
HAL driver's `iree_hal_metal_descriptor_set_layout_t` and
`iree_hal_metal_pipeline_layout_t` are just containers holding the information
up for recording [command buffer dispatch](#command-buffer-dispatch).

### Command buffer dispatch

Metal HAL driver command buffer dispatch recording performs the following steps
with the current active `MTLComputeCommandEncoder`:

1. Bind the `MTLComputePipelineState` for the current entry function.
1. Encode the push constants using `setBytes:length:atIndex`.
1. For each bound descriptor set at set #`S`:
   1. Create a [`MTLArgumentEncoder`][mtl-argument-encoder] for encoding an
      associated argument `MTLBuffer`.
   1. For each bound resource buffer at binding #`B` in this descriptor set,
      encode it to the argument buffer index #`B` with
      `setBuffer::offset::atIndex:` and inform the `MTLComputeCommandEncoder`
      that the dispatch will use this resource with `useResource:usage:`.
   1. Set the argument `MTLBuffer` to buffer index #`S`.
1. Dispatch with `dispatchThreadgroups:threadsPerThreadgroup:`.

[metal-feature-set]: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
[macos-version-share]: https://gs.statcounter.com/macos-version-market-share/desktop/worldwide
[ios-version-share]: https://developer.apple.com/support/app-store/
[iree-hal]: https://github.com/openxla/iree/tree/main/runtime/src/iree/hal
[hal-allocator]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/allocator.h
[hal-buffer]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/buffer.h
[hal-command-buffer]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/command_buffer.h
[hal-descriptor-set-layout]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/pipeline_layout.h
[hal-pipeline-layout]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/pipeline_layout.h
[hal-device]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/device.h
[hal-driver]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/driver.h
[hal-executable]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/executable.h
[hal-executable-cache]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/executable_cache.h
[hal-semaphore]: https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/semaphore.h
[metal-device]: https://github.com/openxla/iree/tree/main/experimental/metal/metal_device.h
[metal-driver]: https://github.com/openxla/iree/tree/main/experimental/metal/metal_driver.h
[metal-kernel-library]: https://github.com/openxla/iree/tree/main/experimental/metal/kernel_library.h
[metal-shared-event]: https://github.com/openxla/iree/tree/main/experimental/metal/shared_event.h
[metal-spirv-target]: https://github.com/openxla/iree/tree/main/compiler/src/iree/compiler/Dialect/HAL/Target/MetalSPIRV
[metal-builtin-kernels]: https://github.com/openxla/iree/tree/main/runtime/src/iree/hal/drivers/metal/builtin/
[mtl-argument-buffer]: https://developer.apple.com/documentation/metal/buffers/about_argument_buffers?language=objc
[mtl-argument-encoder]: https://developer.apple.com/documentation/metal/mtlargumentencoder?language=objc
[mtl-buffer]: https://developer.apple.com/documentation/metal/mtlbuffer?language=objc
[mtl-command-buffer]: https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc
[mtl-command-encoder]: https://developer.apple.com/documentation/metal/mtlcommandencoder?language=objc
[mtl-device]: https://developer.apple.com/documentation/metal/mtldevice?language=objc
[mtl-function]: https://developer.apple.com/documentation/metal/mtlfunction?language=objc
[mtl-library]: https://developer.apple.com/documentation/metal/mtllibrary?language=objc
[mtl-shared-event]: https://developer.apple.com/documentation/metal/mtlsharedevent?language=objc
[mtl-storage-mode]: https://developer.apple.com/documentation/metal/mtlstoragemode?language=objc
[msl-spec]: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
[msl-cl-library]: https://developer.apple.com/documentation/metal/libraries/building_a_library_with_metal_s_command-line_tools?language=objc
[objc-arc]: https://en.wikipedia.org/wiki/Automatic_Reference_Counting
[flatbuffer]: https://google.github.io/flatbuffers/
[mmap]: https://en.wikipedia.org/wiki/Mmap
[moltenvk]: https://github.com/KhronosGroup/MoltenVK
[spirv-cross]: https://github.com/KhronosGroup/SPIRV-Cross
[vulkan-spirv-target]: https://github.com/openxla/iree/tree/main/compiler/src/iree/compiler/Dialect/HAL/Target/VulkanSPIRV
[vulkan-cmd-dispatch]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdDispatch.html
