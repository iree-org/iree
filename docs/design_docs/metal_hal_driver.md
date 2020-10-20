---
layout: default
permalink: design-docs/metal-hal-driver
title: "Metal HAL Driver"
parent: Design Docs
---

# Metal HAL Driver
{: .no_toc }

This document lists technical details regarding the Metal HAL driver. Note that
the Metal HAL driver is working in progress; this document is expected to be
updated along the way.

IREE provides a [Hardware Abstraction Layer (HAL)][iree-hal] as a common
interface to different compute accelerators. IREE HAL's design draws inspiration
from modern GPU architecture and APIs; so implementing a HAL driver using modern
GPU APIs is generally straightforward. This applies to the Metal HAL driver.

## Overall Design Choices

### Metal Versions

The Metal HAL driver expects Metal 2+. Metal 2 introduces useful features like
argument buffer, performance shaders, and others, that can improve performance
and make IREE HAL implementation simpler. Metal 2 was released late 2017 and are
supported since macOS High Sierra and iOS 11. It is already dominant
([macOS][macos-version-share], [iOS][ios-version-share]) right now.

### Programming Languages and Libraries

The Metal HAL driver lives under the [`iree/hal/metal/`][iree-metal] directory.
Header (`.h`) and implementation (`.mm`) files are put adjacent to each other.

The Metal framework only exposes Objective-C or Swift programming language APIs.
Metal HAL driver needs to inherit from common HAL abstraction classes, which are
C++. So we use [Objective-C++][objcxx] for implementing the Metal HAL driver.
The headers try to stay with pure C/C++ syntax when possible, except for
`#import <Metal/Metal.h>` and using Metal `id` types.

### Object Lifetime Management

Objective-C uses refcount for tracking object lifetime and managing memory. This
is traditionally done manually by sending `retain` and `release` messages to
Objective-C objects. Modern Objective-C allows developers to opt in to use
[Automatic Reference Counting][objc-arc] to let the compiler to automatically
deduce and insert `retain`/`release` where possible to simplify the burdern of
manual management.

We don't use ARC in the Metal HAL driver given that IREE has its own object
[refcount][iree-refptr] and lifetime management mechanism. Metal HAL GPU objects
are tracked with that to be consistent with others. Each Metal HAL GPU object
`retain`s the underlying Metal `id<MTL*>` object on construction and `release`s
on destruction.

## GPU Objects

Metal is one of the main modern GPU APIs that provide more explicit control over
the hardware. The mapping between IREE HAL classes and Metal protocols are
relatively straightforward:

IREE HAL Class                             | Metal Protocol
:----------------------------------------: | :------------:
[`hal::Driver`][hal-driver]                | N/A
[`hal::Device`][hal-device]                | [`MTLDevice`][mtl-device]
[`hal::CommandQueue`][hal-command-queue]   | [`MTLCommandQueue`][mtl-command-queue]
[`hal::CommandBuffer`][hal-command-buffer] | [`MTLCommandBuffer`][mtl-command-buffer]
[`hal::Semaphore`][hal-semaphore]          | [`MTLSharedEvent`][mtl-shared-event]
[`hal::Allocator`][hal-allocator]          | N/A
[`hal::Buffer`][hal-buffer]                | [`MTLBuffer`][mtl-buffer]
[`hal::Executable`][hal-executable]        | [`MTLLibrary`][mtl-library]
[`hal::ExecutableCache`][hal-executable-cache] | N/A
[`hal::DescriptorSetLayout`][hal-descriptor-set-layout] | N/A
[`hal::DescriptorSet`][hal-descriptor-set] | N/A
[`hal::ExecutableLayout`][hal-executable-layout] | N/A

In the following subsections, we go over each pair to provide more details.

### Driver

There is no native driver abstraction in Metal. IREE's Metal HAL driver still
provides a [`hal::metal::MetalDriver`][metal-driver] subclass inheriting from
common [`hal::Driver`][hal-driver] class. `hal::metal::MetalDriver` just
`retain`s all available Metal devices in the system during its lifetime to
provide similar interface as other HAL drivers.

### Device

[`hal::metal::MetalDevice`][metal-device] inherits [`hal::Device`][hal-device]
to provide the interface to Metal GPU device by wrapping a `id<MTLDevice>`. Upon
construction, `hal::metal::MetalDevice` creates and retains one queue for both
dispatch and transfer during its lifetime.

Metal requres command buffers to be created from a `MTLCommandQueue`. In IREE
HAL, command buffers are directly created from the `hal::Device`.
`hal::metal::MetalDevice` chooses the proper queue to create the command buffer
under the hood.

### Command queue

IREE HAL command queue follows Vulkan for modelling submission. Specifically,
`hal::CommandQueue::Submit()` takes a `SubmissionBatch`, which contains a list
of waiting `hal::Semaphore`s, a list of command buffers, and a list signaling
`hal::Semaphore`s. There is no direct mapping in Metal; so
[`hal::metal::MetalCommandQueue`][metal-command-queue] performs the submission
in three steps:

1.  Create a new `MTLCommandBuffer` to `encodeWaitForEvent:value` for all
    waiting `hal::Semaphore`s and commit this command buffer.
1.  Commit all command buffers in the `SubmissionBatch`.
1.  Create a new `MTLCommandBuffer` to `encodeSignalEvent:value` for all
    signaling `hal::Semaphore`s and commit this command buffer.

There is also no direct `WaitIdle()` for
[`MTLCommandQueue`][mtl-command-queue]s. `hal::metal::MetalCommandQueue`
implements `WaitIdle()` by committing an empty `MTLCommandBuffer` and
registering a complete handler for it to signal a semaphore to wake the current
thread, which is put into sleep by waiting on the semaphore.

### Command buffer

In Metal, commands are recorded into a command buffer with three different kinds
of [command encoders][mtl-command-encoder]: `MTLRenderCommandEncoder`,
`MTLComputeCommandEncoder`, `MTLBlitCommandEncoder`, and
`MTLParallelRenderCommandEncoder`. Each encoder has its own create/end call.
There is no overall begin/end call for the whold command buffer. So even
[`hal::metal::MetalCommandBuffer`][metal-command-buffer] implements an overall
`Begin()`/`End()` call, under the hood it may create a new command encoder for a
specific API call.

### Timeline semaphore

[`hal::Semaphore`][hal-semaphore] allows host->device, device->host, host->host,
and device->device synchronization. It maps to Vulkan timeline semaphore. In
Metal world, the counterpart would be [`MTLSharedEvent`][mtl-shared-event]. Most
of the `hal::Semaphore` APIs are simple to implement in
[`MetalSharedEvent`][metal-shared-event], with `Wait()` as an exception. A
listener is registered on the `MTLSharedEvent` with
`notifyListener:atValue:block:` to singal a semaphore to wake the current
thread, which is put into sleep by waiting on the semaphore.

### Allocator

At the moment the Metal HAL driver just has a very simple
[`hal::Allocator`][hal-allocator] implementation. It just wraps a `MTLDevice`
and redirects all allocation requests to the `MTLDevice`. No page/pool/slab or
whatever. This is only meant to get started. In the future we should have a
better memory allocation library, probably by layering the
[Vulkan Memory Allocator][vma] on top of [`MTLHeap`][mtl-heap].

### Buffer

IREE [`hal::Buffer`][hal-buffer] maps Metal `MTLBuffer`. See
[Memory Management](#memory-management) for more details.

### Executable

IREE [`hal::Executable`][hal-executable] represents a GPU program archive with
a driver-defined format. It maps naturally to Metal [`MTLLibrary`][mtl-library].
An entry point in a `MTLLibrary` is a [`MTLFunction`][mtl-function]. We define
[`hal::metal::MetalKernelLibrary`][metal-kernel-library] to wrap around a
`MTLLibrary`, its `MTLFunction`s, and also `MTLComputePipelineState` objects
constructed from `MTLFunction`s.

### Executable cache

IREE [`hal::ExecutableCache`][hal-executable-cache] is modelling a cache of
preprared GPU executables for a particular device. At the moment the Metal
HAL driver does not peforming any cache on GPU programs; it simply reads the
program from the FlatBuffer and hands it over to Metal driver.

### DescriptorSetLayout, DescriptorSet, ExecutableLayout

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
[SPIR-V code generation pipeline][spirv-codegen] and then cross compile the
generated SPIR-V into MSL source with [SPIRV-Cross][spirv-cross]. This is
actually a fair common practice for targeting multiple GPU APIs in graphics
programming world. For example, the Vulkan implmenation in macOS/iOs,
[MoltenVK][moltenvk], is also doing the same for shaders/kernels. The path
is actually quite robust, as demonstrated by various games on top of MoltenVK.

Therefore, in IREE, we have a [`MetalSPIRVTargetBackend`][metal-spirv-target],
which pulls in the normal MHLO to Linalg and Linalg to SPIR-V passes to form
the compilation pipeline. The difference would be to provide a suitable
SPIR-V target environment to drive the compilation, which one can derive from
the Metal GPU families to target. (Not implemented yet; TODO for the future.)
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
module FlatBuffer. Right now we just encode the MSL source strings and compile
them at Metal run-time. In the future this should be changed to allow encoding
the library instead.

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
the compute kernel. IREE's HAL is inspired by the Vulkan API; it models several
concepts related to GPU resource management explicitly:

* [`hal::DescriptorSetLayout`][hal-descriptor-set-layout]: a schema for
  describing an array of descriptor bindings. Each descriptor binding specifies
  the resource type, access mode and other information.
* [`hal::DescriptorSet`][hal-descriptor-set]: a concrete set of resources that
  gets bound to a compute pipeline in a batch. It must match the
  `DescriptorSetLayout` describing its layout. `DescriptorSet` can be thought as
  the "object" from the `DescriptorSetLayout` "class".
* [`hal::ExecutableLayout`][hal-executable-layout]: a schema for describing all
  the resources accessed by a compute pipeline. It includes zero or more
  `DescriptorSetLayout`s and (optional) push constants.

One can create `DescriptorSetLayout`, `DescriptorSet`, and `ExecutableLayout`
objects beforehand to avoid incurring overhead during tight computing loops
and also amortize costs by sharing these objects. However, this isn't totally
matching Metal's paradigm.

In the Metal framework, the closest concept to `DescriptorSet` would be [argument
buffer][mtl-argument-buffer]. There is no direct correspondence to
`DescriptorSetLayout` and `ExecutableLayout`. Rather, the layout is implicitly
encoded in Metal shaders as MSL structs. The APIs for creating argument buffers
do not encourage early creation without pipelines: one typically creates them
for each `MTLFunction`. Besides, unlike Vulkan where different descriptor sets
can have the same binding number, in Metal even if we have multiple argument
buffers, the indices for resources are in the same namespace and are typically
assigned sequentially. That means we need to remap `DescriptorSet`s with a set
number greater than zero by applying an offset to each of its bindings.

All of this means it's better to defer the creation of the argument buffer
until the point of compute pipeline creation and dispatch. Therefore, although
the Metal HAL driver provides the implementation for `DescriptorSet`
(i.e., `hal::metal::MetalArgumentBuffer`), `DescriptorSetLayout` (i.e.,
`hal::metal::MetalArgumentBufferLayout`), and `ExecutableLayout` (i.e.,
`hal::metal::MetalPipelineArgumentBufferLayout`), they are just containers
holding the information up until the [command buffer
dispatch](#command-buffer-dispatch) time.

With the above said, the overall idea is still to map one descriptor set to one
argument buffer. It just means we need to condense and remap the bindings.

### Command buffer dispatch

`MetalCommandBuffer::Dispatch()` performs the following steps with the current
active `MTLComputeCommandEncoder`:

1. Bind the `MTLComputePipelineState` for the current entry function queried
   from `MetalKernelLibrary`.
1. For each bound descriptor set at set #`S`:
   1. Create a [`MTLArgumentEncoder`][mtl-argument-encoder] for encoding an
      associated argument `MTLBuffer`.
   1. For each bound resource buffer at binding #`B` in this descriptor set,
      encode it to the argument buffer index #`B` with
      `setBuffer::offset::atIndex:` and inform the `MTLComputeCommandEncoder`
      that the dispatch will use this resource with `useResource:usage:`.
  1. Set the argument `MTLBuffer` to buffer index #`S`.
1. Dispatch with `dispatchThreadgroups:threadsPerThreadgroup:`.

(TODO: condense and remap bindings)

## Memory Management

### Storage type

Metal provides four [`MTLStorageMode`][mtl-storage-mode] options:

*   `MTLStorageModeShared`: The resource is stored in system memory and is
    accessible to both the CPU and the GPU.
*   `MTLStorageModeManaged`: The CPU and GPU may maintain separate copies of the
    resource, and any changes must be explicitly synchronized.
*   `MTLStorageModePrivate`: The resource can be accessed only by the GPU.
*   `MTLStorageMemoryless`: The resource’s contents can be accessed only by the
    GPU and only exist temporarily during a render pass.

Among them, `MTLStorageModeManaged` is only available on macOS.

IREE HAL defines serveral [`MemoryType`][hal-buffer]. They need to map to the
above storage modes:

*   If `kDeviceLocal` but not `kHostVisible`, `MTLStorageModePrivate` is chosen.
*   If `kDeviceLocal` and `kHostVisible`:
    *   If macOS, `MTLStorageModeManaged` can be chosen.
    *   Otherwise, `MTLStorageModeShared` is chosen.
*   If not `DeviceLocal` but `kDeviceVisible`, `MTLStorageModeShared` is chosen.
*   If not `kDeviceLocal` and not `kDeviceVisible`, `MTLStorageModeShared` is
    chosen. (TODO: We should probably use host buffer here.)

IREE HAL also allows to create buffers with `kHostCoherent` bit. This may still
be backed by `MTLStorageModeManaged` `MTLBuffer`s in macOS. To respect the
`kHostCoherent` protocol, the Metal HAL driver will perform necessary
`InValidate`/`Flush` operations automatically under the hood.

[macos-version-share]: https://gs.statcounter.com/macos-version-market-share/desktop/worldwide
[ios-version-share]: https://developer.apple.com/support/app-store/
[iree-hal]: https://github.com/google/iree/tree/main/iree/hal
[iree-metal]: https://github.com/google/iree/tree/main/iree/hal/metal
[iree-refptr]: https://github.com/google/iree/blob/main/iree/base/ref_ptr.h
[hal-allocator]: https://github.com/google/iree/blob/main/iree/hal/allocator.h
[hal-buffer]: https://github.com/google/iree/blob/main/iree/hal/buffer.h
[hal-command-queue]: https://github.com/google/iree/blob/main/iree/hal/command_queue.h
[hal-command-buffer]: https://github.com/google/iree/blob/main/iree/hal/command_buffer.h
[hal-descriptor-set]: https://github.com/google/iree/blob/main/iree/hal/descriptor_set.h
[hal-descriptor-set-layout]: https://github.com/google/iree/blob/main/iree/hal/descriptor_set_layout.h
[hal-executable-layout]: https://github.com/google/iree/blob/main/iree/hal/executable_layout.h
[hal-device]: https://github.com/google/iree/blob/main/iree/hal/device.h
[hal-driver]: https://github.com/google/iree/blob/main/iree/hal/driver.h
[hal-executable]: https://github.com/google/iree/blob/main/iree/hal/executable.h
[hal-executable-cache]: https://github.com/google/iree/blob/main/iree/hal/executable_cache.h
[hal-semaphore]: https://github.com/google/iree/blob/main/iree/hal/semaphore.h
[metal-command-queue]: https://github.com/google/iree/blob/main/iree/hal/metal/metal_command_queue.h
[metal-command-buffer]: https://github.com/google/iree/blob/main/iree/hal/metal/metal_command_buffer.h
[metal-device]: https://github.com/google/iree/blob/main/iree/hal/metal/metal_device.h
[metal-driver]: https://github.com/google/iree/blob/main/iree/hal/metal/metal_driver.h
[metal-kernel-library]: https://github.com/google/iree/blob/main/iree/hal/metal/metal_kernel_library.h
[metal-shared-event]: https://github.com/google/iree/blob/main/iree/hal/metal/metal_shared_event.h
[metal-spirv-target]: https://github.com/google/iree/tree/hal-metal/iree/compiler/Dialect/HAL/Target/MetalSPIRV
[mtl-argument-buffer]: https://developer.apple.com/documentation/metal/buffers/about_argument_buffers?language=objc
[mtl-argument-encoder]: https://developer.apple.com/documentation/metal/mtlargumentencoder?language=objc
[mtl-buffer]: https://developer.apple.com/documentation/metal/mtlbuffer?language=objc
[mtl-command-buffer]: https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc
[mtl-command-encoder]: https://developer.apple.com/documentation/metal/mtlcommandencoder?language=objc
[mtl-command-queue]: https://developer.apple.com/documentation/metal/mtlcommandqueue?language=objc
[mtl-device]: https://developer.apple.com/documentation/metal/mtldevice?language=objc
[mtl-function]: https://developer.apple.com/documentation/metal/mtlfunction?language=objc
[mtl-heap]: https://developer.apple.com/documentation/metal/mtlheap?language=objc
[mtl-library]: https://developer.apple.com/documentation/metal/mtllibrary?language=objc
[mtl-shared-event]: https://developer.apple.com/documentation/metal/mtlsharedevent?language=objc
[mtl-storage-mode]: https://developer.apple.com/documentation/metal/mtlstoragemode?language=objc
[msl-spec]: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
[msl-cl-library]: https://developer.apple.com/documentation/metal/libraries/building_a_library_with_metal_s_command-line_tools?language=objc
[objc-arc]: https://en.wikipedia.org/wiki/Automatic_Reference_Counting
[objcxx]: https://en.wikipedia.org/wiki/Objective-C#Objective-C++
[flatbuffer]: https://google.github.io/flatbuffers/
[mmap]: https://en.wikipedia.org/wiki/Mmap
[moltenvk]: https://github.com/KhronosGroup/MoltenVK
[spirv-codegen]: https://google.github.io/iree/design-docs/codegen-passes
[spirv-cross]: https://github.com/KhronosGroup/SPIRV-Cross
[vma]: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
[vulkan-spirv-target]: https://github.com/google/iree/tree/hal-metal/iree/compiler/Dialect/HAL/Target/VulkanSPIRV
[vulkan-cmd-dispatch]: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDispatch.html