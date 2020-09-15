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
[hal-device]: https://github.com/google/iree/blob/main/iree/hal/device.h
[hal-driver]: https://github.com/google/iree/blob/main/iree/hal/driver.h
[hal-semaphore]: https://github.com/google/iree/blob/main/iree/hal/semaphore.h
[metal-command-queue]: https://github.com/google/iree/blob/main/iree/hal/metal/metal_command_queue.h
[metal-command-buffer]: https://github.com/google/iree/blob/main/iree/hal/metal/metal_command_buffer.h
[metal-device]: https://github.com/google/iree/blob/main/iree/hal/metal/metal_device.h
[metal-driver]: https://github.com/google/iree/blob/main/iree/hal/metal/metal_driver.h
[metal-shared-event]: https://github.com/google/iree/blob/main/iree/hal/metal/metal_shared_event.h
[mtl-buffer]: https://developer.apple.com/documentation/metal/mtlbuffer?language=objc
[mtl-command-buffer]: https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc
[mtl-command-encoder]: https://developer.apple.com/documentation/metal/mtlcommandencoder?language=objc
[mtl-command-queue]: https://developer.apple.com/documentation/metal/mtlcommandqueue?language=objc
[mtl-device]: https://developer.apple.com/documentation/metal/mtldevice?language=objc
[mtl-heap]: https://developer.apple.com/documentation/metal/mtlheap?language=objc
[mtl-shared-event]: https://developer.apple.com/documentation/metal/mtlsharedevent?language=objc
[mtl-storage-mode]: https://developer.apple.com/documentation/metal/mtlstoragemode?language=objc
[objc-arc]: https://en.wikipedia.org/wiki/Automatic_Reference_Counting
[objcxx]: https://en.wikipedia.org/wiki/Objective-C#Objective-C++
[vma]: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator