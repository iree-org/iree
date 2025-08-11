// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_BUFFER_H_
#define IREE_HAL_BUFFER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/queue.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_allocator_t iree_hal_allocator_t;
typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Whole length of the underlying buffer.
#define IREE_HAL_WHOLE_BUFFER ((iree_device_size_t)(-1))

// A bitfield specifying properties for a memory type.
enum iree_hal_memory_type_bits_t {
  IREE_HAL_MEMORY_TYPE_NONE = 0u,

  // The allocator will choose the optimal memory type based on buffer usage.
  // Allocation will succeed if there is a heap available for the allocator to
  // place the memory into.
  //
  // Additional bits can be provided to restrict the set of memory types that
  // are chosen. For example, if the user knows that a bulk of the accesses will
  // happen from device the IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL bit can be set to
  // force the allocator to place it on device.
  //
  // This bit is only used during allocation.
  IREE_HAL_MEMORY_TYPE_OPTIMAL = 1u << 0,

  // Memory allocated with this type can be mapped for host access using
  // iree_hal_buffer_map_range.
  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE = 1u << 1,

  // The host cache management commands iree_hal_buffer_mapping_flush_range and
  // iree_hal_buffer_mapping_invalidate_range are not needed to flush host
  // writes to the device or make device writes visible to the host,
  // respectively.
  IREE_HAL_MEMORY_TYPE_HOST_COHERENT = 1u << 2,

  // Memory allocated with this type is cached on the host. Host memory
  // accesses to uncached memory are slower than to cached memory, however
  // uncached memory is always host coherent. MappedMemory::Flush must be used
  // to ensure the device has visibility into any changes made on the host and
  // Invalidate must be used to ensure the host has visibility into any changes
  // made on the device.
  IREE_HAL_MEMORY_TYPE_HOST_CACHED = 1u << 3,

  // Memory is accessible as normal host allocated memory.
  IREE_HAL_MEMORY_TYPE_HOST_LOCAL = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                                    (1u << 6),

  // The allocator will choose the optimal memory type based on buffer usage,
  // preferring to place the allocation in host-local memory.
  //
  // Users should set this when it is known that a bulk of accesses to the
  // buffer will be done by the host, such as readback/download staging buffers.
  // It should be expected that device access will be slow.
  //
  // This bit is only used during allocation.
  // Allocations will fail if there is no host-local memory type that can
  // satisfy all requested usage.
  IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST =
      IREE_HAL_MEMORY_TYPE_OPTIMAL | IREE_HAL_MEMORY_TYPE_HOST_LOCAL,

  // Memory allocated with this type is visible to the device for execution.
  // Being device visible does not mean the memory must reside on device (as
  // does IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL). Though an allocation may be
  // visible to the device and therefore useable for execution it may require
  // expensive mapping or implicit transfers.
  IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE = 1u << 4,

  // Memory allocated with this type is the most efficient for device access.
  // Devices may support using memory that is not device local via
  // IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE but doing so can incur non-trivial
  // performance penalties. Device local memory, on the other hand, is
  // guaranteed to be fast for all operations.
  IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL =
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE | (1u << 5),

  // The allocator will choose the optimal memory type based on buffer usage,
  // preferring to place the allocation in device-local memory.
  //
  // Users should set this when it is known that a bulk of the accesses to the
  // buffer will be done by the device, such as device transfer and dispatch
  // operations or light usage of host -> device upload staging buffers.
  // It should be expected that host access will be slow.
  //
  // This bit is only used during allocation.
  // Allocations will fail if there is no host-local memory type that can
  // satisfy all requested usage.
  IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE =
      IREE_HAL_MEMORY_TYPE_OPTIMAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
};
typedef uint32_t iree_hal_memory_type_t;

// Parses a memory type bitfield from a string.
// See iree_bitfield_parse for usage.
IREE_API_EXPORT iree_status_t iree_hal_memory_type_parse(
    iree_string_view_t value, iree_hal_memory_type_t* out_value);

// Formats a memory type bitfield as a string.
// See iree_bitfield_format for usage.
IREE_API_EXPORT iree_string_view_t iree_hal_memory_type_format(
    iree_hal_memory_type_t value, iree_bitfield_string_temp_t* out_temp);

// A bitfield specifying how memory will be accessed in a mapped memory region.
enum iree_hal_memory_access_bits_t {
  // Memory is not mapped.
  IREE_HAL_MEMORY_ACCESS_NONE = 0u,
  // Memory will be read.
  // If a buffer is only mapped for reading it may still be possible to write to
  // it but the results will be undefined (as it may present coherency issues).
  IREE_HAL_MEMORY_ACCESS_READ = 1u << 0,
  // Memory will be written.
  // If a buffer is only mapped for writing it may still be possible to read
  // from it but the results will be undefined or incredibly slow (as it may
  // be mapped by the driver as uncached).
  IREE_HAL_MEMORY_ACCESS_WRITE = 1u << 1,
  // Memory will be discarded prior to mapping.
  // The existing contents will be undefined after mapping and must be written
  // to ensure validity.
  IREE_HAL_MEMORY_ACCESS_DISCARD = 1u << 2,
  // Memory will be discarded and completely overwritten in a single operation.
  IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE =
      IREE_HAL_MEMORY_ACCESS_WRITE | IREE_HAL_MEMORY_ACCESS_DISCARD,
  // A flag that can be applied to any access type to indicate that the buffer
  // storage being accessed may alias with other accesses occurring concurrently
  // within or across operations. The lack of the flag indicates that the access
  // is guaranteed not to alias (ala C's `restrict` keyword).
  IREE_HAL_MEMORY_ACCESS_MAY_ALIAS = 1u << 3,
  // A flag that can be applied to any access type to indicate that the buffer
  // storage may not be aligned.
  IREE_HAL_MEMORY_ACCESS_UNALIGNED = 1u << 4,
  // Memory access may perform any operation and should not be validated.
  // Used upon access to bypass access verification at the API boundary and
  // effectively provides a `void*`.
  // This should only be used by device-side code where it is known-safe to
  // bypass the access verification.
  IREE_HAL_MEMORY_ACCESS_ANY = 1u << 5,
  // Memory may have any operation performed on it.
  // Note that this explicitly includes 'DISCARD', which means that the
  // mapped memory will have undefined contents. Do not use this access
  // mode if you intend the existing contents to be accessible.
  IREE_HAL_MEMORY_ACCESS_ALL = IREE_HAL_MEMORY_ACCESS_READ |
                               IREE_HAL_MEMORY_ACCESS_WRITE |
                               IREE_HAL_MEMORY_ACCESS_DISCARD,
};
typedef uint16_t iree_hal_memory_access_t;

// Parses a memory access bitfield from a string.
// See iree_bitfield_parse for usage.
IREE_API_EXPORT iree_status_t iree_hal_memory_access_parse(
    iree_string_view_t value, iree_hal_memory_access_t* out_value);

// Formats a memory access bitfield as a string.
// See iree_bitfield_format for usage.
IREE_API_EXPORT iree_string_view_t iree_hal_memory_access_format(
    iree_hal_memory_access_t value, iree_bitfield_string_temp_t* out_temp);

// Bitfield that defines how a buffer is intended to be used.
// Usage allows the driver to appropriately place the buffer for more
// efficient operations of the specified types. Validation will fail if a buffer
// is used in ways other than it was declared it would be used in.
enum iree_hal_buffer_usage_bits_t {
  IREE_HAL_BUFFER_USAGE_NONE = 0u,

  // ==== IREE_HAL_BUFFER_USAGE_TRANSFER_* =====================================

  // Buffer is used as a source for transfer operations.
  // Buffer may be a source for:
  //   iree_hal_command_buffer_copy_buffer
  //
  // Maps to:
  //  - D3D12_RESOURCE_STATE_COPY_SOURCE
  //  - GPUBufferUsage.COPY_SRC
  //  - VK_BUFFER_USAGE_TRANSFER_SRC_BIT
  IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE = 1u << 0,

  // Buffer is used as a target for transfer operations.
  // Buffer may be a target for:
  //   iree_hal_command_buffer_fill_buffer
  //   iree_hal_command_buffer_update_buffer
  //   iree_hal_command_buffer_copy_buffer
  //
  // Maps to:
  //  - D3D12_RESOURCE_STATE_COPY_DEST
  //  - GPUBufferUsage.COPY_DST
  //  - VK_BUFFER_USAGE_TRANSFER_DST_BIT
  IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET = 1u << 1,

  // Buffer contents are transferred using DMA operations.
  // Buffer may be a source or target for:
  //   iree_hal_command_buffer_fill_buffer
  //   iree_hal_command_buffer_update_buffer
  //   iree_hal_command_buffer_copy_buffer
  //
  // Maps to:
  //  - D3D12_RESOURCE_STATE_COPY_SOURCE | D3D12_RESOURCE_STATE_COPY_DEST
  //  - GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  //  - VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
  IREE_HAL_BUFFER_USAGE_TRANSFER = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
                                   IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,

  // ==== IREE_HAL_BUFFER_USAGE_DISPATCH_* =====================================

  // Buffer contents are used for indirect dispatch workgroup parameters.
  // Dispatch parameters must be aligned on 16-byte boundaries and be of the
  // form `struct { uint32_t workgroup_xyz[3]; }`.
  //
  // Maps to:
  //  - D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
  //  - GPUBufferUsage.INDIRECT
  //  - MTLResourceUsageRead
  //  - VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
  IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS = 1u << 8,

  // Buffer contents are uniformly read by dispatches.
  // These may occasionally be written as storage buffers in cases of
  // data-dependent sequences but are largely read-only and may have total size
  // limitations (~32-64KB visible per binding).
  //
  // Uniform buffers can be used in place of push constants or as additional
  // storage for when push constant resources are exhausted. As push constants
  // must be recorded into the command buffer any values that may change if the
  // command buffer were to be reused should be put in a mutable uniform
  // buffer instead. By using uniform buffers (vs storage buffers) the hardware
  // can perform better caching and coalescing as they are guaranteed to be
  // read-only across all workgroups in a dispatch.
  //
  // Maps to:
  //  - D3D12_CONSTANT_BUFFER_VIEW_DESC
  //  - GPUBufferUsage.UNIFORM
  //  - MTLResourceUsageRead
  //  - VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
  IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ = 1u << 9,

  // Buffer contents are read by dispatches as storage buffers.
  // Read-only buffers can enable non-local prefetching and replication.
  IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ = 1u << 10,

  // Buffer contents are written by dispatches as storage buffers.
  // Write-only buffers can reduce cache pollution and writeback latency.
  IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_WRITE = 1u << 11,

  // Buffer contents are read and written by dispatches as storage buffers.
  // Storage buffers allow random read/write access to underlying data using
  // flexible data formats and alignment. Atomic operations may be allowed
  // depending on implementation.
  //
  // Maps to:
  //  - D3D12_UNORDERED_ACCESS_VIEW_DESC::D3D12_BUFFER_UAV
  //    + D3D12_RESOURCE_STATE_UNORDERED_ACCESS
  //  - GPUBufferUsage.STORAGE
  //  - MTLResourceUsageRead | MTLResourceUsageWrite
  //  - VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
  IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ |
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_WRITE,

  // Buffer contents are read by dispatches as images.
  // Depending on the implementation this may be ignored or treated the same as
  // IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ.
  IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_READ = 1u << 12,

  // Buffer contents are written by dispatches as images.
  // Depending on the implementation this may be ignored or treated the same as
  // IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_WRITE.
  IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_WRITE = 1u << 13,

  // Buffer contents are read and written by dispatches as images.
  // Depending on the implementation this may be ignored or treated the same as
  // IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE. If supported then additional
  // hardware resources may be required to perform the binding.
  //
  // Storage buffers are preferred in most cases due to the more flexible data
  // types and access allowance. On specific hardware images may allow use of
  // fixed-function sampling hardware and texture caches that are not available
  // on the storage buffer path. The largest benefit from using images is around
  // uniform loads (all invocations in a workgroup loading the same location) on
  // low-end hardware in order to tickle driver compiler optimizations. In
  // almost all other cases the image path can incur significant additional
  // latency or correctness hazards especially in data-dependent operations.
  //
  // Maps to:
  //  - D3D12_SHADER_RESOURCE_VIEW_DESC::D3D12_BUFFER_SRV
  //  - MTLResourceUsageRead | MTLResourceUsageWrite
  //  - VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT
  IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE =
      IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_READ |
      IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_WRITE,

  // Buffer contents are available for use by all dispatch-related operations.
  IREE_HAL_BUFFER_USAGE_DISPATCH =
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
      IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ |
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
      IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE,

  // ==== IREE_HAL_BUFFER_USAGE_SHARING_* ======================================

  // Buffer can be exported via iree_hal_allocator_export_buffer.
  // Exported buffers may require special allocation behavior (dedicated
  // allocations, higher alignment, etc) and may impose lifetime restrictions.
  IREE_HAL_BUFFER_USAGE_SHARING_EXPORT = 1u << 16,

  // Buffer can be automatically replicated across peers.
  // When multiple devices use the same buffer an implementation is allowed to
  // clone the buffer per-device in order to keep accesses local.
  // Implementations are free to ignore this flag if doing so would require
  // additional overhead (managed memory/global locking/etc).
  IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE = 1u << 17,

  // Buffer can be used concurrently by multiple queues.
  // This may incur additional implicit synchronization overhead.
  // When omitted the buffer is assumed to be exclusive to a queue and may not
  // be accessible from other queues without explicit transfer operations.
  IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT = 1u << 18,

  // Buffer is immutable once initialized and implementations are allowed to
  // avoid synchronization/transfers done in cases where the buffer may be
  // mutable. Implementations are allowed to protect the buffer contents for
  // read-only access if they support it.
  IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE = 1u << 19,

  // ==== IREE_HAL_BUFFER_USAGE_MAPPING_* ======================================

  // Buffer may be mapped for scoped host access.
  // Each iree_hal_buffer_map_range must be paired with an
  // iree_hal_buffer_unmap_range.
  //
  // Concurrent access across host and device are not allowed during scoped
  // mappings and will lead to desynchronization. If concurrent access is
  // required then persistent mapping can be used (if supported) and otherwise
  // staging buffers with transfer operations can preserve proper pipelining.
  IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED = 1u << 24,

  // Buffer may be mapped for persistent host access.
  // iree_hal_buffer_map_range _may_ be paired with a matching
  // iree_hal_buffer_unmap_range but it's not required.
  //
  // These mappings can persist for the lifetime of the buffer and allow for
  // concurrent usage from both host and device. Depending on the memory type
  // additional synchronization may be required via flushes and invalidation.
  // Not all implementations support this and of those that do many may place
  // such allocations in slower or more restrictive memory.
  IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT = 1u << 25,

  // Buffer is allowed to be mapped only if doing so is cheap.
  // If omitted and one of the other mapping usage flags is set the allocation
  // will fail if the buffer cannot be allocated in mapped memory. This can be
  // used to optimistically request mapping for staging buffers that avoids
  // additional allocations and copies. Code setting this flag is expected to
  // handle the transfers when mapping is not used by checking the allocated
  // buffer usage bits.
  IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL = 1u << 26,

  // Indicates that the mapped memory will be accessed by the host randomly.
  // Reads and writes will be non-contiguous or non-temporal and host-cached
  // memory is strongly preferred.
  IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM = 1u << 27,

  // Indicates that the mapped memory will be written sequentially (memcpy/etc).
  // The host will only write to the mapped memory and do so using a sequential
  // operation (memcpy, memset, etc). Reads may fail or be *extremely* slow.
  IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE = 1u << 28,

  // Buffer may be mapped for scoped random-access host access.
  // Allocation will fail if mapping is not available; if attempting to
  // optimistically map in order to avoid staging transfers then add the
  // IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL bit.
  //
  // WARNING: mapping can be extremely expensive, use limited hardware
  // resources, introduce data hazards, and synchronize host and device
  // execution. Unless an application knows that such issues will not arise
  // (as in tests where there's never concurrent usage) mapping should be used
  // judiciously: do not assume mapping is a high-performance technique!
  //
  // If an application knows its access characteristics (such as memcpy only)
  // then prefer specifying the bits directly and including
  // IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE. Random access (set
  // by default with this value) can severely harm device performance.
  IREE_HAL_BUFFER_USAGE_MAPPING = IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                                  IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM,

  // ==== IREE_HAL_BUFFER_USAGE_* helpers ======================================

  // Default usage mode covering transfer and dispatch.
  // Most internal buffers will be allocated for this usage and external buffers
  // should use this unless specific usage is required (such as mapping).
  IREE_HAL_BUFFER_USAGE_DEFAULT =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
};
typedef uint32_t iree_hal_buffer_usage_t;

// Parses a buffer usage bitfield from a string.
// See iree_bitfield_parse for usage.
IREE_API_EXPORT iree_status_t iree_hal_buffer_usage_parse(
    iree_string_view_t value, iree_hal_buffer_usage_t* out_value);

// Formats a buffer usage bitfield as a string.
// See iree_bitfield_format for usage.
IREE_API_EXPORT iree_string_view_t iree_hal_buffer_usage_format(
    iree_hal_buffer_usage_t value, iree_bitfield_string_temp_t* out_temp);

// Buffer overlap testing results.
typedef enum iree_hal_buffer_overlap_e {
  // No overlap between the two buffers.
  IREE_HAL_BUFFER_OVERLAP_DISJOINT = 0,
  // Partial overlap between the two buffers.
  IREE_HAL_BUFFER_OVERLAP_PARTIAL,
  // Complete overlap between the two buffers (they are the same).
  IREE_HAL_BUFFER_OVERLAP_COMPLETE,
} iree_hal_buffer_overlap_t;

typedef uint32_t iree_hal_buffer_compatibility_t;

// A bitfield specifying buffer transfer behavior.
enum iree_hal_transfer_buffer_flag_bits_t {
  // TODO(benvanik): flags controlling blocking, flushing, invalidation, and
  // persistence. We may also want to set a bit that causes failure on emulated
  // transfers that would otherwise be really expensive.
  IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT = 0,
};
typedef uint32_t iree_hal_transfer_buffer_flags_t;

// Determines buffer mapping behavior.
enum iree_hal_mapping_mode_bits_t {
  // Buffers are mapped as part of a scoped map-access-unmap sequence.
  // If there are any in-flight operations using the buffer contents are
  // undefined though they may deceivingly still seem correct under certain
  // implementations.
  IREE_HAL_MAPPING_MODE_SCOPED = 1u << 0,

  // Buffers are mapped persistently and concurrently accessible by both the
  // host and device. Mapping happens once and so long as there are any live
  // mappings the buffer will remain accessible. Not all implementations or
  // buffer memory types support this, and even ones that do may not support
  // coherent cross-device sharing.
  IREE_HAL_MAPPING_MODE_PERSISTENT = 1u << 1,
};
typedef uint32_t iree_hal_mapping_mode_t;

//===----------------------------------------------------------------------===//
// iree_hal_buffer_placement_t
//===----------------------------------------------------------------------===//

// Flags describing the placement of a buffer on a device and its allocation
// semantics. This information is only valid on allocated buffers and not
// wrappers that may hold references to them.
typedef uint32_t iree_hal_buffer_placement_flags_t;
enum iree_hal_buffer_placement_flag_bits_t {
  IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE = 0u,
  // Buffer was allocated with an asynchronous allocation API such as
  // iree_hal_device_queue_alloca and/or can be deallocated with an asynchronous
  // deallocation API such as iree_hal_device_queue_dealloca.
  IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS = 1u << 0,
  // Buffer lifetime is indeterminate indicating that the compiler or
  // application allocating the buffer is unable to determine when it is safe to
  // deallocate the buffer. Explicit deallocation requests are ignored and the
  // buffer deallocation will happen synchronously when the last remaining
  // reference to the buffer is released.
  IREE_HAL_BUFFER_PLACEMENT_FLAG_INDETERMINATE_LIFETIME = 1u << 1,
  // TODO(benvanik): flags for discrete/external to allow for quick export
  // checks.
};

// Describes the origin of an allocated buffer.
// This is used internally to route buffers back to pools and can be used by
// hosting layers to route deallocations to appropriate devices/queues.
// This information is generally only valid for allocated buffers (the result of
// an iree_hal_buffer_allocated_buffer query).
typedef struct iree_hal_buffer_placement_t {
  // The device the buffer was allocated from. Unretained.
  // Only valid for allocated buffers and not any intermediates (subspans, etc).
  // May be NULL if the buffer is not associated with any particular device such
  // as a free-floating heap-allocated buffer on the host.
  iree_hal_device_t* device;
  // Queues on the device to which the buffer is available. Depending on the
  // device this may indicate which queues have exclusive access to the buffer
  // or which queues have optimal access. This may be broader than the original
  // request if the buffer is able to be accessed by other queues without
  // penalty. Usage of the buffer for queue read/write or asynchronous
  // deallocation via iree_hal_device_queue_dealloca is only legal with a queue
  // affinity that is a subset of this affinity set.
  iree_hal_queue_affinity_t queue_affinity;
  // Describes the placement behavior of a buffer on a device and its allocation
  // semantics.
  iree_hal_buffer_placement_flags_t flags;
  uint32_t reserved;
} iree_hal_buffer_placement_t;

// Returns a placement indicating that the buffer has no direct device it is
// associated with. Commonly used for free-floating buffer handles such as heap
// wrapped or allocated buffers that come from outside of the HAL.
static inline iree_hal_buffer_placement_t iree_hal_buffer_placement_undefined(
    void) {
  iree_hal_buffer_placement_t placement = {0};
  return placement;
}

// Returns true if the |placement| is undefined and the buffer has no direct
// device it is associated with.
static inline bool iree_hal_buffer_placement_is_undefined(
    const iree_hal_buffer_placement_t placement) {
  return placement.device == NULL;
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_params_t
//===----------------------------------------------------------------------===//

// Parameters defining how a buffer should be allocated.
//
// Designed to be zero-initialized: any field with a 0 value will be assigned
// a default as indicated in the field description.
//
// For ergonomics when used from C++ w/o named initializers the first field is
// the most commonly used so that it can be initialized by location:
//    some_fn(..., {IREE_HAL_BUFFER_USAGE_FOO}, ...)
typedef struct iree_hal_buffer_params_t {
  // Specifies the usage allowed by HAL APIs and aids in memory placement.
  // Devices may have different memory types for different usage and require
  // the intended usage to be declared upon allocation. It's always best to
  // limit the allowed usage bits to precisely what the actual usage will be to
  // avoid additional copies, synchronization, and expensive emulation.
  //
  // If 0 then the usage will default to all usage modes.
  iree_hal_buffer_usage_t usage;

  // Specifies the access allowed to the memory via the HAL APIs.
  // For example, if the IREE_HAL_MEMORY_ACCESS_WRITE bit is not set then any
  // API call that would write to the memory will fail (such as
  // iree_hal_command_buffer_update_buffer). This does not limit any untrusted
  // dispatch or external use of the buffer and should not be treated as a
  // memory protection mechanism.
  //
  // If 0 then the access will be set as IREE_HAL_MEMORY_ACCESS_ALL.
  iree_hal_memory_access_t access;

  // Specifies the memory type properties used for selecting a memory space.
  // This should often be IREE_HAL_MEMORY_TYPE_OPTIMAL to allow the allocator
  // to place the allocation based on usage bits but can be specified if the
  // exact memory type must be used for compatibility with external code.
  //
  // If 0 then the type will be set as IREE_HAL_MEMORY_TYPE_OPTIMAL.
  iree_hal_memory_type_t type;

  // Queue affinity bitmap indicating which queues may access this buffer.
  // For NUMA devices this can be used to more tightly scope the allocation to
  // particular device memory and provide better pool placement. When a device
  // supports peering or replication the affinity bitmap will be used to choose
  // which subdevices require configuration.
  //
  // If 0 then the buffer will be available on any queue as if
  // IREE_HAL_QUEUE_AFFINITY_ANY was specified.
  iree_hal_queue_affinity_t queue_affinity;

  // Minimum alignment, in bytes, of the resulting allocation.
  // The actual alignment may be any value greater-than-or-equal-to this value.
  //
  // If 0 then the alignment will be decided by the allocator based on optimal
  // device parameters.
  iree_device_size_t min_alignment;
} iree_hal_buffer_params_t;

// Canonicalizes |params| fields when zero initialization is used.
static inline void iree_hal_buffer_params_canonicalize(
    iree_hal_buffer_params_t* params) {
  if (!params->usage) {
    params->usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  }
  if (!params->access) {
    params->access = IREE_HAL_MEMORY_ACCESS_ALL;
  }
  if (!params->type) {
    params->type = IREE_HAL_MEMORY_TYPE_OPTIMAL;
  }
  if (!params->queue_affinity) {
    params->queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  }
}

// Returns |params| with the given |usage| bits OR'ed in.
static inline iree_hal_buffer_params_t iree_hal_buffer_params_with_usage(
    const iree_hal_buffer_params_t params, iree_hal_buffer_usage_t usage) {
  iree_hal_buffer_params_t result = params;
  if (!result.usage) {
    result.usage =
        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  }
  result.usage |= usage;
  return result;
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_mapping_t
//===----------------------------------------------------------------------===//

// Implementation-specific private mapping data.
typedef struct iree_hal_buffer_mapping_impl_t {
  // Byte offset within the buffer where the mapped data begins.
  iree_device_size_t byte_offset;
  // Used for validation only.
  iree_hal_memory_access_t allowed_access;
  // Tracking flags.
  uint32_t is_persistent : 1;
  uint32_t reserved_flags : 31;
  // Backing implementation data.
  // For backends that require additional tracking (shadow data structures/etc)
  // this can be used to store references to them for the duration of the
  // mapping.
  uint64_t reserved[1];
} iree_hal_buffer_mapping_impl_t;

// Reference to a buffer's mapped memory.
typedef struct iree_hal_buffer_mapping_t {
  // Contents of the buffer. Behavior is undefined if an access is performed
  // whose type was not specified during mapping.
  //
  // The bytes available may be greater than what was requested if platform
  // alignment rules require it. Only memory defined by the given span may be
  // accessed.
  iree_byte_span_t contents;

  // Buffer providing the backing storage for the mapping.
  // When mapped with IREE_HAL_MAPPING_MODE_SCOPED the buffer will be retained
  // until it is unmapped. When mapped with IREE_HAL_MAPPING_MODE_PERSISTENT the
  // caller is responsible for retaining the buffer.
  struct iree_hal_buffer_t* buffer;

  // Used internally - do not modify.
  // Implementations are allowed to use the reserved fields for their own
  // storage but should otherwise ignore the remaining parts.
  iree_hal_buffer_mapping_impl_t impl;
} iree_hal_buffer_mapping_t;

//===----------------------------------------------------------------------===//
// iree_hal_buffer_release_callback_t
//===----------------------------------------------------------------------===//

typedef void(IREE_API_PTR* iree_hal_buffer_release_fn_t)(
    void* user_data, struct iree_hal_buffer_t* buffer);

// A callback issued when a buffer is released.
typedef struct {
  // Callback function pointer.
  iree_hal_buffer_release_fn_t fn;
  // User data passed to the callback function. Unowned.
  void* user_data;
} iree_hal_buffer_release_callback_t;

// Returns a no-op buffer release callback that implies that no cleanup is
// required.
static inline iree_hal_buffer_release_callback_t
iree_hal_buffer_release_callback_null(void) {
  iree_hal_buffer_release_callback_t callback = {NULL, NULL};
  return callback;
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

// Allocated memory buffer wrapper type and utilities.
//
// Buffers are the basic unit of memory used by the inference system. They may
// be allocated such that they are accessible from the host (normal C++ code
// running on the main CPU), a particular device (such as an accelerator) or
// family of devices, or from some mix of all of those.
//
// The type of memory a buffer is allocated within has implications on it's
// performance and lifetime. For example if an application attempts to use a
// host-allocated buffer (IREE_HAL_MEMORY_TYPE_HOST_LOCAL) on an accelerator
// with discrete memory the accelerator may either be unable to access the
// memory or take a non-trivial performance hit when attempting to do so
// (involving setting up kernel mappings, doing DMA transfers, etc). Likewise,
// trying to access a device-allocated buffer
// (IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) may incur similar overhead or not be
// possible at all. This may be due to restrictions in the memory visibility,
// address spaces, mixed endianness or pointer widths, and other weirdness.
//
// The memory types (defined by a bitfield of iree_hal_memory_type_t values)
// that a particular context (host or device) may use vary from device to device
// and must be queried by the application when allocating buffers. It's strongly
// recommended that the most specific memory type be set as possible. For
// example allocating a buffer with IREE_HAL_MEMORY_TYPE_HOST_COHERENT even when
// it will never be used in a way that requires coherency may occupy address
// space reservations or memory mapping that would otherwise not be needed.
//
// As buffers may sometimes not be accessible from the host the base buffer type
// does not allow for direct void* access and instead buffers must be either
// manipulated using utility functions (such as ReadData or WriteData) or by
// mapping them into a host-accessible address space via MapMemory. Buffers must
// be unmapped before any command may use them.
//
// Buffers may equate (roughly) 1:1 with an allocation either from the host heap
// or a device. iree_hal_buffer_subspan can be used to reference subspans of
// buffers like std::span - though unlike std::span the returned buffer holds
// a reference to the parent buffer.
typedef struct iree_hal_buffer_t iree_hal_buffer_t;

// Returns success iff the buffer was allocated with the given memory type.
IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_memory_type(
    iree_hal_memory_type_t actual_memory_type,
    iree_hal_memory_type_t expected_memory_type);

// Returns success iff the buffer allows the requested access.
IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_access(
    iree_hal_memory_access_t allowed_memory_access,
    iree_hal_memory_access_t required_memory_access);

// Returns success iff the buffer usage allows the given usage type.
IREE_API_EXPORT iree_status_t
iree_hal_buffer_validate_usage(iree_hal_buffer_usage_t allowed_usage,
                               iree_hal_buffer_usage_t required_usage);

// Returns success iff the given byte range falls within the valid buffer.
IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_range(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length);

// Adjusts the offset and length of a buffer subrange and returns the new
// subrange. Fails if the range is invalid.
IREE_API_EXPORT iree_status_t iree_hal_buffer_calculate_range(
    iree_device_size_t base_offset, iree_device_size_t max_length,
    iree_device_size_t offset, iree_device_size_t length,
    iree_device_size_t* out_adjusted_offset,
    iree_device_size_t* out_adjusted_length);

// Tests whether the given buffers overlap, including support for subspans.
// IREE_HAL_WHOLE_BUFFER may be used for |lhs_length| and/or |rhs_length| to use
// the lengths of those buffers, respectively.
IREE_API_EXPORT iree_hal_buffer_overlap_t iree_hal_buffer_test_overlap(
    iree_hal_buffer_t* lhs_buffer, iree_device_size_t lhs_offset,
    iree_device_size_t lhs_length, iree_hal_buffer_t* rhs_buffer,
    iree_device_size_t rhs_offset, iree_device_size_t rhs_length);

// Returns a reference to a subspan of the |buffer|.
// If |byte_length| is IREE_HAL_WHOLE_BUFFER the remaining bytes in the buffer
// after |byte_offset| (possibly 0) will be selected.
//
// The parent buffer will remain alive for the lifetime of the subspan
// returned. If the subspan is a small portion this may cause additional
// memory to remain allocated longer than required.
//
// Returns the given |buffer| if the requested span covers the entire range.
// |out_buffer| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_hal_buffer_subspan(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer);

// Retains the given |buffer| for the caller.
IREE_API_EXPORT void iree_hal_buffer_retain(iree_hal_buffer_t* buffer);

// Releases the given |buffer| from the caller.
IREE_API_EXPORT void iree_hal_buffer_release(iree_hal_buffer_t* buffer);

// Returns a pointer to the buffer containing the actual allocation.
// The buffer represents a span of the allocated bytes defined by byte_offset
// and byte_length. If the provided buffer *is* the allocated buffer then the
// returned value will be the provided buffer pointer.
IREE_API_EXPORT iree_hal_buffer_t* iree_hal_buffer_allocated_buffer(
    const iree_hal_buffer_t* buffer);

// Returns the size of the resource memory allocation in bytes.
// This may be rounded up from the originally requested size or the ideal
// size for the resource based on device restrictions.
IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_allocation_size(const iree_hal_buffer_t* buffer);

// Returns the original placement of the allocated buffer.
// The placement applies to the entire underlying allocated buffer and not the
// potential subspan of the |buffer| handle. Many buffer handles may be backed
// by the same allocation. It's possible for placements to change over the
// lifetime of a buffer as it is moved across devices but the origin will always
// accept actions on the buffer such as deallocation.
//
// Note that not all buffers have a placement: e.g. host buffers allocated as
// free-floating objects will have no device assigned.
// iree_hal_buffer_placement_is_undefined can be used to check for this case
// explicitly.
IREE_API_EXPORT iree_hal_buffer_placement_t
iree_hal_buffer_allocation_placement(const iree_hal_buffer_t* buffer);

// Preserves the underlying buffer allocation for the caller.
// Preservation is a way to track lifetime of an asynchronously-allocated buffer
// on multiple device timelines. Incrementing the preserve count indicates that
// there is a new co-owner of the buffer lifetime and that owner must make a
// corresponding iree_hal_buffer_allocation_discard call to release their
// ownership and possibly deallocate the buffer.
//
// Though intended for asynchronously-allocated buffers it is fine to preserve
// synchronously-allocated ones. Any code that _may_ receive asynchronously
// allocated buffers must properly balance their preserves and discards. Code
// that will never receive asynchronously allocated buffers - such as those
// using the inline HAL - can ignore tracking.
//
// This preservation roughly translates to retaining logical ownership of the
// allocation and may differ from the buffer object reference count. As an
// example if the Python GC hasn't run there may still be several references to
// the buffer object even after the application has stopped using the buffer.
// Tracking the preserve count independently allows the application to eagerly
// deallocate the buffer without relying on the lifetime of the object to do so.
//
// A preserved buffer will still be deallocated if there are no longer any
// references to the buffer object. Preserving the buffer only prevents any
// other owner from deallocating it while there are references outstanding.
// See iree_hal_buffer_allocation_discard for more information about releasing
// ownership.
IREE_API_EXPORT void iree_hal_buffer_allocation_preserve(
    iree_hal_buffer_t* buffer);

// Discards a previously preserved buffer allocation for the caller.
// Decrementing the preserve count indicates that the owner is releasing its
// ownership. When the last owner discards their ownership it is safe to
// deallocate the buffer allocation even if there are still references remaining
// to the buffer object.
//
// Any code that _may_ receive asynchronously allocated buffers must properly
// balance their preserves and discards. Code that will never receive
// asynchronously allocated buffers - such as those using the inline HAL - can
// ignore tracking as there's no asynchronous deallocation and allocation
// lifetime is tied to buffer object lifetime. Note that unbalanced discards
// will result in either correctness issues (buffer is deallocated too early) or
// extended lifetime (buffer cannot be deallocated until all buffer object
// references have been released).
//
// Returns true if the caller was the last owner of the allocation and it can
// now be deallocated.
//
// Example (note that expensive queue operations are guarded):
//   if (iree_hal_buffer_allocation_discard(buffer)) {
//     placement = iree_hal_buffer_allocation_placement(buffer);
//     if (iree_all_bits_set(placement.flags,
//                           IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS)) {
//       <timeline logic>
//       iree_hal_device_queue_dealloca(
//           placement.device, placement.queue_affinity,
//           wait_semaphore_list, signal_semaphore_list,
//           IREE_HAL_DEALLOCA_FLAG_NONE,
//           iree_hal_buffer_allocated_buffer(buffer));
//     }
//   }
IREE_API_EXPORT IREE_MUST_USE_RESULT bool iree_hal_buffer_allocation_discard(
    iree_hal_buffer_t* buffer);

// Returns true if the caller is the last owner preserving an allocation.
// This can be used to reuse a buffer that has no other owners.
//
// Note that an allocated buffer may have multiple suballocations referencing it
// and this query is only for the entire allocation. When reusing a buffer one
// should ensure the allocation size matches (or is within threshold) so that a
// reuse of 16MB doesn't keep an underlying allocation of 16GB wired.
//
// Since device allocators are expected to reuse memory if in doubt prefer to
// dealloca and alloca. This method should only be used in situations where the
// buffer types are known to the application (such as fixed input and output
// buffers).
//
// Example:
//   if (iree_hal_buffer_allocation_is_terminal(buffer)) {
//     new_buffer = buffer;  // safe to reuse
//   } else {
//     iree_hal_device_queue_alloca(..., &new_buffer);  // need a new buffer
//   }
IREE_API_EXPORT bool iree_hal_buffer_allocation_is_terminal(
    const iree_hal_buffer_t* buffer);

// Returns the offset in bytes of the buffer within its allocated_buffer.
IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_byte_offset(const iree_hal_buffer_t* buffer);

// Returns the size in bytes of the buffer.
IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_byte_length(const iree_hal_buffer_t* buffer);

// Returns the memory type the buffer was allocated with.
IREE_API_EXPORT
iree_hal_memory_type_t iree_hal_buffer_memory_type(
    const iree_hal_buffer_t* buffer);

// Returns the allowed memory access modes.
// These may be more strict than the underlying allocation, for example when the
// buffer is exposing read-only memory that may be in mutable pages.
IREE_API_EXPORT
iree_hal_memory_access_t iree_hal_buffer_allowed_access(
    const iree_hal_buffer_t* buffer);

// Returns the allowed buffer usage modes.
IREE_API_EXPORT
iree_hal_buffer_usage_t iree_hal_buffer_allowed_usage(
    const iree_hal_buffer_t* buffer);

// Sets a range of the buffer to binary zero.
//
// Requires that the buffer has the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
// The byte range in |buffer| will be flushed if needed.
//
// It is strongly recommended that buffer operations are performed on transfer
// queues; using this synchronous function may incur additional cache flushes
// and synchronous blocking behavior and is not supported on all buffer types.
// See iree_hal_command_buffer_fill_buffer.
IREE_API_EXPORT iree_status_t iree_hal_buffer_map_zero(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length);

// Sets a range of the buffer to the given value.
// Only |pattern_length| values with 1, 2, or 4 bytes are supported.
//
// Requires that the buffer has the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
// The byte range in |buffer| will be flushed if needed.
//
// It is strongly recommended that buffer operations are performed on transfer
// queues; using this synchronous function may incur additional cache flushes
// and synchronous blocking behavior and is not supported on all buffer types.
// See iree_hal_command_buffer_fill_buffer.
IREE_API_EXPORT iree_status_t iree_hal_buffer_map_fill(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, const void* pattern,
    iree_host_size_t pattern_length);

// Reads a block of data from the buffer at the given offset.
//
// Requires that the buffer has the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
//
// It is strongly recommended that buffer operations are performed on transfer
// queues; using this synchronous function may incur additional cache flushes
// and synchronous blocking behavior and is not supported on all buffer types.
// See iree_hal_command_buffer_copy_buffer.
IREE_API_EXPORT iree_status_t iree_hal_buffer_map_read(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    void* target_buffer, iree_device_size_t data_length);

// Writes a block of byte data into the buffer at the given offset.
//
// Requires that the buffer has the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
// The byte range in |target_buffer| will be flushed if needed.
//
// It is strongly recommended that buffer operations are performed on transfer
// queues; using this synchronous function may incur additional cache flushes
// and synchronous blocking behavior and is not supported on all buffer types.
// See iree_hal_command_buffer_update_buffer and
// iree_hal_command_buffer_copy_buffer.
IREE_API_EXPORT iree_status_t iree_hal_buffer_map_write(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_device_size_t data_length);

// Copies data from the provided |source_buffer| into the |target_buffer|.
//
// Requires that both buffers have the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
// The byte range in |target_buffer| will be flushed if needed. Both buffers
// need not come from the same device.
//
// It is strongly recommended that buffer operations are performed on transfer
// queues; using this synchronous function may incur additional cache flushes
// and synchronous blocking behavior and is not supported on all buffer types.
// See iree_hal_command_buffer_copy_buffer.
IREE_API_EXPORT iree_status_t iree_hal_buffer_map_copy(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t data_length);

// Maps the buffer to be accessed as a host pointer into |out_buffer_mapping|.
// The byte offset and byte length may be adjusted for device alignment.
// The output data pointer will be properly aligned to the start of the data.
// Fails if the memory could not be mapped (invalid access type, invalid
// range, or unsupported memory type).
//
// Requires that the buffer has the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
// If the buffer is not IREE_HAL_MEMORY_TYPE_HOST_COHERENT then the caller must
// invalidate the byte range they want to access to update the visibility of the
// mapped memory.
IREE_API_EXPORT iree_status_t iree_hal_buffer_map_range(
    iree_hal_buffer_t* buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access, iree_device_size_t byte_offset,
    iree_device_size_t byte_length,
    iree_hal_buffer_mapping_t* out_buffer_mapping);

// Prepares for mapping the buffer to be accessed as a host pointer into
// |out_buffer_mapping|. The byte offset and byte length may be adjusted for
// device alignment. The output data pointer will be properly aligned to the
// start of the data. Fails if the memory could not be mapped (invalid access
// type, invalid range, or unsupported memory type).
//
// Requires that the buffer has the IREE_HAL_BUFFER_USAGE_MAPPING bit set.
// If the buffer is not IREE_HAL_MEMORY_TYPE_HOST_COHERENT then the caller must
// invalidate the byte range they want to access to update the visibility of the
// mapped memory.
//
// This is the first part of a paired operation with
// iree_hal_buffer_commit_map_range. This allows callers to prepare for mapping
// (performing all of the validation) without actually resolving the host
// pointer yet. Once prepared the mapping must be unmapped with
// iree_hal_buffer_unmap_range even if it is never committed.
//
// Callers are allowed to prepare mappings prior to the |buffer| having
// allocated storage. Committing the mapping requires that storage has been
// bound for the duration the mapping will be live.
//
// Example usage:
//  iree_hal_buffer_prepare_map_range(..., &mapping);
//  if (maybe) iree_hal_buffer_commit_map_range(..., &mapping);
//  iree_hal_buffer_unmap_range(&mapping);
IREE_API_EXPORT iree_status_t iree_hal_buffer_prepare_map_range(
    iree_hal_buffer_t* buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access, iree_device_size_t byte_offset,
    iree_device_size_t byte_length,
    iree_hal_buffer_mapping_t* out_buffer_mapping);

// Commits a mapping operation from iree_hal_buffer_prepare_map_range.
// May fail for internal reasons but not any of those previously validated
// during preparation.
IREE_API_EXPORT iree_status_t iree_hal_buffer_commit_map_range(
    iree_hal_buffer_t* buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_hal_buffer_mapping_t* buffer_mapping);

// Unmaps the buffer as was previously mapped to |buffer_mapping|.
//
// If the buffer is not IREE_HAL_MEMORY_TYPE_HOST_COHERENT then the caller must
// flush the byte range they want to make available to other threads/devices.
//
// May fail, though unlikely to do so for read-only mapping and the result can
// be safely ignored using iree_status_ignore. If writing then users must check
// the status to ensure their writes succeeded.
IREE_API_EXPORT iree_status_t
iree_hal_buffer_unmap_range(iree_hal_buffer_mapping_t* buffer_mapping);

// Invalidates ranges of non-coherent memory from the host caches.
// This guarantees that device writes to the memory ranges provided are
// visible on the host. Use before reading from non-coherent memory.
//
// Only required for memory types without IREE_HAL_MEMORY_TYPE_HOST_COHERENT.
IREE_API_EXPORT iree_status_t iree_hal_buffer_mapping_invalidate_range(
    iree_hal_buffer_mapping_t* buffer_mapping, iree_device_size_t byte_offset,
    iree_device_size_t byte_length);

// Flushes ranges of non-coherent memory from the host caches.
// This guarantees that host writes to the memory ranges provided are available
// for device access. Use after writing to non-coherent memory.
//
// Only required for memory types without IREE_HAL_MEMORY_TYPE_HOST_COHERENT.
IREE_API_EXPORT iree_status_t iree_hal_buffer_mapping_flush_range(
    iree_hal_buffer_mapping_t* buffer_mapping, iree_device_size_t byte_offset,
    iree_device_size_t byte_length);

// Calculates and returns a byte subspan range within a buffer mapping.
// The byte range provided is local to the mapping. May return a 0-length span.
// IREE_HAL_WHOLE_BUFFER can be used for |byte_length|.
//
// Note that the access requirements of the mapping still hold: if the memory is
// not host coherent and writeable then the caller must use the
// iree_hal_buffer_mapping_invalidate_range and
// iree_hal_buffer_mapping_flush_range methods to ensure memory is in the
// expected state.
IREE_API_EXPORT iree_status_t iree_hal_buffer_mapping_subspan(
    iree_hal_buffer_mapping_t* buffer_mapping,
    iree_hal_memory_access_t memory_access, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_byte_span_t* out_span);

//===----------------------------------------------------------------------===//
// iree_hal_subspan_buffer_t
//===----------------------------------------------------------------------===//

// Creates a buffer referencing a subspan of some base allocation.
// Optionally |device_allocator| can be provided if this subspan references
// managed buffers that need deallocation callbacks.
IREE_API_EXPORT iree_status_t iree_hal_subspan_buffer_create(
    iree_hal_buffer_t* allocated_buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer);

//===----------------------------------------------------------------------===//
// iree_hal_heap_buffer_t
//===----------------------------------------------------------------------===//

// Wraps an existing host allocation in a buffer.
// When the buffer is destroyed the provided |release_callback| will be called.
//
// The buffer must be aligned to at least IREE_HAL_HEAP_BUFFER_ALIGNMENT and if
// it is not the call will fail with IREE_STATUS_OUT_OF_RANGE.
//
// |out_buffer| must be released by the caller. |data| must be kept live for the
// lifetime of the wrapping buffer.
iree_status_t iree_hal_heap_buffer_wrap(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_byte_span_t data, iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_buffer_vtable_t {
  // Must be iree_hal_buffer_recycle.
  void(IREE_API_PTR* recycle)(iree_hal_buffer_t* buffer);
  void(IREE_API_PTR* destroy)(iree_hal_buffer_t* buffer);

  iree_status_t(IREE_API_PTR* map_range)(iree_hal_buffer_t* buffer,
                                         iree_hal_mapping_mode_t mapping_mode,
                                         iree_hal_memory_access_t memory_access,
                                         iree_device_size_t local_byte_offset,
                                         iree_device_size_t local_byte_length,
                                         iree_hal_buffer_mapping_t* mapping);

  iree_status_t(IREE_API_PTR* unmap_range)(iree_hal_buffer_t* buffer,
                                           iree_device_size_t local_byte_offset,
                                           iree_device_size_t local_byte_length,
                                           iree_hal_buffer_mapping_t* mapping);

  iree_status_t(IREE_API_PTR* invalidate_range)(
      iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
      iree_device_size_t local_byte_length);

  iree_status_t(IREE_API_PTR* flush_range)(
      iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
      iree_device_size_t local_byte_length);
} iree_hal_buffer_vtable_t;
static_assert(offsetof(iree_hal_buffer_vtable_t, recycle) == 0,
              "iree_hal_resource_vtable_t expects destroy at offset 0, we want "
              "to recycle instead");

// NOTE: this shared data structure may be a mistake. If vtables were free we
// would not provide this and rely on each buffer implementation to implement
// all of the accessor methods. Indirection through vtables costs, though, so
// we hoist the common information that every buffer implementation needs here.
// Since this adds a fixed cost to every buffer on every implementation we
// should keep the structure as small as reasonable.
//
// NOTE: the internals of this structure are an implementation detail and may
// change at any time. If there's no API accessor for a field then assume it
// should not be used except by HAL buffer implementations.
struct iree_hal_buffer_t {
  iree_hal_resource_t resource;  // must be at 0
  // Underlying buffer allocation. If this points back at this buffer structure
  // then the buffer is an allocated buffer itself and otherwise the underlying
  // allocation is referenced and retained.
  iree_hal_buffer_t* allocated_buffer;
  // Total size of the buffer allocation in its underlying storage.
  // This is captured on each buffer including non-allocated buffers so that
  // internal pooling/suballocation costs can be represented.
  iree_device_size_t allocation_size;
  // Offset into the underlying allocated buffer this buffer range starts at.
  iree_device_size_t byte_offset;
  // Length of the buffer range in the underlying allocated buffer. This is the
  // logical length exposed to users.
  iree_device_size_t byte_length;

  // Placement of the buffer on a device/queue set. Captured only for allocated
  // buffers.
  iree_hal_buffer_placement_t placement;

  // Hacky back reference to an allocator that should be notified when the
  // buffer is released. This is a hack to support interception of buffers by
  // pooling layers and is slated for removal.
  //
  // TODO(#19159): remove iree_hal_allocator_deallocate_buffer when pooling no
  // longer requires the pooling_allocator on iree_hal_buffer_t.
  iree_hal_allocator_t* pooling_allocator;

  // A counter indicating the number of active preservation requests.
  // This roughly translates to the number of logical "owners" of the allocation
  // and may differ from the buffer object reference count. Note that this
  // should only be used to schedule deallocations: the buffer will remain live
  // and all fields valid until its reference count drops to 0 and its host-side
  // data structures are freed.
  // See iree_hal_buffer_allocation_preserve for more information.
  iree_atomic_uint32_t preserve_count;

  // TODO(benvanik): bit pack these; could be ~4 bytes vs 12.
  iree_hal_memory_type_t memory_type;
  iree_hal_buffer_usage_t allowed_usage;
  iree_hal_memory_access_t allowed_access;

  // Unused padding that more flags or identifiers can be placed in, such as
  // which implementation pool owns the buffer.
  uint16_t reserved;

  // Implementation-defined flags used for additional bookkeeping or routing
  // by the buffer implementation.
  uint32_t flags;
};

IREE_API_EXPORT void iree_hal_buffer_initialize(
    iree_hal_buffer_placement_t placement, iree_hal_buffer_t* allocated_buffer,
    iree_device_size_t allocation_size, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    const iree_hal_buffer_vtable_t* vtable, iree_hal_buffer_t* buffer);

// TODO(#19159): remove iree_hal_allocator_deallocate_buffer when pooling no
// longer requires the pooling_allocator on iree_hal_buffer_t. When buffers can
// use their normal destroy callback to return themselves to pools then we won't
// need this extra recycle thunk.
//
// Recycles |buffer| by releasing it to the origin it is associated with via the
// release callback (or destroying it, if none was specified).
// The |buffer| pointer may remain valid if it is returned to a pool but callers
// must assume its contents are undefined as if it had been freed.
IREE_API_EXPORT void iree_hal_buffer_recycle(iree_hal_buffer_t* buffer);

// Destroys |buffer| and frees its memory.
// Implementations must use iree_hal_buffer_recycle in their vtables for the
// common iree_hal_resource_t destroy callback as this is only to be used by
// release callbacks that want to free the buffer.
IREE_API_EXPORT void iree_hal_buffer_destroy(iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_BUFFER_H_
