// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ALLOCATOR_H_
#define IREE_HAL_ALLOCATOR_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// A bitmap indicating logical device queue affinity.
// Used to direct submissions to specific device queues or locate memory nearby
// where it will be used. The meaning of the bits in the bitmap is
// implementation-specific: a bit may represent a logical queue in an underlying
// API such as a VkQueue or a physical queue such as a discrete virtual device.
//
// Bitwise operations can be performed on affinities; for example AND'ing two
// affinities will produce the intersection and OR'ing will produce the union.
// This enables just-in-time selection as a command buffer could be made
// available to some set of queues when recorded and then AND'ed with an actual
// set of queues to execute on during submission.
typedef uint64_t iree_hal_queue_affinity_t;

// Specifies that any queue may be selected.
#define IREE_HAL_QUEUE_AFFINITY_ANY ((iree_hal_queue_affinity_t)(-1))

// TBD: placeholder for reserving unique pools.
// The intent is that semantically meaningful pools can be defined like
// "transient" "variable" "constant" "external" (matching what we use in the
// compiler) such that allocators don't need to infer based on usage flags.
enum iree_hal_allocator_pool_bits_t {
  IREE_HAL_ALLOCATOR_POOL_DEFAULT = 0u,
};
typedef uint32_t iree_hal_allocator_pool_t;

// Describes a heap of allocatable memory of a specific type.
// Each allocator exposes one or more heaps with differing characteristics. In
// local CPU execution or GPUs with unified memory there may only be one heap
// that covers all memory types and usage while in an out-of-process/sandboxed
// or discrete GPU configuration there may be multiple heaps representing the
// differing memory system properties.
//
// Allocation requests are routed to heaps based on the provided buffer
// properties by matching the first heap in the list that meets the
// requirements. When enumerated the heaps will be in preferred usage order such
// that the matching will always select the most preferred memory heap first
// (intended to be the fastest for a given usage). If no heaps can satisfy a
// given request then allocation will fail.
typedef struct iree_hal_allocator_memory_heap_t {
  // Bits that describe the residency and behavior of the memory type.
  iree_hal_memory_type_t type;

  // Indicates what kind of usage is allowed of buffers allocated from this
  // heap. For example, exclusive device local memory may not allow mapping
  // while cached host local memory may not allow usage in dispatches.
  iree_hal_buffer_usage_t allowed_usage;

  // Maximum size, in bytes, of any individual allocation of this type.
  // Allocations over this size will fail. Note that due to fragmentation it's
  // also possible for allocations under this size to fail.
  iree_device_size_t max_allocation_size;

  // Minimum alignment, in bytes, of allocations of this type.
  // Allocation requests will have their alignment rounded up to at least this.
  iree_device_size_t min_alignment;
} iree_hal_allocator_memory_heap_t;

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

// A bitfield indicating compatible behavior for buffers in an allocator.
enum iree_hal_buffer_compatibility_bits_t {
  // Indicates (in the absence of other bits) the buffer is not compatible with
  // the allocator or device at all. Any attempts to use the buffer for any
  // usage will fail. This will happen if the buffer is device-local to another
  // device without peering and not visible to the host.
  IREE_HAL_BUFFER_COMPATIBILITY_NONE = 0u,

  // Indicates that the allocator could allocate new buffers of this type and
  // usage natively. Allocations with the queried parameters may still fail due
  // to runtime conditions (out of memory, fragmentation, etc) but are otherwise
  // valid.
  IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE = 1u << 0,

  // Indicates that the allocator could import external buffers of this type and
  // usage natively. Imports may fail due to runtime conditions (out of handles,
  // invalid pointer address spaces/page parameters, etc) but are otherwise
  // valid.
  IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE = 1u << 1,

  // Indicates that the allocator could export external buffers of this type and
  // usage natively. Exports may fail due to runtime conditions (out of handles,
  // etc) but are otherwise valid.
  IREE_HAL_BUFFER_COMPATIBILITY_EXPORTABLE = 1u << 2,

  // Indicates that the buffer can be used as a transfer source or target on the
  // a device queue (such as being the source or target of a DMA operation,
  // etc). If not set then the buffer may still be usable for
  // iree_hal_buffer_map_copy but not with queued operations.
  IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER = 1u << 10,

  // Indicates that the buffer can be used as an input/output to a dispatch.
  IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH = 1u << 11,

  // Indicates that buffers allocated this way may have much lower performance
  // than others. An example would be a buffer that must be allocated in host
  // memory and accessed by the device over the PCI bus as compared to device
  // local memory that will be significantly faster.
  IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE = 1u << 20,
};
typedef uint32_t iree_hal_buffer_compatibility_t;

// Parses a buffer compatibility bitfield from a string.
// See iree_bitfield_parse for usage.
IREE_API_EXPORT iree_status_t iree_hal_buffer_compatibility_parse(
    iree_string_view_t value, iree_hal_buffer_compatibility_t* out_value);

// Formats a buffer compatibility bitfield as a string.
// See iree_bitfield_format for usage.
IREE_API_EXPORT iree_string_view_t
iree_hal_buffer_compatibility_format(iree_hal_buffer_compatibility_t value,
                                     iree_bitfield_string_temp_t* out_temp);

// Defines the type of an external buffer handle.
// Each type may only be usable in a subset of implementations and platforms and
// may even vary based on the runtime device properties or buffer instance.
//
// See the notes on each type for requirements; compatibility often requires
// the handle to check and trying to import/export is the most reliable way to
// check for support.
//
// The Vulkan documentation on external memory covers a lot of the design
// decisions made here:
// https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory.html
typedef enum iree_hal_external_buffer_type_e {
  IREE_HAL_EXTERNAL_BUFFER_TYPE_NONE = 0,

  // A host pointer allocated from an external allocator.
  // An imported/exported buffer does not own a reference to the memory and the
  // caller is responsible for ensuring the memory remains live for as long as
  // the iree_hal_buffer_t referencing it.
  //
  // CPU:
  //  When using the default heap allocator this is just a host pointer.
  //
  // CUDA:
  //  Requires device support.
  //  Uses cuMemHostRegister / cuMemHostUnregister.
  //  The memory type specified on import/export determines the required device
  //  capabilities.
  //
  // Vulkan:
  //  Requires VK_EXT_external_memory_host.
  //  Uses VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT.
  IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,

  // A device pointer allocated from an external allocator.
  // An imported/exported buffer does not own a reference to the memory and the
  // caller is responsible for ensuring the memory remains live for as long as
  // the iree_hal_buffer_t referencing it.
  //
  // CPU:
  //  When using the default heap allocator this is just a host pointer.
  //
  // CUDA:
  //  Buffer usage is declared on import.
  //
  // Vulkan:
  //  Requires VK_KHR_buffer_device_address.
  //  Treats the pointer as VkDeviceAddress.
  IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,

  // A driver/device-specific POSIX file descriptor handle.
  // The handle supports dup, dup2, close, and transport using the SCM_RIGHTS
  // control message. All other usage with system APIs is undefined.
  // An imported/exported handle owns a reference to the underlying allocator
  // memory. May only be shared with the same underlying driver and device
  //
  // CUDA:
  //  Requires device support.
  //  Uses CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD.
  //
  // Vulkan:
  //  Requires device support.
  //  Uses VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT.
  IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD,

  // A driver/device-specific Win32 HANDLE.
  // The handle supports DuplicateHandle, CompareObjectHandles, CloseHandle, and
  // Get/SetHandleInformation. All other usage with system APIs is undefined.
  // An imported/exported handle owns a reference to the underlying allocator
  // memory. Must only be shared with the same underlying driver and device.
  //
  // CUDA:
  //  Requires device support.
  //  Uses CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32.
  //
  // Vulkan:
  //  Requires device support.
  //  Uses VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT.
  IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_WIN32,

  // TODO(benvanik): additional memory types:
  //  shared memory fd (shmem)/mapped file
  //  VkBuffer?
  //  VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
  //  VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID
} iree_hal_external_buffer_type_t;

// Flags for controlling iree_hal_external_buffer_t implementation details.
enum iree_hal_external_buffer_flag_bits_t {
  IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE = 0u,
};
typedef uint32_t iree_hal_external_buffer_flags_t;

// Handle to a typed external buffer.
// This is a non-owning reference and the underlying allocation must remain
// valid for as long as the handle is in use. Some buffer types support internal
// referencing counting but in general ownership remains with the caller.
// See the type enum for more information.
typedef struct iree_hal_external_buffer_t {
  // Type of the resource used to interpret the handle.
  iree_hal_external_buffer_type_t type;
  // Flags indicating buffer compatibility.
  iree_hal_external_buffer_flags_t flags;
  // Total size of the external resource in bytes.
  iree_device_size_t size;
  union {
    // IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION
    struct {
      // Host memory pointer.
      void* ptr;
    } host_allocation;
    // IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION
    struct {
      // Device memory pointer. Pointer width may vary across devices so it is
      // always treated as a 64-bit integer here.
      uint64_t ptr;
    } device_allocation;
    // IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD
    struct {
      int fd;
    } opaque_fd;
    // IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_WIN32
    struct {
      void* handle;
    } opaque_win32;
  } handle;
} iree_hal_external_buffer_t;

//===----------------------------------------------------------------------===//
// Statistics/reporting
//===----------------------------------------------------------------------===//

// Aggregate allocation statistics.
typedef struct iree_hal_allocator_statistics_t {
#if IREE_STATISTICS_ENABLE
  iree_device_size_t host_bytes_peak;
  iree_device_size_t host_bytes_allocated;
  iree_device_size_t host_bytes_freed;
  iree_device_size_t device_bytes_peak;
  iree_device_size_t device_bytes_allocated;
  iree_device_size_t device_bytes_freed;
  // TODO(benvanik): mapping information (discarded, mapping ranges,
  //                 flushed/invalidated, etc).
#else
  int reserved;
#endif  // IREE_STATISTICS_ENABLE
} iree_hal_allocator_statistics_t;

// Formats allocator statistics as a pretty-printed multi-line string.
IREE_API_EXPORT iree_status_t iree_hal_allocator_statistics_format(
    const iree_hal_allocator_statistics_t* statistics,
    iree_string_builder_t* builder);

//===----------------------------------------------------------------------===//
// iree_hal_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_allocator_t iree_hal_allocator_t;

// Retains the given |allocator| for the caller.
IREE_API_EXPORT void iree_hal_allocator_retain(iree_hal_allocator_t* allocator);

// Releases the given |allocator| from the caller.
IREE_API_EXPORT void iree_hal_allocator_release(
    iree_hal_allocator_t* allocator);

// Returns the host allocator used for allocating host objects.
IREE_API_EXPORT iree_allocator_t iree_hal_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT allocator);

// Trims cached/unused pooled buffers, if any.
IREE_API_EXPORT
iree_status_t iree_hal_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT allocator);

// Queries the aggregate statistics from the allocator since creation.
// Thread-safe; statistics are captured at the time the call is made.
//
// NOTE: statistics may be compiled out in some configurations and this call
// will become a memset(0).
IREE_API_EXPORT void iree_hal_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics);

// Prints the current allocation statistics of |allocator| to |file|.
// No-op if statistics are not enabled (IREE_STATISTICS_ENABLE).
IREE_API_EXPORT iree_status_t iree_hal_allocator_statistics_fprint(
    FILE* file, iree_hal_allocator_t* IREE_RESTRICT allocator);

// Queries the available memory heaps used for servicing allocation requests.
// The resulting heaps are sorted in preferred performance order for common
// execution with the most preferred first.
//
// If |heaps| is NULL then the call will query the heap count. This allows for
// preallocation of storage:
//   iree_host_size_t count = 0;
//   iree_hal_allocator_query_memory_heaps(allocator, 0, NULL, &count);
//   ... heaps = iree_alloca(sizeof(heap) * count);
//   iree_hal_allocator_query_memory_heaps(allocator, count, heaps, &count);
//
// Returns the total count in |out_count| and if |capacity| is large enough
// will fill |heaps| with the heap information.
// Returns IREE_STATUS_OUT_OF_RANGE if |capacity| is too small.
IREE_API_EXPORT iree_status_t iree_hal_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT allocator, iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count);

// Returns a bitmask indicating what operations with buffers of the given type
// are available on the allocator. If any parameters or the allocation size must
// be changed by the allocator to match device requirements they will be
// returned in the optional |out_params| and |out_allocation_size| arguments.
//
// For buffers allocated from the given allocator it's expected that the result
// will always be non-NONE. For buffers that originate from another allocator
// there may be limited support for cross-device usage.
//
// Returning IREE_HAL_BUFFER_COMPATIBILITY_NONE indicates that the buffer must
// be transferred externally into a buffer compatible with the device the
// allocator services.
IREE_API_EXPORT iree_hal_buffer_compatibility_t
iree_hal_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t* out_params,
    iree_device_size_t* out_allocation_size);

// Allocates a buffer from the allocator.
// If |initial_data| is provided then the bytes will be copied into the device
// buffer. To avoid the copy when device-accessible constant data is used prefer
// iree_hal_allocator_import_buffer when available.
//
// The memory type of the buffer returned may differ from the requested value
// if the device can provide more functionality; for example, if requesting
// IREE_HAL_MEMORY_TYPE_HOST_VISIBLE but the memory is really host cached you
// may get a buffer back with IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
// IREE_HAL_MEMORY_TYPE_HOST_CACHED. The only requirement is that the buffer
// satisfy the required bits.
//
// |out_buffer| must be released by the caller.
// Fails if the memory type requested for the given usage cannot be serviced.
// Callers can use iree_hal_allocator_query_buffer_compatibility to decide their
// memory use strategy.
IREE_API_EXPORT iree_status_t iree_hal_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_const_byte_span_t initial_data, iree_hal_buffer_t** out_buffer);

// TODO(benvanik): iree_hal_allocator_query_external_buffer_compatibility to
// check for support without needing an external buffer already. There's a few
// usage modes and it'd be nice to have a single function for it to keep the
// interface slimmer.

// Imports an externally-owned |external_buffer| to a buffer handle.
// See notes on iree_hal_external_buffer_type_t for ownership information;
// depending on the type the caller may be responsible for ensuring the external
// buffer remains valid for the duration it is in use by the returned
// iree_hal_buffer_t. The returned external buffer may only be usable with the
// same driver/device.
//
// iree_hal_allocator_query_buffer_compatibility can be used to query whether a
// buffer can be imported when using the given memory type and usage. A
// compatibility result containing IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE
// means the import _may_ succeed however if the pointer/page range is not in a
// supported mode (no read access, etc) this call will fail with
// IREE_STATUS_OUT_OF_RANGE.
//
// An optional |release_callback| can be provided to allow the caller to listen
// for when the underlying resource is no longer in use by the HAL. This can
// be used to perform lifetime management or flushing.
//
// |out_buffer| must be released by the caller.
// Fails with IREE_STATUS_UNAVAILABLE if the allocator cannot import the buffer
// into the given memory type. This may be due to unavailable device/platform
// capabilities or the memory type the external buffer was allocated with.
IREE_API_EXPORT iree_status_t iree_hal_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_params_t params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer);

// Exports an allocator-owned |buffer| to an external buffer handle.
// See the notes on iree_hal_external_buffer_type_t for ownership information.
// Upon successful return the caller is responsible for any required lifetime
// management on the external buffer which may include ensuring that the
// provided source |buffer| is kept live. The returned external buffer may only
// be usable with the same driver/device.
//
// Fails with IREE_STATUS_UNAVAILABLE if the allocator cannot export the buffer
// into the external type. This may be due to unavailable device/platform
// capabilities or the memory type the buffer was allocated with.
IREE_API_EXPORT iree_status_t iree_hal_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer);

//===----------------------------------------------------------------------===//
// iree_hal_heap_allocator_t
//===----------------------------------------------------------------------===//

// Creates a host-local heap allocator that can be used when buffers are
// required that will not interact with a real hardware device (such as those
// used in file IO or tests). Buffers allocated with this will not be compatible
// with real device allocators and will likely incur a copy (or failure) if
// used.
//
// The buffers created from the allocator will use |host_allocator| for their
// metadata and |data_allocator| for their device storage allocations. If the
// two are the same the buffers will be allocated in a single flat slab.
IREE_API_EXPORT iree_status_t iree_hal_allocator_create_heap(
    iree_string_view_t identifier, iree_allocator_t data_allocator,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

//===----------------------------------------------------------------------===//
// iree_hal_allocator_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_allocator_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_allocator_t* IREE_RESTRICT allocator);

  iree_allocator_t(IREE_API_PTR* host_allocator)(
      const iree_hal_allocator_t* IREE_RESTRICT allocator);

  iree_status_t(IREE_API_PTR* trim)(
      iree_hal_allocator_t* IREE_RESTRICT allocator);

  void(IREE_API_PTR* query_statistics)(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics);

  iree_status_t(IREE_API_PTR* query_memory_heaps)(
      iree_hal_allocator_t* IREE_RESTRICT allocator, iree_host_size_t capacity,
      iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
      iree_host_size_t* IREE_RESTRICT out_count);

  iree_hal_buffer_compatibility_t(IREE_API_PTR* query_buffer_compatibility)(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_buffer_params_t* IREE_RESTRICT params,
      iree_device_size_t* IREE_RESTRICT allocation_size);

  iree_status_t(IREE_API_PTR* allocate_buffer)(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      const iree_hal_buffer_params_t* IREE_RESTRICT params,
      iree_device_size_t allocation_size, iree_const_byte_span_t initial_data,
      iree_hal_buffer_t** IREE_RESTRICT out_buffer);

  void(IREE_API_PTR* deallocate_buffer)(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_buffer_t* IREE_RESTRICT buffer);

  iree_status_t(IREE_API_PTR* import_buffer)(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      const iree_hal_buffer_params_t* IREE_RESTRICT params,
      iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
      iree_hal_buffer_release_callback_t release_callback,
      iree_hal_buffer_t** IREE_RESTRICT out_buffer);

  iree_status_t(IREE_API_PTR* export_buffer)(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_buffer_t* IREE_RESTRICT buffer,
      iree_hal_external_buffer_type_t requested_type,
      iree_hal_external_buffer_flags_t requested_flags,
      iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer);
} iree_hal_allocator_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_allocator_vtable_t);

IREE_API_EXPORT void iree_hal_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT allocator);

IREE_API_EXPORT void iree_hal_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer);

#if IREE_STATISTICS_ENABLE

// Records a buffer allocation to |statistics|.
static inline void iree_hal_allocator_statistics_record_alloc(
    iree_hal_allocator_statistics_t* statistics,
    iree_hal_memory_type_t memory_type, iree_device_size_t allocation_size) {
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    statistics->host_bytes_allocated += allocation_size;
    statistics->host_bytes_peak =
        iree_max(statistics->host_bytes_peak, statistics->host_bytes_allocated -
                                                  statistics->host_bytes_freed);
  } else {
    statistics->device_bytes_allocated += allocation_size;
    statistics->device_bytes_peak = iree_max(
        statistics->device_bytes_peak,
        statistics->device_bytes_allocated - statistics->device_bytes_freed);
  }
}

// Records a buffer deallocation to |statistics|.
static inline void iree_hal_allocator_statistics_record_free(
    iree_hal_allocator_statistics_t* statistics,
    iree_hal_memory_type_t memory_type, iree_device_size_t allocation_size) {
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    statistics->host_bytes_freed += allocation_size;
  } else {
    statistics->device_bytes_freed += allocation_size;
  }
}

#else
#define iree_hal_allocator_statistics_record_alloc(...)
#define iree_hal_allocator_statistics_record_free(...)
#endif  // IREE_STATISTICS_ENABLE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ALLOCATOR_H_
