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
  // iree_hal_buffer_copy_data but not with queued operations.
  IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER = 1u << 10,

  // Indicates that the buffer can be used as an input/output to a dispatch.
  IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH = 1u << 11,
};
typedef uint32_t iree_hal_buffer_compatibility_t;

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
  // CUDA:
  //  Requires device support.
  //  Uses cuMemHostRegister / cuMemHostUnregister.
  //  The memory type specified on import/export determines the required device
  //  capabilities.
  //
  // Vulkan:
  //  Requires VK_EXT_external_memory_host.
  //  Requires device support.
  //  Uses VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT.
  IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION = 1,

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
  IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD = 2,

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
  IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_WIN32 = 3,

  // TODO(benvanik): additional memory types:
  //  shared memory fd (shmem)/mapped file
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
IREE_API_EXPORT iree_allocator_t
iree_hal_allocator_host_allocator(const iree_hal_allocator_t* allocator);

// Trims cached/unused pooled buffers, if any.
IREE_API_EXPORT
iree_status_t iree_hal_allocator_trim(iree_hal_allocator_t* allocator);

// Queries the aggregate statistics from the allocator since creation.
// Thread-safe; statistics are captured at the time the call is made.
//
// NOTE: statistics may be compiled out in some configurations and this call
// will become a memset(0).
IREE_API_EXPORT void iree_hal_allocator_query_statistics(
    iree_hal_allocator_t* allocator,
    iree_hal_allocator_statistics_t* out_statistics);

// Prints the current allocation statistics of |allocator| to |file|.
// No-op if statistics are not enabled (IREE_STATISTICS_ENABLE).
IREE_API_EXPORT iree_status_t iree_hal_allocator_statistics_fprint(
    FILE* file, iree_hal_allocator_t* allocator);

// Returns a bitmask indicating what operations with buffers of the given type
// are available on the allocator.
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
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_usage_t intended_usage, iree_device_size_t allocation_size);

// Allocates a buffer from the allocator.
// If |initial_data| is provided then the bytes will be copied into the device
// buffer. To avoid the copy when constant data is used prefer
// iree_hal_allocator_wrap_buffer when available.
// Fails if the memory type requested for the given usage cannot be serviced.
// Callers can use iree_hal_allocator_can_allocate to decide their memory use
// strategy.
//
// The memory type of the buffer returned may differ from the requested value
// if the device can provide more functionality; for example, if requesting
// IREE_HAL_MEMORY_TYPE_HOST_VISIBLE but the memory is really host cached you
// may get a buffer back with IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
// IREE_HAL_MEMORY_TYPE_HOST_CACHED. The only requirement is that the buffer
// satisfy the required bits.
//
// Fails if it is not possible to allocate and satisfy all placements for the
// requested |allowed_usage|.
// |out_buffer| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_hal_allocator_allocate_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_const_byte_span_t initial_data, iree_hal_buffer_t** out_buffer);

// Wraps an existing host allocation in a buffer.
//
// iree_hal_allocator_query_buffer_compatibility can be used to query whether a
// buffer can be wrapped when using the given memory type and usage. A
// compatibility result containing IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE
// means the wrap may succeed however if the pointer/page range is not in a
// supported mode (no read access, etc) this call may still fail.
//
// |data_allocator| will be used to free the memory when the buffer is
// destroyed. iree_allocator_null() can be passed to indicate the buffer does
// not own the data.
//
// Fails if the allocator cannot access host memory in this way.
// |out_buffer| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_hal_allocator_wrap_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer);

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
// Fails with IREE_STATUS_UNAVAILABLE if the allocator cannot import the buffer
// into the given memory type. This may be due to unavailable device/platform
// capabilities or the memory type the external buffer was allocated with.
IREE_API_EXPORT iree_status_t iree_hal_allocator_import_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_external_buffer_t* external_buffer,
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
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* out_external_buffer);

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
  void(IREE_API_PTR* destroy)(iree_hal_allocator_t* allocator);

  iree_allocator_t(IREE_API_PTR* host_allocator)(
      const iree_hal_allocator_t* allocator);

  iree_status_t(IREE_API_PTR* trim)(iree_hal_allocator_t* allocator);

  void(IREE_API_PTR* query_statistics)(
      iree_hal_allocator_t* allocator,
      iree_hal_allocator_statistics_t* out_statistics);

  iree_hal_buffer_compatibility_t(IREE_API_PTR* query_buffer_compatibility)(
      iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
      iree_hal_buffer_usage_t allowed_usage,
      iree_hal_buffer_usage_t intended_usage,
      iree_device_size_t allocation_size);

  iree_status_t(IREE_API_PTR* allocate_buffer)(
      iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
      iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
      iree_const_byte_span_t initial_data, iree_hal_buffer_t** out_buffer);

  void(IREE_API_PTR* deallocate_buffer)(iree_hal_allocator_t* allocator,
                                        iree_hal_buffer_t* buffer);

  iree_status_t(IREE_API_PTR* wrap_buffer)(
      iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
      iree_hal_memory_access_t allowed_access,
      iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
      iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer);

  iree_status_t(IREE_API_PTR* import_buffer)(
      iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
      iree_hal_memory_access_t allowed_access,
      iree_hal_buffer_usage_t allowed_usage,
      iree_hal_external_buffer_t* external_buffer,
      iree_hal_buffer_t** out_buffer);

  iree_status_t(IREE_API_PTR* export_buffer)(
      iree_hal_allocator_t* allocator, iree_hal_buffer_t* buffer,
      iree_hal_external_buffer_type_t requested_type,
      iree_hal_external_buffer_flags_t requested_flags,
      iree_hal_external_buffer_t* out_external_buffer);
} iree_hal_allocator_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_allocator_vtable_t);

IREE_API_EXPORT void iree_hal_allocator_destroy(
    iree_hal_allocator_t* allocator);

IREE_API_EXPORT void iree_hal_allocator_deallocate_buffer(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* buffer);

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
