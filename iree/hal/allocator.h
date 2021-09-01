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

  // Indicates that the buffer can be used as a transfer source or target on the
  // a device queue (such as being the source or target of a DMA operation,
  // etc). If not set then the buffer may still be usable for
  // iree_hal_buffer_copy_data but not with queued operations.
  IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER = 1u << 10,

  // Indicates that the buffer can be used as an input/output to a dispatch.
  IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH = 1u << 11,
};
typedef uint32_t iree_hal_buffer_compatibility_t;

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

// Queries the aggregate statistics from the allocator since creation.
// Thread-safe; statistics are captured at the time the call is made.
//
// NOTE: statistics may be compiled out in some configurations and this call
// will become a memset(0).
IREE_API_EXPORT void iree_hal_allocator_query_statistics(
    iree_hal_allocator_t* allocator,
    iree_hal_allocator_statistics_t* out_statistics);

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
    iree_hal_buffer_t** out_buffer);

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

// Prints the current allocation statistics of |allocator| to |file|.
// No-op if statistics are not enabled (IREE_STATISTICS_ENABLE).
IREE_API_EXPORT iree_status_t iree_hal_allocator_statistics_fprint(
    FILE* file, iree_hal_allocator_t* allocator);

//===----------------------------------------------------------------------===//
// iree_hal_heap_allocator_t
//===----------------------------------------------------------------------===//

// Creates a host-local heap allocator that can be used when buffers are
// required that will not interact with a real hardware device (such as those
// used in file IO or tests). Buffers allocated with this will not be compatible
// with real device allocators and will likely incur a copy (or failure) if
// used.
IREE_API_EXPORT iree_status_t iree_hal_allocator_create_heap(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

//===----------------------------------------------------------------------===//
// iree_hal_allocator_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_allocator_vtable_t {
  // << HAL C porting in progress >>
  IREE_API_UNSTABLE

  void(IREE_API_PTR* destroy)(iree_hal_allocator_t* allocator);

  iree_allocator_t(IREE_API_PTR* host_allocator)(
      const iree_hal_allocator_t* allocator);

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
      iree_hal_buffer_t** out_buffer);

  iree_status_t(IREE_API_PTR* wrap_buffer)(
      iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
      iree_hal_memory_access_t allowed_access,
      iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
      iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer);
} iree_hal_allocator_vtable_t;

IREE_API_EXPORT void iree_hal_allocator_destroy(
    iree_hal_allocator_t* allocator);

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
