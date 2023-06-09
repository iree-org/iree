// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/buffer_heap_impl.h"
#include "iree/hal/resource.h"

typedef struct iree_hal_heap_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_allocator_t data_allocator;
  iree_string_view_t identifier;
  IREE_STATISTICS(iree_hal_heap_allocator_statistics_t statistics;)
} iree_hal_heap_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_heap_allocator_vtable;

iree_hal_heap_allocator_t* iree_hal_heap_allocator_cast(
    iree_hal_allocator_t* IREE_RESTRICT base_value) {
  return (iree_hal_heap_allocator_t*)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_create_heap(
    iree_string_view_t identifier, iree_allocator_t data_allocator,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_heap_allocator_t* allocator = NULL;
  iree_host_size_t total_size =
      iree_sizeof_struct(*allocator) + identifier.size;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_heap_allocator_vtable,
                                 &allocator->resource);
    allocator->host_allocator = host_allocator;
    allocator->data_allocator = data_allocator;
    iree_string_view_append_to_buffer(
        identifier, &allocator->identifier,
        (char*)allocator + iree_sizeof_struct(*allocator));

    IREE_STATISTICS({
      // All start initialized to zero.
      iree_slim_mutex_initialize(&allocator->statistics.mutex);
    });

    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_heap_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_heap_allocator_t* allocator =
      iree_hal_heap_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_STATISTICS(iree_slim_mutex_deinitialize(&allocator->statistics.mutex));

  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_heap_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_heap_allocator_t* allocator =
      (iree_hal_heap_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_heap_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_heap_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_heap_allocator_t* allocator =
        iree_hal_heap_allocator_cast(base_allocator);
    iree_slim_mutex_lock(&allocator->statistics.mutex);
    memcpy(out_statistics, &allocator->statistics.base,
           sizeof(*out_statistics));
    iree_slim_mutex_unlock(&allocator->statistics.mutex);
  });
}

static iree_status_t iree_hal_heap_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  const iree_host_size_t count = 1;
  if (out_count) *out_count = count;
  if (capacity < count) {
    // NOTE: lightweight as this is hit in normal pre-sizing usage.
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  heaps[0] = (iree_hal_allocator_memory_heap_t){
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
              IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      .allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                       IREE_HAL_BUFFER_USAGE_DISPATCH |
                       IREE_HAL_BUFFER_USAGE_SHARING_EXPORT |
                       IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE |
                       IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT |
                       IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE |
                       IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                       IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
                       IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
                       IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
                       IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE,
      .max_allocation_size = ~(iree_device_size_t)0,
      .min_alignment = IREE_HAL_HEAP_BUFFER_ALIGNMENT,
  };
  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_heap_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  // All buffers can be allocated on the heap and all heap-accessible buffers
  // can be imported/exported.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
      IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE |
      IREE_HAL_BUFFER_COMPATIBILITY_EXPORTABLE;

  // Buffers can only be used on the queue if they are device visible.
  // This is not a strict requirement of heap buffers but matches devices that
  // have discrete memory spaces (remoting/sandboxed, GPUs, etc) and makes it
  // much easier to find issues of buffer definition with local devices that
  // will cause issues when used with real devices.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
    }
    if (iree_any_bit_set(params->usage,
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  // Always ensure we are host-visible.
  params->type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;

  // We are now optimal.
  params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;

  // Host currently uses mapping to copy buffers, which is done a lot.
  // We could probably remove this mutation by preventing copies in those cases.
  // TODO(benvanik): check if transfer is still required for DMA copy source.
  params->usage |=
      IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_TRANSFER;

  return compatibility;
}

static iree_status_t iree_hal_heap_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size, iree_const_byte_span_t initial_data,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_heap_allocator_t* allocator =
      iree_hal_heap_allocator_cast(base_allocator);

  // Coerce options into those required for use by heap-based devices.
  iree_hal_buffer_params_t compat_params = *params;
  if (!iree_all_bits_set(iree_hal_heap_allocator_query_buffer_compatibility(
                             base_allocator, &compat_params, &allocation_size),
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
  }

  // Allocate the buffer (both the wrapper and the contents).
  iree_hal_heap_allocator_statistics_t* statistics = NULL;
  IREE_STATISTICS(statistics = &allocator->statistics);
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_heap_buffer_create(
      base_allocator, statistics, &compat_params, allocation_size, initial_data,
      allocator->data_allocator, allocator->host_allocator, &buffer));

  *out_buffer = buffer;
  return iree_ok_status();
}

static void iree_hal_heap_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  // We don't do any pooling yet.
  // TODO(benvanik): move stats tracking here.
  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_heap_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // Coerce options into those required for use by heap-based devices.
  iree_hal_buffer_params_t compat_params = *params;
  iree_device_size_t allocation_size = external_buffer->size;
  if (!iree_all_bits_set(iree_hal_heap_allocator_query_buffer_compatibility(
                             base_allocator, &compat_params, &allocation_size),
                         IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot import a buffer with the given parameters");
  }

  void* ptr = NULL;
  switch (external_buffer->type) {
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION:
      ptr = external_buffer->handle.host_allocation.ptr;
      break;
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION:
      ptr = (void*)external_buffer->handle.device_allocation.ptr;
      break;
    default:
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "external buffer type not supported");
  }

  return iree_hal_heap_buffer_wrap(
      base_allocator, compat_params.type, compat_params.access,
      compat_params.usage, external_buffer->size,
      iree_make_byte_span(ptr, external_buffer->size), release_callback,
      out_buffer);
}

static iree_status_t iree_hal_heap_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  if (requested_type != IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "external buffer type not supported");
  }

  // Map the entire buffer persistently, if possible.
  iree_hal_buffer_mapping_t mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
      iree_hal_buffer_allowed_access(buffer), 0, IREE_WHOLE_BUFFER, &mapping));

  // Note that the returned pointer is unowned.
  out_external_buffer->type = requested_type;
  out_external_buffer->flags = requested_flags;
  out_external_buffer->size = mapping.contents.data_length;
  out_external_buffer->handle.host_allocation.ptr = mapping.contents.data;
  return iree_ok_status();
}

static const iree_hal_allocator_vtable_t iree_hal_heap_allocator_vtable = {
    .destroy = iree_hal_heap_allocator_destroy,
    .host_allocator = iree_hal_heap_allocator_host_allocator,
    .trim = iree_hal_heap_allocator_trim,
    .query_statistics = iree_hal_heap_allocator_query_statistics,
    .query_memory_heaps = iree_hal_heap_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_heap_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_heap_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_heap_allocator_deallocate_buffer,
    .import_buffer = iree_hal_heap_allocator_import_buffer,
    .export_buffer = iree_hal_heap_allocator_export_buffer,
};
