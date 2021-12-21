// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/buffer_heap_impl.h"
#include "iree/hal/resource.h"

typedef struct iree_hal_heap_buffer_t {
  iree_hal_buffer_t base;

  iree_byte_span_t data;
  iree_allocator_t data_allocator;

  // Optional statistics shared with the allocator.
  IREE_STATISTICS(iree_hal_heap_allocator_statistics_t* statistics;)
} iree_hal_heap_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_heap_buffer_vtable;

// Allocates a buffer with the metadata and storage split.
// This results in an additional host allocation but allows for user-overridden
// data storage allocations.
static iree_status_t iree_hal_heap_buffer_allocate_split(
    iree_device_size_t allocation_size, iree_allocator_t data_allocator,
    iree_allocator_t host_allocator, iree_hal_heap_buffer_t** out_buffer,
    iree_byte_span_t* out_data) {
  // Try allocating the storage first as it's the most likely to fail if OOM.
  out_data->data_length = allocation_size;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(data_allocator, allocation_size,
                                             (void**)&out_data->data));

  // Allocate the host metadata wrapper.
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(**out_buffer), (void**)out_buffer);
  if (!iree_status_is_ok(status)) {
    // Need to free the storage we just allocated.
    iree_allocator_free(data_allocator, out_data->data);
  }
  return status;
}

// Allocates a buffer with the metadata as a prefix to the storage.
// This results in a single allocation per buffer but requires that both the
// metadata and storage live together.
static iree_status_t iree_hal_heap_buffer_allocate_slab(
    iree_device_size_t allocation_size, iree_allocator_t host_allocator,
    iree_hal_heap_buffer_t** out_buffer, iree_byte_span_t* out_data) {
  // NOTE: we want the buffer data to always be 16-byte aligned.
  iree_hal_heap_buffer_t* buffer = NULL;
  iree_host_size_t header_size =
      iree_host_align(iree_sizeof_struct(*buffer), 16);
  iree_host_size_t total_size = header_size + allocation_size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&buffer));
  *out_buffer = buffer;
  *out_data =
      iree_make_byte_span((uint8_t*)buffer + header_size, allocation_size);
  return iree_ok_status();
}

iree_status_t iree_hal_heap_buffer_create(
    iree_hal_allocator_t* allocator,
    iree_hal_heap_allocator_statistics_t* statistics,
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_allocator_t data_allocator, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // If the data and host allocators are the same we can allocate more
  // efficiently as a large slab. Otherwise we need to allocate both the
  // metadata and the storage independently.
  bool same_allocator =
      memcmp(&data_allocator, &host_allocator, sizeof(data_allocator)) == 0;

  iree_hal_heap_buffer_t* buffer = NULL;
  iree_byte_span_t data = iree_make_byte_span(NULL, 0);
  iree_status_t status =
      same_allocator
          ? iree_hal_heap_buffer_allocate_slab(allocation_size, host_allocator,
                                               &buffer, &data)
          : iree_hal_heap_buffer_allocate_split(allocation_size, data_allocator,
                                                host_allocator, &buffer, &data);

  if (iree_status_is_ok(status)) {
    iree_hal_buffer_initialize(host_allocator, allocator, &buffer->base,
                               allocation_size, 0, allocation_size, memory_type,
                               allowed_access, allowed_usage,
                               &iree_hal_heap_buffer_vtable, &buffer->base);
    buffer->data = data;
    buffer->data_allocator =
        same_allocator ? iree_allocator_null() : data_allocator;

    IREE_STATISTICS({
      if (statistics != NULL) {
        buffer->statistics = statistics;
        iree_slim_mutex_lock(&statistics->mutex);
        iree_hal_allocator_statistics_record_alloc(
            &statistics->base, memory_type, allocation_size);
        iree_slim_mutex_unlock(&statistics->mutex);
      }
    });

    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_heap_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_byte_span_t data, iree_allocator_t data_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(allocator);
  iree_hal_heap_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_initialize(host_allocator, allocator, &buffer->base,
                               allocation_size, 0, data.data_length,
                               memory_type, allowed_access, allowed_usage,
                               &iree_hal_heap_buffer_vtable, &buffer->base);
    buffer->data = data;
    buffer->data_allocator = data_allocator;
    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_heap_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_heap_buffer_t* buffer = (iree_hal_heap_buffer_t*)base_buffer;
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_STATISTICS({
    if (buffer->statistics != NULL) {
      iree_slim_mutex_lock(&buffer->statistics->mutex);
      iree_hal_allocator_statistics_record_free(&buffer->statistics->base,
                                                base_buffer->memory_type,
                                                base_buffer->allocation_size);
      iree_slim_mutex_unlock(&buffer->statistics->mutex);
    }
  });

  iree_allocator_free(buffer->data_allocator, buffer->data.data);
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_heap_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    void** out_data_ptr) {
  iree_hal_heap_buffer_t* buffer = (iree_hal_heap_buffer_t*)base_buffer;
  *out_data_ptr = buffer->data.data + local_byte_offset;

  // If we mapped for discard scribble over the bytes. This is not a mandated
  // behavior but it will make debugging issues easier. Alternatively for
  // heap buffers we could reallocate them such that ASAN yells, but that
  // would only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(*out_data_ptr, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG

  return iree_ok_status();
}

static void iree_hal_heap_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, void* data_ptr) {
  // No-op here as we always have the pointer.
}

static iree_status_t iree_hal_heap_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_atomic_thread_fence(iree_memory_order_acquire);
  return iree_ok_status();
}

static iree_status_t iree_hal_heap_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_atomic_thread_fence(iree_memory_order_release);
  return iree_ok_status();
}

static const iree_hal_buffer_vtable_t iree_hal_heap_buffer_vtable = {
    .destroy = iree_hal_heap_buffer_destroy,
    .map_range = iree_hal_heap_buffer_map_range,
    .unmap_range = iree_hal_heap_buffer_unmap_range,
    .invalidate_range = iree_hal_heap_buffer_invalidate_range,
    .flush_range = iree_hal_heap_buffer_flush_range,
};
