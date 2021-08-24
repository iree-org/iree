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
#include "iree/hal/resource.h"

typedef struct iree_hal_heap_buffer_t {
  iree_hal_buffer_t base;

  iree_byte_span_t data;
  iree_allocator_t data_allocator;
} iree_hal_heap_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_heap_buffer_vtable;

iree_status_t iree_hal_heap_buffer_create(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: we want the buffer data to always be 16-byte aligned.
  iree_hal_heap_buffer_t* buffer = NULL;
  iree_host_size_t header_size =
      iree_host_align(iree_sizeof_struct(*buffer), 16);
  iree_host_size_t total_size = header_size + allocation_size;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_heap_buffer_vtable,
                                 &buffer->base.resource);
    buffer->base.allocator = allocator;
    buffer->base.allocated_buffer = &buffer->base;
    buffer->base.allocation_size = allocation_size;
    buffer->base.byte_offset = 0;
    buffer->base.byte_length = allocation_size;
    buffer->base.memory_type = memory_type;
    buffer->base.allowed_access = allowed_access;
    buffer->base.allowed_usage = allowed_usage;
    buffer->data =
        iree_make_byte_span((uint8_t*)buffer + header_size, allocation_size);
    buffer->data_allocator = iree_allocator_null();  // freed with the buffer
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

  iree_hal_heap_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(iree_hal_allocator_host_allocator(allocator),
                            sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_heap_buffer_vtable,
                                 &buffer->base.resource);
    buffer->base.allocator = allocator;
    buffer->base.allocated_buffer = &buffer->base;
    buffer->base.allocation_size = allocation_size;
    buffer->base.byte_offset = 0;
    buffer->base.byte_length = data.data_length;
    buffer->base.memory_type = memory_type;
    buffer->base.allowed_access = allowed_access;
    buffer->base.allowed_usage = allowed_usage;
    buffer->data = data;
    buffer->data_allocator = data_allocator;
    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_heap_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_heap_buffer_t* buffer = (iree_hal_heap_buffer_t*)base_buffer;
  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(iree_hal_buffer_allocator(base_buffer));
  IREE_TRACE_ZONE_BEGIN(z0);

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
