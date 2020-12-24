// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/detail.h"

typedef struct iree_hal_heap_buffer_s {
  iree_hal_buffer_t base;

  iree_byte_span_t data;
  iree_allocator_t data_allocator;
} iree_hal_heap_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_heap_buffer_vtable;

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_heap_buffer_wrap(
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
  return iree_ok_status();
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

static iree_status_t iree_hal_heap_buffer_fill(
    iree_hal_buffer_t* base_buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_heap_buffer_t* buffer = (iree_hal_heap_buffer_t*)base_buffer;
  switch (pattern_length) {
    case 1: {
      uint8_t* data = (uint8_t*)(buffer->data.data + byte_offset);
      uint8_t value_bits = *(const uint8_t*)(pattern);
      memset(data, value_bits, byte_length);
      break;
    }
    case 2: {
      uint16_t* data = (uint16_t*)(buffer->data.data + byte_offset);
      uint16_t value_bits = *(const uint16_t*)(pattern);
      for (iree_device_size_t i = 0; i < byte_length / sizeof(uint16_t); ++i) {
        data[i] = value_bits;
      }
      break;
    }
    case 4: {
      uint32_t* data = (uint32_t*)(buffer->data.data + byte_offset);
      uint32_t value_bits = *(const uint32_t*)(pattern);
      for (iree_device_size_t i = 0; i < byte_length / sizeof(uint32_t); ++i) {
        data[i] = value_bits;
      }
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported fill pattern length: %zu",
                              pattern_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_heap_buffer_read_data(
    iree_hal_buffer_t* base_buffer, iree_device_size_t source_offset,
    void* target_buffer, iree_device_size_t data_length) {
  iree_hal_heap_buffer_t* buffer = (iree_hal_heap_buffer_t*)base_buffer;
  memcpy(target_buffer, buffer->data.data + source_offset, data_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_heap_buffer_write_data(
    iree_hal_buffer_t* base_buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_device_size_t data_length) {
  iree_hal_heap_buffer_t* buffer = (iree_hal_heap_buffer_t*)base_buffer;
  memcpy(buffer->data.data + target_offset, source_buffer, data_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_heap_buffer_copy_data(
    iree_hal_buffer_t* base_source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* base_target_buffer, iree_device_size_t target_offset,
    iree_device_size_t data_length) {
  // target_buffer is a heap buffer - source_buffer may be anything.
  iree_hal_heap_buffer_t* target_buffer =
      (iree_hal_heap_buffer_t*)base_target_buffer;
  void* target_ptr = target_buffer->data.data + target_offset;

  // We can avoid jumping through a bunch of hoops if we see the source/target
  // are from the same allocator (meaning they are both heap buffers).
  if (iree_hal_buffer_allocator(base_source_buffer) ==
      iree_hal_buffer_allocator(base_target_buffer)) {
    // Both are definitely heap buffers.
    iree_hal_heap_buffer_t* source_buffer =
        (iree_hal_heap_buffer_t*)base_source_buffer;
    memcpy(target_ptr, source_buffer->data.data + source_offset, data_length);
    return iree_ok_status();
  }

  // target_buffer is a heap buffer, source_buffer is anything. Map so that
  // we do the copy in the way most compatible with all backends.
  iree_hal_buffer_mapping_t source_mapping;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_range(base_source_buffer, IREE_HAL_MEMORY_ACCESS_READ,
                                source_offset, data_length, &source_mapping));
  memcpy(target_ptr, source_mapping.contents.data, data_length);
  return iree_hal_buffer_unmap_range(&source_mapping);
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

static iree_status_t iree_hal_heap_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, void* data_ptr) {
  // No-op here as we always have the pointer.
  return iree_ok_status();
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
    .fill = iree_hal_heap_buffer_fill,
    .read_data = iree_hal_heap_buffer_read_data,
    .write_data = iree_hal_heap_buffer_write_data,
    .copy_data = iree_hal_heap_buffer_copy_data,
    .map_range = iree_hal_heap_buffer_map_range,
    .unmap_range = iree_hal_heap_buffer_unmap_range,
    .invalidate_range = iree_hal_heap_buffer_invalidate_range,
    .flush_range = iree_hal_heap_buffer_flush_range,
};
