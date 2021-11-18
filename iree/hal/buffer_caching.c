// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/buffer_caching.h"

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/allocator_caching.h"

extern const iree_hal_buffer_vtable_t iree_hal_caching_buffer_vtable;

static iree_hal_caching_buffer_t* iree_hal_caching_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_caching_buffer_vtable);
  return (iree_hal_caching_buffer_t*)base_value;
}

iree_status_t iree_hal_caching_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_caching_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(iree_hal_allocator_host_allocator(allocator),
                            sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_caching_buffer_vtable,
                                 &buffer->resource);
    buffer->delegate_buffer = *out_buffer;
    buffer->next = NULL;
    *out_buffer = (iree_hal_buffer_t*)buffer;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_caching_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_memory_type_t memory_type = iree_hal_buffer_memory_type(base_buffer);
  // Cache host visible device memory using caching allocator.
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) &&
      iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    iree_status_t status = iree_hal_allocator_add_buffer_to_cache(base_buffer);
    if (iree_status_is_ok(status)) {
      return;
    }
  }
  iree_hal_caching_buffer_t* buffer = iree_hal_caching_buffer_cast(base_buffer);
  iree_hal_buffer_destroy(buffer->delegate_buffer);
}

static iree_status_t iree_hal_caching_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode, iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    void** out_data_ptr) {
  iree_hal_caching_buffer_t* buffer = iree_hal_caching_buffer_cast(base_buffer);
  return iree_hal_buffer_map_range(buffer->delegate_buffer, memory_access,
      local_byte_offset, local_byte_length, *out_data_ptr);
}

static void iree_hal_caching_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, void* data_ptr) {
}

static iree_status_t iree_hal_caching_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do.
  return iree_ok_status();
}

static iree_status_t iree_hal_caching_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do.
  return iree_ok_status();
}

const iree_hal_buffer_vtable_t iree_hal_caching_buffer_vtable = {
    .destroy = iree_hal_caching_buffer_destroy,
    .map_range = iree_hal_caching_buffer_map_range,
    .unmap_range = iree_hal_caching_buffer_unmap_range,
    .invalidate_range = iree_hal_caching_buffer_invalidate_range,
    .flush_range = iree_hal_caching_buffer_flush_range,
};
