// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/null/buffer.h"

//===----------------------------------------------------------------------===//
// iree_hal_null_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_null_buffer_t {
  iree_hal_buffer_t base;
  iree_hal_buffer_release_callback_t release_callback;
} iree_hal_null_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_null_buffer_vtable;

static iree_hal_null_buffer_t* iree_hal_null_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_null_buffer_vtable);
  return (iree_hal_null_buffer_t*)base_value;
}

static const iree_hal_null_buffer_t* iree_hal_null_buffer_const_cast(
    const iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_null_buffer_vtable);
  return (const iree_hal_null_buffer_t*)base_value;
}

iree_status_t iree_hal_null_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  iree_hal_null_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));
  iree_hal_buffer_initialize(host_allocator, allocator, &buffer->base,
                             allocation_size, byte_offset, byte_length,
                             memory_type, allowed_access, allowed_usage,
                             &iree_hal_null_buffer_vtable, &buffer->base);
  buffer->release_callback = release_callback;

  // TODO(null): retain or take ownership of provided handles/pointers/etc.
  // Implementations may want to pass in an internal buffer type discriminator
  // if there are multiple or use different top-level iree_hal_buffer_t
  // implementations.
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "buffer wrapping not implemented");

  if (iree_status_is_ok(status)) {
    *out_buffer = &buffer->base;
  } else {
    iree_hal_buffer_release(&buffer->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_null_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_null_buffer_t* buffer = iree_hal_null_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Optionally call a release callback when the buffer is destroyed. Not all
  // implementations may require this but it's cheap and provides additional
  // flexibility.
  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }

  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_null_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_null_buffer_t* buffer = iree_hal_null_buffer_cast(base_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  // TODO(null): perform mapping as described. Note that local-to-buffer range
  // adjustment may be required. The resulting mapping is populated with
  // standard information such as contents indicating the host addressable
  // memory range of the mapped buffer and implementation-specific information
  // if additional resources are required. iree_hal_buffer_emulated_map_range
  // can be used by implementations that have no way of providing host pointers
  // at a large cost (alloc + device->host transfer on map and host->device
  // transfer + dealloc on umap). Try not to use that.
  (void)buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "buffer mapping not implemented");

  return status;
}

static iree_status_t iree_hal_null_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_null_buffer_t* buffer = iree_hal_null_buffer_cast(base_buffer);

  // TODO(null): reverse of map_range. Note that cache invalidation is explicit
  // via invalidate_range and need not be performed here. If using emulated
  // mapping this must call iree_hal_buffer_emulated_unmap_range to release the
  // transient resources.
  (void)buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "buffer mapping not implemented");

  return status;
}

static iree_status_t iree_hal_null_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_null_buffer_t* buffer = iree_hal_null_buffer_cast(base_buffer);

  // TODO(null): invalidate the range if required by the buffer. Writes on the
  // device are expected to be visible to the host after this returns.
  (void)buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "buffer mapping not implemented");

  return status;
}

static iree_status_t iree_hal_null_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_null_buffer_t* buffer = iree_hal_null_buffer_cast(base_buffer);

  // TODO(null): flush the range if required by the buffer. Writes on the
  // host are expected to be visible to the device after this returns.
  (void)buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "buffer mapping not implemented");

  return status;
}

static const iree_hal_buffer_vtable_t iree_hal_null_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_null_buffer_destroy,
    .map_range = iree_hal_null_buffer_map_range,
    .unmap_range = iree_hal_null_buffer_unmap_range,
    .invalidate_range = iree_hal_null_buffer_invalidate_range,
    .flush_range = iree_hal_null_buffer_flush_range,
};
