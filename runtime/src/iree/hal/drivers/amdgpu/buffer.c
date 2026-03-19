// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/buffer.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_buffer_t
//===----------------------------------------------------------------------===//

// A buffer backed by an HSA memory pool allocation.
// The allocation is host-visible (fine-grained) so map/unmap are trivial.
typedef struct iree_hal_amdgpu_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;

  // Unowned libhsa handle for freeing the allocation on destroy.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // HSA-allocated pointer. Accessible from both host and device when allocated
  // from a fine-grained pool, or device-only from a coarse-grained pool.
  void* host_ptr;
} iree_hal_amdgpu_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_amdgpu_buffer_vtable;

static iree_hal_amdgpu_buffer_t* iree_hal_amdgpu_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_buffer_vtable);
  return (iree_hal_amdgpu_buffer_t*)base_value;
}

iree_status_t iree_hal_amdgpu_buffer_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    void* host_ptr, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  iree_hal_amdgpu_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));
  iree_hal_buffer_initialize(placement, &buffer->base, allocation_size,
                             /*byte_offset=*/0,
                             /*byte_length=*/allocation_size, memory_type,
                             allowed_access, allowed_usage,
                             &iree_hal_amdgpu_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->libhsa = libhsa;
  buffer->host_ptr = host_ptr;

  *out_buffer = &buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdgpu_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_amdgpu_buffer_t* buffer = iree_hal_amdgpu_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Free the HSA allocation.
  if (buffer->host_ptr) {
    IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(buffer->libhsa),
                                                    buffer->host_ptr));
  }

  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_amdgpu_buffer_t* buffer = iree_hal_amdgpu_buffer_cast(base_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  // Fine-grained HSA allocations are directly host-accessible.
  mapping->contents = iree_make_byte_span(
      (uint8_t*)buffer->host_ptr + local_byte_offset, local_byte_length);

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  // Nothing to do — fine-grained memory is always coherent.
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do — fine-grained memory is always coherent.
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do — fine-grained memory is always coherent.
  return iree_ok_status();
}

static const iree_hal_buffer_vtable_t iree_hal_amdgpu_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_amdgpu_buffer_destroy,
    .map_range = iree_hal_amdgpu_buffer_map_range,
    .unmap_range = iree_hal_amdgpu_buffer_unmap_range,
    .invalidate_range = iree_hal_amdgpu_buffer_invalidate_range,
    .flush_range = iree_hal_amdgpu_buffer_flush_range,
};
