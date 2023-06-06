// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/buffer.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "experimental/webgpu/webgpu_device.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/utils/buffer_transfer.h"

// TODO(benvanik): decouple via injection.
#include "experimental/webgpu/simple_allocator.h"

typedef struct iree_hal_webgpu_buffer_t {
  iree_hal_buffer_t base;
  iree_hal_device_t* device;  // unowned
  WGPUBuffer handle;
  bool is_mapped;
} iree_hal_webgpu_buffer_t;

extern const iree_hal_buffer_vtable_t iree_hal_webgpu_buffer_vtable;

static iree_hal_webgpu_buffer_t* iree_hal_webgpu_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_buffer_vtable);
  return (iree_hal_webgpu_buffer_t*)base_value;
}

iree_status_t iree_hal_webgpu_buffer_wrap(
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    WGPUBuffer handle, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_initialize(host_allocator, device_allocator, &buffer->base,
                               allocation_size, byte_offset, byte_length,
                               memory_type, allowed_access, allowed_usage,
                               &iree_hal_webgpu_buffer_vtable, &buffer->base);
    buffer->device = device;
    buffer->handle = handle;
    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (buffer->is_mapped) {
    wgpuBufferUnmap(buffer->handle);
  }

  // NOTE: this immediately destroys the buffer (in theory) and it must not be
  // in use. That's ok because we also have that requirement in the HAL.
  wgpuBufferDestroy(buffer->handle);

  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

WGPUBuffer iree_hal_webgpu_buffer_handle(const iree_hal_buffer_t* base_buffer) {
  iree_hal_webgpu_buffer_t* buffer =
      iree_hal_webgpu_buffer_cast((iree_hal_buffer_t*)base_buffer);
  IREE_ASSERT_ARGUMENT(buffer);
  return buffer->handle;
}

static iree_status_t iree_hal_webgpu_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  // WebGPU does not allow for synchronous buffer mapping.
  // Use wgpuBufferMapAsync directly to avoid this emulation.
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);
  return iree_hal_buffer_emulated_map_range(
      buffer->device, base_buffer, mapping_mode, memory_access,
      local_byte_offset, local_byte_length, mapping);
}

static iree_status_t iree_hal_webgpu_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  // WebGPU does not allow for synchronous buffer mapping.
  // Use wgpuBufferMapAsync directly to avoid this emulation.
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);
  return iree_hal_buffer_emulated_unmap_range(buffer->device, base_buffer,
                                              local_byte_offset,
                                              local_byte_length, mapping);
}

static iree_status_t iree_hal_webgpu_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do.
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do.
  return iree_ok_status();
}

const iree_hal_buffer_vtable_t iree_hal_webgpu_buffer_vtable = {
    .destroy = iree_hal_webgpu_buffer_destroy,
    .map_range = iree_hal_webgpu_buffer_map_range,
    .unmap_range = iree_hal_webgpu_buffer_unmap_range,
    .invalidate_range = iree_hal_webgpu_buffer_invalidate_range,
    .flush_range = iree_hal_webgpu_buffer_flush_range,
};
