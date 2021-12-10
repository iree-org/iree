// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/webgpu/buffer.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"

// TODO(benvanik): decouple via injection.
#include "iree/hal/webgpu/simple_allocator.h"

typedef struct iree_hal_webgpu_buffer_t {
  iree_hal_buffer_t base;
  WGPUDevice device;
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
    WGPUDevice device, iree_hal_allocator_t* allocator,
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    WGPUBuffer handle, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_webgpu_buffer_vtable,
                                 &buffer->base.resource);
    buffer->base.allocator = allocator;
    buffer->base.allocated_buffer = &buffer->base;
    buffer->base.allocation_size = allocation_size;
    buffer->base.byte_offset = byte_offset;
    buffer->base.byte_length = byte_length;
    buffer->base.memory_type = memory_type;
    buffer->base.allowed_access = allowed_access;
    buffer->base.allowed_usage = allowed_usage;
    buffer->device = device;
    buffer->handle = handle;
    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);
  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(iree_hal_buffer_allocator(base_buffer));
  IREE_TRACE_ZONE_BEGIN(z0);

  if (buffer->is_mapped) {
    wgpuBufferUnmap(buffer->handle);
  }

  iree_hal_webgpu_simple_allocator_free(
      buffer->base.allocator, buffer->base.memory_type, buffer->handle,
      buffer->base.allocation_size);
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
    void** out_data_ptr) {
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);

  if (!iree_all_bits_set(buffer->base.memory_type,
                         IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "trying to map memory not host visible");
  }

  if (!buffer->is_mapped) {
    WGPUMapMode map_mode = WGPUMapMode_None;
    if (iree_all_bits_set(mapping_mode, IREE_HAL_MEMORY_ACCESS_READ)) {
      map_mode |= WGPUMapMode_Read;
    }
    if (iree_all_bits_set(mapping_mode, IREE_HAL_MEMORY_ACCESS_WRITE)) {
      map_mode |= WGPUMapMode_Write;
    }
    IREEWGPUBufferMapSyncStatus sync_status = iree_wgpuBufferMapSync(
        buffer->device, buffer->handle, map_mode, (size_t)local_byte_offset,
        (size_t)local_byte_length);
    switch (sync_status) {
      case IREEWGPUBufferMapSyncStatus_Success:
        // Succeeded!
        break;
      case IREEWGPUBufferMapSyncStatus_Error:
        return iree_make_status(IREE_STATUS_INTERNAL, "failed to map buffer");
      default:
      case IREEWGPUBufferMapSyncStatus_Unknown:
        return iree_make_status(IREE_STATUS_UNKNOWN, "failed to map buffer");
      case IREEWGPUBufferMapSyncStatus_DeviceLost:
        return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "device lost while mapping buffer");
    }
  }

  uint8_t* data_ptr = NULL;
  if (iree_all_bits_set(memory_access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
    data_ptr = (uint8_t*)wgpuBufferGetMappedRange(
        buffer->handle, local_byte_offset, local_byte_length);
  } else {
    data_ptr = (uint8_t*)wgpuBufferGetConstMappedRange(
        buffer->handle, local_byte_offset, local_byte_length);
  }
  if (!data_ptr) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to get mapped buffer range");
  }

  // If we mapped for discard scribble over the bytes. This is not a mandated
  // behavior but it will make debugging issues easier. Alternatively for
  // heap buffers we could reallocate them such that ASAN yells, but that
  // would only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(data_ptr, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG

  *out_data_ptr = data_ptr;
  return iree_ok_status();
}

static void iree_hal_webgpu_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, void* data_ptr) {
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);
  wgpuBufferUnmap(buffer->handle);
  buffer->is_mapped = false;
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
