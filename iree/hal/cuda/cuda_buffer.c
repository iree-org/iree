// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cuda/cuda_buffer.h"

#include "iree/base/tracing.h"
#include "iree/hal/cuda/cuda_allocator.h"
#include "iree/hal/cuda/status_util.h"

typedef struct iree_hal_cuda_buffer_s {
  iree_hal_buffer_t base;
  void* host_ptr;
  CUdeviceptr device_ptr;
} iree_hal_cuda_buffer_t;

extern const iree_hal_buffer_vtable_t iree_hal_cuda_buffer_vtable;

static iree_hal_cuda_buffer_t* iree_hal_cuda_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_buffer_vtable);
  return (iree_hal_cuda_buffer_t*)base_value;
}

iree_status_t iree_hal_cuda_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    CUdeviceptr device_ptr, void* host_ptr, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(iree_hal_allocator_host_allocator(allocator),
                            sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_cuda_buffer_vtable,
                                 &buffer->base.resource);
    buffer->base.allocator = allocator;
    buffer->base.allocated_buffer = &buffer->base;
    buffer->base.allocation_size = allocation_size;
    buffer->base.byte_offset = byte_offset;
    buffer->base.byte_length = byte_length;
    buffer->base.memory_type = memory_type;
    buffer->base.allowed_access = allowed_access;
    buffer->base.allowed_usage = allowed_usage;
    buffer->host_ptr = host_ptr;
    buffer->device_ptr = device_ptr;
    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_cuda_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_cuda_buffer_t* buffer = iree_hal_cuda_buffer_cast(base_buffer);
  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(iree_hal_buffer_allocator(base_buffer));
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_allocator_free(buffer->base.allocator, buffer->device_ptr,
                               buffer->host_ptr, buffer->base.memory_type);
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    void** out_data_ptr) {
  iree_hal_cuda_buffer_t* buffer = iree_hal_cuda_buffer_cast(base_buffer);

  if (!iree_all_bits_set(buffer->base.memory_type,
                         IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "trying to map memory not host visible");
  }

  uint8_t* data_ptr = (uint8_t*)(buffer->host_ptr) + local_byte_offset;
  // If we mapped for discard scribble over the bytes. This is not a mandated
  // behavior but it will make debugging issues easier. Alternatively for
  // heap buffers we could reallocate them such that ASAN yells, but that
  // would only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(data_ptr + local_byte_offset, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG
  *out_data_ptr = data_ptr;
  return iree_ok_status();
}

static void iree_hal_cuda_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, void* data_ptr) {
  // nothing to do.
}

static iree_status_t iree_hal_cuda_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do.
  return iree_ok_status();
}

CUdeviceptr iree_hal_cuda_buffer_device_pointer(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_cuda_buffer_t* buffer = iree_hal_cuda_buffer_cast(base_buffer);
  return buffer->device_ptr;
}

const iree_hal_buffer_vtable_t iree_hal_cuda_buffer_vtable = {
    .destroy = iree_hal_cuda_buffer_destroy,
    .map_range = iree_hal_cuda_buffer_map_range,
    .unmap_range = iree_hal_cuda_buffer_unmap_range,
    .invalidate_range = iree_hal_cuda_buffer_invalidate_range,
    .flush_range = iree_hal_cuda_buffer_flush_range,
};
