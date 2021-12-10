// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/webgpu/simple_allocator.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/webgpu/buffer.h"

typedef struct iree_hal_webgpu_simple_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  WGPUDevice device;
  iree_string_view_t identifier;
  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_webgpu_simple_allocator_t;

extern const iree_hal_allocator_vtable_t
    iree_hal_webgpu_simple_allocator_vtable;

static iree_hal_webgpu_simple_allocator_t*
iree_hal_webgpu_simple_allocator_cast(iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_simple_allocator_vtable);
  return (iree_hal_webgpu_simple_allocator_t*)base_value;
}

iree_status_t iree_hal_webgpu_simple_allocator_create(
    WGPUDevice device, iree_string_view_t identifier,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_simple_allocator_t* allocator = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*allocator), (void**)&allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_webgpu_simple_allocator_vtable,
                                 &allocator->resource);
    allocator->host_allocator = host_allocator;
    allocator->device = device;
    iree_string_view_append_to_buffer(
        identifier, &allocator->identifier,
        (char*)allocator + iree_sizeof_struct(*allocator));
    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_simple_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_webgpu_simple_allocator_t* allocator =
      iree_hal_webgpu_simple_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_webgpu_simple_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  iree_hal_webgpu_simple_allocator_t* allocator =
      (iree_hal_webgpu_simple_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static void iree_hal_webgpu_simple_allocator_query_statistics(
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_statistics_t* out_statistics) {
  IREE_STATISTICS({
    iree_hal_webgpu_simple_allocator_t* allocator =
        iree_hal_webgpu_simple_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_hal_buffer_compatibility_t
iree_hal_webgpu_simple_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_usage_t intended_usage,
    iree_device_size_t allocation_size) {
  // TODO(benvanik): check to ensure the allocator can serve the memory type.

  // Disallow usage not permitted by the buffer itself. Since we then use this
  // to determine compatibility below we'll naturally set the right compat flags
  // based on what's both allowed and intended.
  intended_usage &= allowed_usage;

  // All buffers can be allocated on the heap.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  // Buffers can only be used on the queue if they are device visible.
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_all_bits_set(intended_usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
    }
    if (iree_all_bits_set(intended_usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  return compatibility;
}

static iree_status_t iree_hal_webgpu_simple_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  iree_hal_webgpu_simple_allocator_t* allocator =
      iree_hal_webgpu_simple_allocator_cast(base_allocator);

  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (allocation_size == 0) allocation_size = 4;

  WGPUBufferUsageFlags usage_flags = WGPUBufferUsage_None;
  if (iree_all_bits_set(allowed_usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    usage_flags |= WGPUBufferUsage_CopySrc;
    usage_flags |= WGPUBufferUsage_CopyDst;
  }
  if (iree_all_bits_set(allowed_usage, IREE_HAL_BUFFER_USAGE_MAPPING)) {
    usage_flags |= WGPUBufferUsage_MapRead;
    usage_flags |= WGPUBufferUsage_MapWrite;
  }
  if (iree_all_bits_set(allowed_usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
    usage_flags |= WGPUBufferUsage_Storage;
    usage_flags |= WGPUBufferUsage_Uniform;
    usage_flags |= WGPUBufferUsage_Indirect;
  }
  iree_hal_memory_access_t allowed_access = IREE_HAL_MEMORY_ACCESS_ANY;
  if (iree_all_bits_set(allowed_usage, IREE_HAL_BUFFER_USAGE_CONSTANT)) {
    allowed_access = IREE_HAL_MEMORY_ACCESS_READ;
  }

  WGPUBufferDescriptor descriptor = {
      .nextInChain = NULL,
      .label = NULL,
      .usage = usage_flags,
      .size = allocation_size,
      .mappedAtCreation = false,
  };
  WGPUBuffer buffer_handle =
      wgpuDeviceCreateBuffer(allocator->device, &descriptor);
  if (!buffer_handle) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "unable to allocate buffer of size %zu",
                            allocation_size);
  }

  iree_status_t status = iree_hal_webgpu_buffer_wrap(
      allocator->device, base_allocator, memory_type, allowed_access,
      allowed_usage, allocation_size,
      /*byte_offset=*/0,
      /*byte_length=*/allocation_size, buffer_handle, allocator->host_allocator,
      out_buffer);

  if (iree_status_is_ok(status)) {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, memory_type, allocation_size));
  } else {
    wgpuBufferDestroy(buffer_handle);
  }
  return status;
}

static iree_status_t iree_hal_webgpu_simple_allocator_wrap_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "wrapping of external buffers not supported");
}

void iree_hal_webgpu_simple_allocator_free(iree_hal_allocator_t* base_allocator,
                                           iree_hal_memory_type_t memory_type,
                                           WGPUBuffer buffer_handle,
                                           iree_device_size_t allocation_size) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  IREE_ASSERT_ARGUMENT(buffer_handle);
  iree_hal_webgpu_simple_allocator_t* allocator =
      iree_hal_webgpu_simple_allocator_cast(base_allocator);

  // NOTE: this immediately destroys the buffer (in theory) and it must not be
  // in use. That's ok because we also have that requirement in the HAL.
  wgpuBufferDestroy(buffer_handle);

  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, memory_type, allocation_size));
}

const iree_hal_allocator_vtable_t iree_hal_webgpu_simple_allocator_vtable = {
    .destroy = iree_hal_webgpu_simple_allocator_destroy,
    .host_allocator = iree_hal_webgpu_simple_allocator_host_allocator,
    .query_statistics = iree_hal_webgpu_simple_allocator_query_statistics,
    .query_buffer_compatibility =
        iree_hal_webgpu_simple_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_webgpu_simple_allocator_allocate_buffer,
    .wrap_buffer = iree_hal_webgpu_simple_allocator_wrap_buffer,
};
