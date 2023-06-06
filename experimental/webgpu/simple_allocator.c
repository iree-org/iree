// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/simple_allocator.h"

#include <stddef.h>

#include "experimental/webgpu/buffer.h"
#include "experimental/webgpu/webgpu_device.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_webgpu_simple_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_device_t* device;
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
    iree_hal_device_t* device, iree_string_view_t identifier,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_simple_allocator_t* allocator = NULL;
  iree_host_size_t struct_size = iree_sizeof_struct(*allocator);
  iree_host_size_t total_size = struct_size + identifier.size;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_webgpu_simple_allocator_vtable,
                                 &allocator->resource);
    allocator->host_allocator = host_allocator;
    allocator->device = device;
    iree_string_view_append_to_buffer(identifier, &allocator->identifier,
                                      (char*)allocator + struct_size);
    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_simple_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_webgpu_simple_allocator_t* allocator =
      iree_hal_webgpu_simple_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_webgpu_simple_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_webgpu_simple_allocator_t* allocator =
      (iree_hal_webgpu_simple_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_webgpu_simple_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_webgpu_simple_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_webgpu_simple_allocator_t* allocator =
        iree_hal_webgpu_simple_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_hal_buffer_compatibility_t
iree_hal_webgpu_simple_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  // TODO(benvanik): check to ensure the allocator can serve the memory type.

  // All buffers can be allocated on the heap.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  // Buffers can only be used on the queue if they are device visible.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
    }
    if (iree_any_bit_set(params->usage,
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  // WebGPU does not support synchronous buffer mapping, so disallow.
  if (iree_all_bits_set(params->usage, IREE_HAL_BUFFER_USAGE_MAPPING)) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }

  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (*allocation_size == 0) *allocation_size = 4;

  return compatibility;
}

static iree_status_t iree_hal_webgpu_simple_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_host_size_t allocation_size, iree_const_byte_span_t initial_data,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  iree_hal_webgpu_simple_allocator_t* allocator =
      iree_hal_webgpu_simple_allocator_cast(base_allocator);

  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (allocation_size == 0) allocation_size = 4;

  WGPUBufferUsageFlags usage_flags = WGPUBufferUsage_None;
  if (iree_all_bits_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    usage_flags |= WGPUBufferUsage_CopySrc;
    usage_flags |= WGPUBufferUsage_CopyDst;
  }
  if (iree_all_bits_set(params->usage, IREE_HAL_BUFFER_USAGE_MAPPING)) {
    // Requirements from https://gpuweb.github.io/gpuweb/#buffer-usage:
    //   * MAP_WRITE can only be combined with COPY_SRC
    //   * MAP_READ  can only be combined with COPY_DST
    //
    // We don't have copy source/dest modeled in IREE's HAL (yet) so for now
    // we only enable mapping if transfer is set and hope it's not a copy dest.
    // Any copy dest buffers (such as for readback) must be allocated directly:
    //     WGPUBufferDescriptor descriptor = {
    //       ...
    //       .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
    //     };
    //     buffer = wgpuDeviceCreateBuffer(device, descriptor);
    //     iree_hal_webgpu_buffer_wrap(..., buffer, ...);
    if (iree_all_bits_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER) &&
        !iree_any_bit_set(params->usage,
                          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
      usage_flags |= WGPUBufferUsage_MapWrite;
      usage_flags &= ~(WGPUBufferUsage_CopyDst);  // Clear CopyDst
    }
  }
  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
    usage_flags |= WGPUBufferUsage_Storage;
  }
  if (iree_any_bit_set(params->usage,
                       IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ)) {
    usage_flags |= WGPUBufferUsage_Uniform;
  }
  if (iree_any_bit_set(params->usage,
                       IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMS)) {
    usage_flags |= WGPUBufferUsage_Indirect;
  }

  const bool has_initial_data = !iree_const_byte_span_is_empty(initial_data);
  WGPUBufferDescriptor descriptor = {
      .nextInChain = NULL,
      .label = NULL,
      .usage = usage_flags,
      .size = allocation_size,
      .mappedAtCreation = has_initial_data,
  };
  WGPUBuffer buffer_handle = wgpuDeviceCreateBuffer(
      iree_hal_webgpu_device_handle(allocator->device), &descriptor);
  if (!buffer_handle) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "unable to allocate buffer of size %" PRIdsz,
                            allocation_size);
  }

  // Upload the initial data into the mapped buffer. In WebGPU the only
  // _somewhat_ efficient path for buffer initialization is setting
  // mappedAtCreation and populating it before unmapping.
  if (has_initial_data) {
    IREE_TRACE_ZONE_BEGIN(z1);
    IREE_TRACE_ZONE_APPEND_VALUE(z1, (uint64_t)initial_data.data_length);
    void* mapped_ptr =
        wgpuBufferGetMappedRange(buffer_handle, 0, initial_data.data_length);
    memcpy(mapped_ptr, initial_data.data, initial_data.data_length);
    wgpuBufferUnmap(buffer_handle);
    IREE_TRACE_ZONE_END(z1);
  }

  iree_status_t status = iree_hal_webgpu_buffer_wrap(
      allocator->device, base_allocator, params->type, params->access,
      params->usage, allocation_size,
      /*byte_offset=*/0,
      /*byte_length=*/allocation_size, buffer_handle, allocator->host_allocator,
      out_buffer);
  if (iree_status_is_ok(status)) {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, params->type, allocation_size));
  } else {
    wgpuBufferDestroy(buffer_handle);
  }
  return status;
}

static void iree_hal_webgpu_simple_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_webgpu_simple_allocator_t* allocator =
      iree_hal_webgpu_simple_allocator_cast(base_allocator);

  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
      iree_hal_buffer_allocation_size(base_buffer)));

  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_webgpu_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "importing from external buffers not supported");
}

static iree_status_t iree_hal_webgpu_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "exporting to external buffers not supported");
}

const iree_hal_allocator_vtable_t iree_hal_webgpu_simple_allocator_vtable = {
    .destroy = iree_hal_webgpu_simple_allocator_destroy,
    .host_allocator = iree_hal_webgpu_simple_allocator_host_allocator,
    .trim = iree_hal_webgpu_simple_allocator_trim,
    .query_statistics = iree_hal_webgpu_simple_allocator_query_statistics,
    .query_buffer_compatibility =
        iree_hal_webgpu_simple_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_webgpu_simple_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_webgpu_simple_allocator_deallocate_buffer,
    .import_buffer = iree_hal_webgpu_allocator_import_buffer,
    .export_buffer = iree_hal_webgpu_allocator_export_buffer,
};
