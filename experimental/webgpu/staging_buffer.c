// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/staging_buffer.h"

#include <stdint.h>

#include "experimental/webgpu/buffer.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"

iree_status_t iree_hal_webgpu_staging_buffer_initialize(
    WGPUDevice device, const WGPULimits* limits,
    iree_hal_allocator_t* device_allocator, uint8_t* host_buffer,
    iree_host_size_t host_buffer_capacity,
    iree_hal_webgpu_staging_buffer_t* out_staging_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(host_buffer);
  IREE_ASSERT_ARGUMENT(out_staging_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_staging_buffer, 0, sizeof(*out_staging_buffer));

  if ((host_buffer_capacity % limits->minUniformBufferOffsetAlignment) != 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host buffer capacity (%zu) must match the buffer "
                            "offset alignment (%d)",
                            host_buffer_capacity,
                            limits->minUniformBufferOffsetAlignment);
  }

  out_staging_buffer->alignment = limits->minUniformBufferOffsetAlignment;
  out_staging_buffer->capacity = (uint32_t)host_buffer_capacity;
  out_staging_buffer->host_buffer = host_buffer;

  const iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ |
               IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
              IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
  };
  iree_hal_buffer_t* device_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_allocate_buffer(
              device_allocator, buffer_params, out_staging_buffer->capacity,
              iree_const_byte_span_empty(), &device_buffer));
  out_staging_buffer->device_buffer = device_buffer;
  iree_hal_buffer_retain(device_buffer);
  out_staging_buffer->device_buffer_handle =
      iree_hal_webgpu_buffer_handle(device_buffer);

  const WGPUBindGroupLayoutEntry buffer_bindings[] = {
      {
          .nextInChain = NULL,
          .binding = 0,
          .visibility = WGPUShaderStage_Compute,
          .buffer =
              {
                  .nextInChain = NULL,
                  .type = WGPUBufferBindingType_Uniform,
                  .hasDynamicOffset = true,
                  .minBindingSize = out_staging_buffer->alignment,
              },
      },
  };
  const WGPUBindGroupLayoutDescriptor group_layout_descriptor = {
      .nextInChain = NULL,
      .label = WGPU_DEBUG_LABEL("_staging_buffer_binding"),
      .entryCount = IREE_ARRAYSIZE(buffer_bindings),
      .entries = buffer_bindings,
  };
  out_staging_buffer->bind_group_layout =
      wgpuDeviceCreateBindGroupLayout(device, &group_layout_descriptor);

  const WGPUBindGroupEntry group_entries[] = {
      {
          .nextInChain = NULL,
          .binding = 0,
          .buffer = out_staging_buffer->device_buffer_handle,
          .offset = 0,
          .size = limits->maxUniformBufferBindingSize,
      },
  };
  const WGPUBindGroupDescriptor descriptor = {
      .nextInChain = NULL,
      .label = WGPU_DEBUG_LABEL("_staging_buffer"),
      .layout = out_staging_buffer->bind_group_layout,
      .entryCount = IREE_ARRAYSIZE(group_entries),
      .entries = group_entries,
  };
  out_staging_buffer->bind_group =
      wgpuDeviceCreateBindGroup(device, &descriptor);

  const WGPUBindGroupLayoutDescriptor empty_group_layout_descriptor = {
      .nextInChain = NULL,
      .label = WGPU_DEBUG_LABEL("_empty_binding"),
      .entryCount = 0,
      .entries = NULL,
  };
  out_staging_buffer->empty_bind_group_layout =
      wgpuDeviceCreateBindGroupLayout(device, &empty_group_layout_descriptor);
  const WGPUBindGroupDescriptor empty_descriptor = {
      .nextInChain = NULL,
      .label = WGPU_DEBUG_LABEL("_empty"),
      .layout = out_staging_buffer->empty_bind_group_layout,
      .entryCount = 0,
      .entries = NULL,
  };
  out_staging_buffer->empty_bind_group =
      wgpuDeviceCreateBindGroup(device, &empty_descriptor);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_webgpu_staging_buffer_deinitialize(
    iree_hal_webgpu_staging_buffer_t* staging_buffer) {
  iree_wgpuBindGroupLayoutDrop(staging_buffer->empty_bind_group_layout);
  iree_wgpuBindGroupDrop(staging_buffer->bind_group);
  iree_wgpuBindGroupLayoutDrop(staging_buffer->bind_group_layout);
  iree_wgpuBindGroupDrop(staging_buffer->empty_bind_group);
  iree_hal_buffer_release(staging_buffer->device_buffer);
}

iree_status_t iree_hal_webgpu_staging_buffer_reserve(
    iree_hal_webgpu_staging_buffer_t* staging_buffer, iree_host_size_t length,
    iree_byte_span_t* out_reservation, uint32_t* out_offset) {
  iree_host_size_t aligned_length =
      iree_host_align(length, staging_buffer->alignment);
  if (aligned_length > staging_buffer->capacity) {
    // Will never fit in the staging buffer.
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reservation (%" PRIhsz
                            ") exceeds the maximum capacity of "
                            "the staging buffer (%" PRIu32 ")",
                            length, staging_buffer->capacity);
  } else if (staging_buffer->offset + aligned_length >
             staging_buffer->capacity) {
    // Flush required - this is not an error but a request to the caller.
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  *out_reservation = iree_make_byte_span(
      staging_buffer->host_buffer + staging_buffer->offset, aligned_length);
  *out_offset = staging_buffer->offset;
  staging_buffer->offset += aligned_length;
  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_staging_buffer_append(
    iree_hal_webgpu_staging_buffer_t* staging_buffer,
    iree_const_byte_span_t source, uint32_t* out_offset) {
  iree_byte_span_t reservation;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_staging_buffer_reserve(
      staging_buffer, source.data_length, &reservation, out_offset));
  memcpy(reservation.data, source.data, source.data_length);
  return iree_ok_status();
}

void iree_hal_webgpu_staging_buffer_flush(
    iree_hal_webgpu_staging_buffer_t* staging_buffer, void** out_source_buffer,
    WGPUBuffer* out_target_buffer, iree_host_size_t* out_length) {
  *out_source_buffer = staging_buffer->host_buffer;
  *out_target_buffer = staging_buffer->device_buffer_handle;
  *out_length = staging_buffer->offset;
  staging_buffer->offset = 0;
}

void iree_hal_webgpu_staging_buffer_reset(
    iree_hal_webgpu_staging_buffer_t* staging_buffer) {
  staging_buffer->offset = 0;
}
