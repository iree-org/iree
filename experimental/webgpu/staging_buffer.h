// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_STAGING_BUFFER_H_
#define IREE_HAL_DRIVERS_WEBGPU_STAGING_BUFFER_H_

#include "experimental/webgpu/platform/webgpu.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Size, in bytes, of the device-local staging buffer.
// An equivalent amount of host memory will be reserved as a scratchpad used
// to build up the buffer prior to submission into the queue timeline.
//
// Larger values here will use more memory while reducing the frequency at which
// the staging buffer needs to be flushed. As most models that run in these
// environments are only a few hundred dispatches per command buffer we can
// approximate an average consumption of 500 dispatches x worst-case 256b per
// dispatch of parameters and get 128KB.
#define IREE_HAL_WEBGPU_STAGING_BUFFER_DEFAULT_CAPACITY (128 * 1024)

// A staging uniform buffer used for uploading parameters to the device.
// This allows for high-frequency writes of parameters at appropriate alignment.
//
// Intended usage is to retain one of these per device queue and use them during
// command buffer recording targeting that particular queue. Parameters are
// scribbled into the staging buffer host memory and then prior to submission
// an upload is scheduled from host->device into the device-local buffer. This
// puts the writes into queue timeline immediately before the commands that use
// it are submitted, and as there is only in-order execution per WebGPU queue
// this provides us a completely queue-ordered set of memory.
typedef struct iree_hal_webgpu_staging_buffer_t {
  // Alignment required on offsets into the buffer.
  // Uniform bindings with dynamic offsets must satisfy this alignment and on
  // some devices it can be as large as 256b.
  uint32_t alignment;
  // Maximum number of bytes in the buffer.
  uint32_t capacity;

  // Host-local buffer pointer.
  uint8_t* host_buffer;
  // Device-local HAL buffer - retains ownership.
  iree_hal_buffer_t* device_buffer;
  // Device-local buffer handle.
  WGPUBuffer device_buffer_handle;

  // Layout of a bind group with a single dynamic uniform buffer binding 0.
  WGPUBindGroupLayout bind_group_layout;
  // Bind group containing the buffer as dynamic at base offset 0.
  WGPUBindGroup bind_group;

  // Layout of an empty bind group, useful for padding within pipeline layouts.
  WGPUBindGroupLayout empty_bind_group_layout;
  // Empty bind group.
  WGPUBindGroup empty_bind_group;

  // Current write offset in the device buffer.
  uint32_t offset;
} iree_hal_webgpu_staging_buffer_t;

// Initializes |out_staging_buffer| using the given |host_buffer| memory.
iree_status_t iree_hal_webgpu_staging_buffer_initialize(
    WGPUDevice device, const WGPULimits* limits,
    iree_hal_allocator_t* device_allocator, uint8_t* host_buffer,
    iree_host_size_t host_buffer_capacity,
    iree_hal_webgpu_staging_buffer_t* out_staging_buffer);

void iree_hal_webgpu_staging_buffer_deinitialize(
    iree_hal_webgpu_staging_buffer_t* staging_buffer);

// Reserves |length| bytes from the staging buffer and returns a pointer to it
// in |out_reservation|.
// Returns RESOURCE_EXHAUSTED if the staging buffer is full and must be flushed
// with iree_hal_webgpu_staging_buffer_flush first.
iree_status_t iree_hal_webgpu_staging_buffer_reserve(
    iree_hal_webgpu_staging_buffer_t* staging_buffer, iree_host_size_t length,
    iree_byte_span_t* out_reservation, uint32_t* out_offset);

// Appends |data| of |length| bytes to the staging buffer.
// Returns RESOURCE_EXHAUSTED if the staging buffer is full and must be flushed
// with iree_hal_webgpu_staging_buffer_flush first.
iree_status_t iree_hal_webgpu_staging_buffer_append(
    iree_hal_webgpu_staging_buffer_t* staging_buffer,
    iree_const_byte_span_t source, uint32_t* out_offset);

// Flushes any pending uploads and returns the source buffer, target buffer,
// and length to upload. |out_length| may be 0 if there is nothing to flush.
void iree_hal_webgpu_staging_buffer_flush(
    iree_hal_webgpu_staging_buffer_t* staging_buffer, void** out_source_buffer,
    WGPUBuffer* out_target_buffer, iree_host_size_t* out_length);

// Resets the staging buffer to clear any pending writes.
void iree_hal_webgpu_staging_buffer_reset(
    iree_hal_webgpu_staging_buffer_t* staging_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_STAGING_BUFFER_H_
