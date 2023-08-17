// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_METAL_STAGING_BUFFER_H_
#define IREE_HAL_DRIVERS_METAL_STAGING_BUFFER_H_

#import <Metal/Metal.h>

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Size, in bytes, of the shared storage mode staging buffer.
// The given amount of system memory will be allocated and is accessible to both
// the CPU and the GPU.
//
// Larger values here will use more memory but allow more concurrent/complex
// command buffers. As most models that run in these environments are only a few
// hundred dispatches per command buffer we can approximate an average
// consumption of 500 dispatches x worst-case 256B per dispatch of parameters
// and get 128KB.
#define IREE_HAL_METAL_STAGING_BUFFER_DEFAULT_CAPACITY (128 * 1024)

// A staging uniform buffer used for uploading parameters to the device.
// This allows for high-frequency writes of parameters at appropriate alignment.
//
// Intended usage is to retain one of these per device queue and use them during
// command buffer recording targeting that particular queue. This avoids
// allocating a lot of small buffers. The underlying buffer has shared storage
// mode; so it resides in system memory and is accessible to both the CPU and
// the GPU.
//
// Parameters handled by this buffer include:
// * Argument buffers for descriptor sets
// * Source buffer for buffer update commands
//
// Thread safe; multiple threads can reserve spaces concurrently.
typedef struct iree_hal_metal_staging_buffer_t {
  // Maximum number of bytes in the buffer.
  uint32_t capacity;

  // Device handle to the buffer.
  id<MTLBuffer> metal_buffer;
  // Host pointer to the buffer.
  uint8_t* host_buffer;

  // Non-recursive mutex guarding access to the offset field.
  iree_slim_mutex_t offset_mutex;

  // Current write offset of the device buffer.
  uint32_t offset IREE_GUARDED_BY(offset_mutex);

  // The number of command buffers that are being recorded or executed on
  // device. If this reaches zero, we know that there are no users of the
  // staging buffer so we can discard the contents and reset the offset to
  // zero.
  iree_atomic_int32_t pending_command_buffers;
} iree_hal_metal_staging_buffer_t;

// Initializes |out_staging_buffer| with the given |buffer_capacity|.
iree_status_t iree_hal_metal_staging_buffer_initialize(
    id<MTLDevice> device, iree_host_size_t buffer_capacity,
    iree_hal_metal_staging_buffer_t* out_staging_buffer);

void iree_hal_metal_staging_buffer_deinitialize(
    iree_hal_metal_staging_buffer_t* staging_buffer);

// Reserves |length| bytes from the staging buffer and returns a pointer to it
// in |out_reservation|.
iree_status_t iree_hal_metal_staging_buffer_reserve(
    iree_hal_metal_staging_buffer_t* staging_buffer, iree_host_size_t length,
    iree_host_size_t alignment, iree_byte_span_t* out_reservation,
    uint32_t* out_offset);

// Appends |data| of |length| bytes to the staging buffer.
iree_status_t iree_hal_metal_staging_buffer_append(
    iree_hal_metal_staging_buffer_t* staging_buffer,
    iree_const_byte_span_t source, iree_host_size_t alignment,
    uint32_t* out_offset);

// Resets the staging buffer to discard all its contents.
void iree_hal_metal_staging_buffer_reset(
    iree_hal_metal_staging_buffer_t* staging_buffer);

// Increases the command buffer using this staging buffer by one.
void iree_hal_metal_staging_buffer_increase_command_buffer_refcount(
    iree_hal_metal_staging_buffer_t* staging_buffer);

// Decreases the command buffer using this staging buffer by one, which may
// trigger reclaiming of resources.
void iree_hal_metal_staging_buffer_decrease_command_buffer_refcount(
    iree_hal_metal_staging_buffer_t* staging_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_METAL_STAGING_BUFFER_H_
