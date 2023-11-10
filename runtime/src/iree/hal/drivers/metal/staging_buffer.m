// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/staging_buffer.h"

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"

iree_status_t iree_hal_metal_staging_buffer_initialize(
    id<MTLDevice> device, iree_host_size_t buffer_capacity,
    iree_hal_metal_staging_buffer_t* out_staging_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_staging_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_staging_buffer, 0, sizeof(*out_staging_buffer));

  // From Metal Best Practices Guide:
  // "For small-sized data that changes frequently, choose the Shared mode. The overhead of copying
  // data to video memory may be more expensive than the overhead of the GPU accessing system memory
  // directly."
  MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
  id<MTLBuffer> metal_buffer = [device newBufferWithLength:buffer_capacity options:options];  // +1
  if (!metal_buffer) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to allocate staging buffer with size = %ld bytes",
                            buffer_capacity);
  }

  out_staging_buffer->capacity = (uint32_t)buffer_capacity;
  out_staging_buffer->metal_buffer = metal_buffer;
  out_staging_buffer->host_buffer = metal_buffer.contents;
  iree_slim_mutex_initialize(&out_staging_buffer->offset_mutex);
  out_staging_buffer->offset = 0;
  iree_atomic_store_int32(&out_staging_buffer->pending_command_buffers, 0,
                          iree_memory_order_relaxed);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_metal_staging_buffer_deinitialize(iree_hal_metal_staging_buffer_t* staging_buffer) {
  iree_slim_mutex_deinitialize(&staging_buffer->offset_mutex);
  [staging_buffer->metal_buffer release];  // -1
}

iree_status_t iree_hal_metal_staging_buffer_reserve(iree_hal_metal_staging_buffer_t* staging_buffer,
                                                    iree_host_size_t length,
                                                    iree_host_size_t alignment,
                                                    iree_byte_span_t* out_reservation,
                                                    uint32_t* out_offset) {
  if (length > staging_buffer->capacity) {
    // This will never fit in the staging buffer.
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reservation (%" PRIhsz " bytes) exceeds the maximum capacity of "
                            "the staging buffer (%" PRIu32 " bytes)",
                            length, staging_buffer->capacity);
  }

  iree_slim_mutex_lock(&staging_buffer->offset_mutex);
  uint32_t aligned_offset = iree_host_align(staging_buffer->offset, alignment);
  if (aligned_offset + length > staging_buffer->capacity) {
    iree_slim_mutex_unlock(&staging_buffer->offset_mutex);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to reserve %" PRIhsz " bytes in staging buffer", length);
  }
  staging_buffer->offset = aligned_offset + length;
  iree_slim_mutex_unlock(&staging_buffer->offset_mutex);

  *out_reservation = iree_make_byte_span(staging_buffer->host_buffer + aligned_offset, length);
  *out_offset = aligned_offset;

  return iree_ok_status();
}

iree_status_t iree_hal_metal_staging_buffer_append(iree_hal_metal_staging_buffer_t* staging_buffer,
                                                   iree_const_byte_span_t source,
                                                   iree_host_size_t alignment,
                                                   uint32_t* out_offset) {
  iree_byte_span_t reservation;
  IREE_RETURN_IF_ERROR(iree_hal_metal_staging_buffer_reserve(staging_buffer, source.data_length,
                                                             alignment, &reservation, out_offset));
  memcpy(reservation.data, source.data, source.data_length);
  return iree_ok_status();
}

void iree_hal_metal_staging_buffer_reset(iree_hal_metal_staging_buffer_t* staging_buffer) {
  iree_slim_mutex_lock(&staging_buffer->offset_mutex);
  staging_buffer->offset = 0;
  iree_slim_mutex_unlock(&staging_buffer->offset_mutex);
}

void iree_hal_metal_staging_buffer_increase_command_buffer_refcount(
    iree_hal_metal_staging_buffer_t* staging_buffer) {
  iree_atomic_fetch_add_int32(&staging_buffer->pending_command_buffers, 1,
                              iree_memory_order_relaxed);
}

void iree_hal_metal_staging_buffer_decrease_command_buffer_refcount(
    iree_hal_metal_staging_buffer_t* staging_buffer) {
  if (iree_atomic_fetch_sub_int32(&staging_buffer->pending_command_buffers, 1,
                                  iree_memory_order_acq_rel) == 1) {
    iree_hal_metal_staging_buffer_reset(staging_buffer);
  }
}
