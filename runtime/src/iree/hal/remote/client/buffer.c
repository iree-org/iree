// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/buffer.h"

#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/protocol/control.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_buffer_t
//===----------------------------------------------------------------------===//

static const iree_hal_buffer_vtable_t iree_hal_remote_client_buffer_vtable;

static void iree_hal_remote_client_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)base_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = buffer->host_allocator;

  // Send fire-and-forget RESOURCE_RELEASE_BATCH to the server.
  // Build a batch of 1 resource.
  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_resource_release_batch_t batch;
    iree_hal_remote_resource_id_t resource_ids[1];
  } message;
  memset(&message, 0, sizeof(message));
  message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_RESOURCE_RELEASE_BATCH;
  message.envelope.message_flags = IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET;
  message.batch.resource_count = 1;
  message.resource_ids[0] = buffer->resource_id;

  iree_const_byte_span_t message_span =
      iree_make_const_byte_span(&message, sizeof(message));
  iree_status_t status = iree_hal_remote_client_device_send_fire_and_forget(
      buffer->device, message_span);
  // Release is best-effort. If the session is already disconnected, the
  // server will clean up the resource when the session closes.
  iree_status_ignore(status);

  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_remote_client_buffer_map_range(
    iree_hal_buffer_t* buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote buffer map_range not yet implemented");
}

static iree_status_t iree_hal_remote_client_buffer_unmap_range(
    iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote buffer unmap_range not yet implemented");
}

static iree_status_t iree_hal_remote_client_buffer_invalidate_range(
    iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote buffer invalidate_range not yet implemented");
}

static iree_status_t iree_hal_remote_client_buffer_flush_range(
    iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote buffer flush_range not yet implemented");
}

static const iree_hal_buffer_vtable_t iree_hal_remote_client_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_remote_client_buffer_destroy,
    .map_range = iree_hal_remote_client_buffer_map_range,
    .unmap_range = iree_hal_remote_client_buffer_unmap_range,
    .invalidate_range = iree_hal_remote_client_buffer_invalidate_range,
    .flush_range = iree_hal_remote_client_buffer_flush_range,
};

iree_status_t iree_hal_remote_client_buffer_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t resource_id,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  iree_hal_remote_client_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_placement_t placement = {
        .device = (iree_hal_device_t*)device,
        .queue_affinity = params->queue_affinity ? params->queue_affinity
                                                 : IREE_HAL_QUEUE_AFFINITY_ANY,
    };
    iree_hal_buffer_initialize(
        placement, &buffer->base, allocation_size,
        /*byte_offset=*/0, /*byte_length=*/allocation_size, params->type,
        params->access, params->usage, &iree_hal_remote_client_buffer_vtable,
        &buffer->base);

    buffer->host_allocator = host_allocator;
    buffer->device = device;
    buffer->resource_id = resource_id;
    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
