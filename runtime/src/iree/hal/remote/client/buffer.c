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
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)base_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate local staging buffer. All access goes through this staging copy:
  // READ pulls data from server into staging, WRITE pushes staging to server
  // on unmap.
  uint8_t* staging = NULL;
  iree_status_t status = iree_allocator_malloc(
      buffer->host_allocator, (iree_host_size_t)local_byte_length,
      (void**)&staging);

  // If READ access, pull current buffer contents from the server.
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(memory_access, IREE_HAL_MEMORY_ACCESS_READ)) {
    struct {
      iree_hal_remote_control_envelope_t envelope;
      iree_hal_remote_buffer_map_request_t body;
    } request;
    memset(&request, 0, sizeof(request));
    request.envelope.message_type = IREE_HAL_REMOTE_CONTROL_BUFFER_MAP;
    request.body.buffer_id = buffer->resource_id;
    request.body.memory_access = (uint32_t)memory_access;
    request.body.offset = local_byte_offset;
    request.body.length = local_byte_length;

    iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
    iree_async_buffer_lease_t response_lease;
    memset(&response_lease, 0, sizeof(response_lease));
    status = iree_hal_remote_client_device_control_rpc(
        buffer->device, iree_make_const_byte_span(&request, sizeof(request)),
        &response_payload, &response_lease);

    if (iree_status_is_ok(status)) {
      if (response_payload.data_length <
          sizeof(iree_hal_remote_buffer_map_response_t)) {
        status =
            iree_make_status(IREE_STATUS_INTERNAL,
                             "BUFFER_MAP response truncated: %" PRIhsz " bytes",
                             response_payload.data_length);
      } else {
        const iree_hal_remote_buffer_map_response_t* response =
            (const iree_hal_remote_buffer_map_response_t*)response_payload.data;
        const uint8_t* response_data =
            response_payload.data + sizeof(*response);
        iree_host_size_t data_available =
            response_payload.data_length - sizeof(*response);
        if (data_available < local_byte_length) {
          status =
              iree_make_status(IREE_STATUS_INTERNAL,
                               "BUFFER_MAP response data too short: %" PRIhsz
                               " bytes, expected %" PRIdsz,
                               data_available, local_byte_length);
        } else {
          memcpy(staging, response_data, (iree_host_size_t)local_byte_length);
        }
      }
      iree_async_buffer_lease_release(&response_lease);
    }
  }

  if (iree_status_is_ok(status)) {
    mapping->contents =
        iree_make_byte_span(staging, (iree_host_size_t)local_byte_length);
    // Store mapping state so flush_range can find the staging data.
    buffer->active_mapping_data = staging;
    buffer->active_mapping_offset = local_byte_offset;
    buffer->active_mapping_length = local_byte_length;
  } else {
    iree_allocator_free(buffer->host_allocator, staging);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_remote_client_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)base_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // If WRITE access was used, push the staging data to the server.
  if (iree_all_bits_set(mapping->impl.allowed_access,
                        IREE_HAL_MEMORY_ACCESS_WRITE)) {
    // Build request: envelope + unmap header + inline data. The data can be
    // arbitrarily large, so heap-allocate the entire request.
    iree_host_size_t header_size =
        sizeof(iree_hal_remote_control_envelope_t) +
        sizeof(iree_hal_remote_buffer_unmap_request_t);
    iree_host_size_t data_size = (iree_host_size_t)local_byte_length;
    iree_host_size_t request_size = header_size + data_size;

    uint8_t* request_buffer = NULL;
    status = iree_allocator_malloc(buffer->host_allocator, request_size,
                                   (void**)&request_buffer);
    if (iree_status_is_ok(status)) {
      memset(request_buffer, 0, header_size);

      iree_hal_remote_control_envelope_t* envelope =
          (iree_hal_remote_control_envelope_t*)request_buffer;
      envelope->message_type = IREE_HAL_REMOTE_CONTROL_BUFFER_UNMAP;

      iree_hal_remote_buffer_unmap_request_t* body =
          (iree_hal_remote_buffer_unmap_request_t*)(request_buffer +
                                                    sizeof(*envelope));
      body->buffer_id = buffer->resource_id;
      body->offset = local_byte_offset;
      body->length = local_byte_length;

      // Copy staging data after the header.
      memcpy(request_buffer + header_size, mapping->contents.data, data_size);

      iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
      iree_async_buffer_lease_t response_lease;
      memset(&response_lease, 0, sizeof(response_lease));
      status = iree_hal_remote_client_device_control_rpc(
          buffer->device,
          iree_make_const_byte_span(request_buffer, request_size),
          &response_payload, &response_lease);
      iree_async_buffer_lease_release(&response_lease);
      iree_allocator_free(buffer->host_allocator, request_buffer);
    }
  }

  // Clear active mapping state and free the staging buffer.
  buffer->active_mapping_data = NULL;
  buffer->active_mapping_offset = 0;
  buffer->active_mapping_length = 0;
  iree_allocator_free(buffer->host_allocator, mapping->contents.data);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Invalidate is a no-op: the next map_range(READ) will pull fresh data.
static iree_status_t iree_hal_remote_client_buffer_invalidate_range(
    iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  return iree_ok_status();
}

// Flush pushes the dirty mapping data to the server without unmapping.
// Uses the active mapping state stored on the buffer during map_range
// to locate the staging data for the specified range.
static iree_status_t iree_hal_remote_client_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)base_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!buffer->active_mapping_data) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "flush_range called without an active mapping");
  }

  // Calculate the offset within the staging buffer. local_byte_offset is
  // absolute within the buffer; the staging buffer starts at
  // active_mapping_offset.
  iree_device_size_t staging_offset =
      local_byte_offset - buffer->active_mapping_offset;
  const uint8_t* staging_data =
      buffer->active_mapping_data + (iree_host_size_t)staging_offset;

  iree_host_size_t header_size = sizeof(iree_hal_remote_control_envelope_t) +
                                 sizeof(iree_hal_remote_buffer_unmap_request_t);
  iree_host_size_t data_size = (iree_host_size_t)local_byte_length;
  iree_host_size_t request_size = 0;
  if (!iree_host_size_checked_add(header_size, data_size, &request_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "flush request size overflow");
  }

  uint8_t* request_buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      buffer->host_allocator, request_size, (void**)&request_buffer);
  if (iree_status_is_ok(status)) {
    memset(request_buffer, 0, header_size);

    iree_hal_remote_control_envelope_t* envelope =
        (iree_hal_remote_control_envelope_t*)request_buffer;
    envelope->message_type = IREE_HAL_REMOTE_CONTROL_BUFFER_UNMAP;

    iree_hal_remote_buffer_unmap_request_t* body =
        (iree_hal_remote_buffer_unmap_request_t*)(request_buffer +
                                                  sizeof(*envelope));
    body->buffer_id = buffer->resource_id;
    body->offset = local_byte_offset;
    body->length = local_byte_length;

    memcpy(request_buffer + header_size, staging_data, data_size);

    iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
    iree_async_buffer_lease_t response_lease;
    memset(&response_lease, 0, sizeof(response_lease));
    status = iree_hal_remote_client_device_control_rpc(
        buffer->device, iree_make_const_byte_span(request_buffer, request_size),
        &response_payload, &response_lease);
    iree_async_buffer_lease_release(&response_lease);
    iree_allocator_free(buffer->host_allocator, request_buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
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
