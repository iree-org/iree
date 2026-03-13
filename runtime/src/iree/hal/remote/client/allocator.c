// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/allocator.h"

#include "iree/hal/remote/client/buffer.h"
#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/protocol/control.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_allocator_t
//===----------------------------------------------------------------------===//

static const iree_hal_allocator_vtable_t
    iree_hal_remote_client_allocator_vtable;

static iree_hal_remote_client_allocator_t*
iree_hal_remote_client_allocator_cast(iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_remote_client_allocator_vtable);
  return (iree_hal_remote_client_allocator_t*)base_value;
}

iree_status_t iree_hal_remote_client_allocator_create(
    iree_hal_remote_client_device_t* device, iree_string_view_t identifier,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_allocator = NULL;

  iree_host_size_t total_size = 0;
  iree_hal_remote_client_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(sizeof(*allocator), &total_size,
                             IREE_STRUCT_FIELD_FAM(identifier.size, char)));
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_remote_client_allocator_vtable,
                                 &allocator->resource);
    allocator->host_allocator = host_allocator;
    allocator->device = device;
    allocator->heaps = NULL;
    allocator->heap_count = 0;

    iree_string_view_append_to_buffer(identifier, &allocator->identifier,
                                      allocator->identifier_storage);

    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_remote_client_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = allocator->host_allocator;

  if (allocator->heaps) {
    iree_allocator_free(host_allocator, allocator->heaps);
  }
  iree_allocator_free(host_allocator, allocator);
  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_remote_client_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_remote_client_allocator_t* allocator =
      (iree_hal_remote_client_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_remote_client_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_remote_client_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
}

static iree_status_t iree_hal_remote_client_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);

  // If we don't have cached heaps, fetch from the server.
  if (!allocator->heaps) {
    // Build BUFFER_QUERY_HEAPS request.
    struct {
      iree_hal_remote_control_envelope_t envelope;
      iree_hal_remote_buffer_query_heaps_request_t request;
    } request_message;
    memset(&request_message, 0, sizeof(request_message));
    request_message.envelope.message_type =
        IREE_HAL_REMOTE_CONTROL_BUFFER_QUERY_HEAPS;

    iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
    iree_async_buffer_lease_t response_lease;
    memset(&response_lease, 0, sizeof(response_lease));
    IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_control_rpc(
        allocator->device,
        iree_make_const_byte_span(&request_message, sizeof(request_message)),
        &response_payload, &response_lease));

    // Parse response.
    iree_status_t status = iree_ok_status();
    if (response_payload.data_length <
        sizeof(iree_hal_remote_buffer_query_heaps_response_t)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "BUFFER_QUERY_HEAPS response too small: %" PRIhsz " bytes",
          response_payload.data_length);
    }

    iree_host_size_t heap_count = 0;
    if (iree_status_is_ok(status)) {
      const iree_hal_remote_buffer_query_heaps_response_t* response =
          (const iree_hal_remote_buffer_query_heaps_response_t*)
              response_payload.data;
      heap_count = response->heap_count;

      iree_host_size_t expected_size =
          sizeof(iree_hal_remote_buffer_query_heaps_response_t) +
          heap_count * sizeof(iree_hal_remote_memory_heap_t);
      if (response_payload.data_length < expected_size) {
        status =
            iree_make_status(IREE_STATUS_INTERNAL,
                             "BUFFER_QUERY_HEAPS response truncated: %" PRIhsz
                             " bytes, expected %" PRIhsz,
                             response_payload.data_length, expected_size);
      }
    }

    // Allocate and populate the cached heap array.
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(
          allocator->host_allocator,
          heap_count * sizeof(iree_hal_allocator_memory_heap_t),
          (void**)&allocator->heaps);
    }

    if (iree_status_is_ok(status)) {
      const iree_hal_remote_memory_heap_t* wire_heaps =
          (const iree_hal_remote_memory_heap_t*)(response_payload.data +
                                                 sizeof(
                                                     iree_hal_remote_buffer_query_heaps_response_t));
      for (iree_host_size_t i = 0; i < heap_count; ++i) {
        allocator->heaps[i].type = (iree_hal_memory_type_t)wire_heaps[i].type;
        allocator->heaps[i].allowed_usage =
            (iree_hal_buffer_usage_t)wire_heaps[i].allowed_usage;
        allocator->heaps[i].max_allocation_size =
            (iree_device_size_t)wire_heaps[i].max_allocation_size;
        allocator->heaps[i].min_alignment =
            (iree_device_size_t)wire_heaps[i].min_alignment;
      }
      allocator->heap_count = heap_count;
    }

    iree_async_buffer_lease_release(&response_lease);
    IREE_RETURN_IF_ERROR(status);
  }

  // Return cached heaps. Follow the HAL contract: always set out_count, then
  // return OUT_OF_RANGE if capacity < count (the standard pre-sizing pattern).
  if (out_count) *out_count = allocator->heap_count;
  if (capacity < allocator->heap_count) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  if (heaps) {
    memcpy(heaps, allocator->heaps,
           capacity * sizeof(iree_hal_allocator_memory_heap_t));
  }
  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_remote_client_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);

  // If heaps haven't been fetched yet, report as allocatable (optimistic).
  // The actual allocation will fail if the server doesn't support it.
  // We don't resolve params->type (e.g. clearing OPTIMAL) here because we
  // don't know what the server's heaps support. The server will resolve during
  // BUFFER_ALLOC.
  if (!allocator->heaps) {
    return IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
           IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER |
           IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
  }

  // Check against cached heaps.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  for (iree_host_size_t i = 0; i < allocator->heap_count; ++i) {
    if ((allocator->heaps[i].type & params->type) == params->type &&
        (allocator->heaps[i].allowed_usage & params->usage) == params->usage) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                       IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER |
                       IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
      break;
    }
  }

  return compatibility;
}

static iree_status_t iree_hal_remote_client_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  // Build BUFFER_ALLOC request.
  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_alloc_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type = IREE_HAL_REMOTE_CONTROL_BUFFER_ALLOC;
  request_message.request.provisional_id =
      IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, 0);
  request_message.request.params.usage = params->usage;
  request_message.request.params.access = (uint16_t)params->access;
  request_message.request.params.type = params->type;
  request_message.request.params.queue_affinity = params->queue_affinity;
  request_message.request.params.min_alignment =
      (uint64_t)params->min_alignment;
  request_message.request.allocation_size = (uint64_t)allocation_size;

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      allocator->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);

  // Parse response.
  iree_hal_remote_resource_id_t resolved_id = 0;
  if (iree_status_is_ok(status)) {
    if (response_payload.data_length <
        sizeof(iree_hal_remote_buffer_alloc_response_t)) {
      status =
          iree_make_status(IREE_STATUS_INTERNAL,
                           "BUFFER_ALLOC response too small: %" PRIhsz " bytes",
                           response_payload.data_length);
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_remote_buffer_alloc_response_t* response =
        (const iree_hal_remote_buffer_alloc_response_t*)response_payload.data;
    resolved_id = response->resolved_id;
  }

  iree_async_buffer_lease_release(&response_lease);

  // Create the local buffer proxy.
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_buffer_create(
        allocator->device, resolved_id, params, allocation_size,
        allocator->host_allocator, out_buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_remote_client_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer) {
  // The buffer's destroy callback handles release notification.
  // This is the default path when the HAL framework calls deallocate.
  iree_hal_buffer_destroy(buffer);
}

static iree_status_t iree_hal_remote_client_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote buffer import not yet implemented");
}

static iree_status_t iree_hal_remote_client_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote buffer export not yet implemented");
}

static const iree_hal_allocator_vtable_t
    iree_hal_remote_client_allocator_vtable = {
        .destroy = iree_hal_remote_client_allocator_destroy,
        .host_allocator = iree_hal_remote_client_allocator_host_allocator,
        .trim = iree_hal_remote_client_allocator_trim,
        .query_statistics = iree_hal_remote_client_allocator_query_statistics,
        .query_memory_heaps =
            iree_hal_remote_client_allocator_query_memory_heaps,
        .query_buffer_compatibility =
            iree_hal_remote_client_allocator_query_buffer_compatibility,
        .allocate_buffer = iree_hal_remote_client_allocator_allocate_buffer,
        .deallocate_buffer = iree_hal_remote_client_allocator_deallocate_buffer,
        .import_buffer = iree_hal_remote_client_allocator_import_buffer,
        .export_buffer = iree_hal_remote_client_allocator_export_buffer,
};
