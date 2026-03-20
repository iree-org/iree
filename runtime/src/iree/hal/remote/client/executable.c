// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/executable.h"

#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/protocol/control.h"

static const iree_hal_executable_vtable_t
    iree_hal_remote_client_executable_vtable;

//===----------------------------------------------------------------------===//
// Cached export metadata
//===----------------------------------------------------------------------===//

// Per-export cached metadata retrieved from the server via
// EXECUTABLE_QUERY_EXPORT RPCs at executable creation time.
typedef struct iree_hal_remote_cached_export_t {
  // Cached export info. name.data points into name_storage.
  iree_hal_executable_export_info_t info;
  // Heap-allocated name string storage.
  char* name_storage;
  // Heap-allocated parameter array. Count is info.parameter_count.
  iree_hal_executable_export_parameter_t* parameters;
} iree_hal_remote_cached_export_t;

typedef struct iree_hal_remote_client_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_remote_client_device_t* device;
  iree_hal_remote_resource_id_t resource_id;
  iree_host_size_t export_count;
  // Array of export_count cached export entries.
  // NULL if export_count == 0.
  iree_hal_remote_cached_export_t* cached_exports;
} iree_hal_remote_client_executable_t;

//===----------------------------------------------------------------------===//
// RPC helpers
//===----------------------------------------------------------------------===//

// Queries a single export's metadata from the server via
// EXECUTABLE_QUERY_EXPORT and populates the cached_export struct.
static iree_status_t iree_hal_remote_client_executable_query_export(
    iree_hal_remote_client_executable_t* executable, uint32_t ordinal,
    iree_hal_remote_cached_export_t* out_cached) {
  memset(out_cached, 0, sizeof(*out_cached));

  // Build request message: envelope + request body.
  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_executable_query_export_request_t request;
  } message;
  memset(&message, 0, sizeof(message));
  message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_EXECUTABLE_QUERY_EXPORT;
  message.request.executable_id = executable->resource_id;
  message.request.export_ordinal = ordinal;

  // Send RPC and wait for response.
  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_control_rpc(
      executable->device, iree_make_const_byte_span(&message, sizeof(message)),
      &response_payload, &response_lease));

  // Parse the response.
  if (response_payload.data_length <
      sizeof(iree_hal_remote_executable_query_export_response_t)) {
    iree_async_buffer_lease_release(&response_lease);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "EXECUTABLE_QUERY_EXPORT response too short: %" PRIhsz " bytes",
        response_payload.data_length);
  }

  const iree_hal_remote_executable_query_export_response_t* response =
      (const iree_hal_remote_executable_query_export_response_t*)
          response_payload.data;

  // Fill in export info.
  out_cached->info.flags =
      (iree_hal_executable_export_flags_t)response->flags;
  out_cached->info.constant_count = response->constant_count;
  out_cached->info.binding_count = response->binding_count;
  out_cached->info.parameter_count = response->parameter_count;
  out_cached->info.workgroup_size[0] = response->workgroup_size[0];
  out_cached->info.workgroup_size[1] = response->workgroup_size[1];
  out_cached->info.workgroup_size[2] = response->workgroup_size[2];
  memset(&out_cached->info.occupancy_info, 0,
         sizeof(out_cached->info.occupancy_info));

  // Extract name from variable-length tail.
  iree_host_size_t name_length = response->name_length;
  iree_host_size_t name_padded = iree_host_align(name_length, 8);
  const uint8_t* cursor =
      response_payload.data +
      sizeof(iree_hal_remote_executable_query_export_response_t);
  iree_host_size_t cursor_remaining =
      response_payload.data_length -
      sizeof(iree_hal_remote_executable_query_export_response_t);

  // Validate there's enough data for name + parameters.
  iree_host_size_t params_wire_size =
      response->parameter_count *
      sizeof(iree_hal_remote_export_parameter_wire_t);
  if (cursor_remaining < name_padded + params_wire_size) {
    iree_async_buffer_lease_release(&response_lease);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "EXECUTABLE_QUERY_EXPORT response data truncated");
  }

  // Copy and cache the name.
  if (name_length > 0) {
    iree_status_t status = iree_allocator_malloc(
        executable->host_allocator, name_length + 1,
        (void**)&out_cached->name_storage);
    if (!iree_status_is_ok(status)) {
      iree_async_buffer_lease_release(&response_lease);
      return status;
    }
    memcpy(out_cached->name_storage, cursor, name_length);
    out_cached->name_storage[name_length] = '\0';
    out_cached->info.name =
        iree_make_string_view(out_cached->name_storage, name_length);
  }
  cursor += name_padded;

  // Copy and cache parameters.
  if (response->parameter_count > 0) {
    iree_status_t status = iree_allocator_malloc(
        executable->host_allocator,
        response->parameter_count *
            sizeof(iree_hal_executable_export_parameter_t),
        (void**)&out_cached->parameters);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(executable->host_allocator,
                          out_cached->name_storage);
      out_cached->name_storage = NULL;
      iree_async_buffer_lease_release(&response_lease);
      return status;
    }
    memset(out_cached->parameters, 0,
           response->parameter_count *
               sizeof(iree_hal_executable_export_parameter_t));
    for (uint16_t i = 0; i < response->parameter_count; ++i) {
      const iree_hal_remote_export_parameter_wire_t* wire =
          (const iree_hal_remote_export_parameter_wire_t*)(cursor +
              i * sizeof(iree_hal_remote_export_parameter_wire_t));
      out_cached->parameters[i].type = wire->type;
      out_cached->parameters[i].flags = wire->flags;
      out_cached->parameters[i].offset = wire->offset;
      out_cached->parameters[i].size = wire->size;
      // Parameter names are not transmitted over the wire.
      out_cached->parameters[i].name = iree_string_view_empty();
    }
  }

  iree_async_buffer_lease_release(&response_lease);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Executable vtable implementation
//===----------------------------------------------------------------------===//

static void iree_hal_remote_client_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Fire-and-forget resource release to the server (same pattern as buffer).
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
  message.resource_ids[0] = executable->resource_id;
  iree_status_ignore(iree_hal_remote_client_device_send_fire_and_forget(
      executable->device,
      iree_make_const_byte_span(&message, sizeof(message))));

  // Free cached export metadata.
  if (executable->cached_exports) {
    for (iree_host_size_t i = 0; i < executable->export_count; ++i) {
      iree_allocator_free(executable->host_allocator,
                          executable->cached_exports[i].name_storage);
      iree_allocator_free(executable->host_allocator,
                          executable->cached_exports[i].parameters);
    }
    iree_allocator_free(executable->host_allocator,
                        executable->cached_exports);
  }

  iree_allocator_t host_allocator = executable->host_allocator;
  iree_allocator_free(host_allocator, executable);
  IREE_TRACE_ZONE_END(z0);
}

static iree_host_size_t iree_hal_remote_client_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  return executable->export_count;
}

static iree_status_t iree_hal_remote_client_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  if (export_ordinal >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %u >= export count %" PRIhsz,
                            export_ordinal, executable->export_count);
  }
  // Return cached metadata from the server query at creation time.
  *out_info = executable->cached_exports[export_ordinal].info;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  if (export_ordinal >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %u >= export count %" PRIhsz,
                            export_ordinal, executable->export_count);
  }
  const iree_hal_remote_cached_export_t* cached =
      &executable->cached_exports[export_ordinal];
  iree_host_size_t count = cached->info.parameter_count;
  if (count > capacity) count = capacity;
  if (count > 0 && cached->parameters) {
    memcpy(out_parameters, cached->parameters,
           count * sizeof(iree_hal_executable_export_parameter_t));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  // Linear search through cached export names.
  for (iree_host_size_t i = 0; i < executable->export_count; ++i) {
    if (iree_string_view_equal(executable->cached_exports[i].info.name,
                               name)) {
      *out_export_ordinal = (iree_hal_executable_export_ordinal_t)i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "export '%.*s' not found among %" PRIhsz " exports",
                          (int)name.size, name.data, executable->export_count);
}

//===----------------------------------------------------------------------===//
// Creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_remote_client_executable_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t resource_id, iree_host_size_t export_count,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  iree_hal_remote_client_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void**)&executable));
  iree_hal_resource_initialize(&iree_hal_remote_client_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->device = device;
  executable->resource_id = resource_id;
  executable->export_count = export_count;
  executable->cached_exports = NULL;

  // Allocate cache for export metadata.
  iree_status_t status = iree_ok_status();
  if (export_count > 0) {
    status = iree_allocator_malloc(
        host_allocator,
        export_count * sizeof(iree_hal_remote_cached_export_t),
        (void**)&executable->cached_exports);
    if (iree_status_is_ok(status)) {
      memset(executable->cached_exports, 0,
             export_count * sizeof(iree_hal_remote_cached_export_t));
    }

    // Query each export from the server and cache the results.
    for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < export_count;
         ++i) {
      status = iree_hal_remote_client_executable_query_export(
          executable, (uint32_t)i, &executable->cached_exports[i]);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    // Cleanup on failure.
    iree_hal_remote_client_executable_destroy(
        (iree_hal_executable_t*)executable);
  }
  return status;
}

iree_hal_remote_resource_id_t iree_hal_remote_client_executable_resource_id(
    iree_hal_executable_t* base_executable) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  return executable->resource_id;
}

static const iree_hal_executable_vtable_t
    iree_hal_remote_client_executable_vtable = {
        .destroy = iree_hal_remote_client_executable_destroy,
        .export_count = iree_hal_remote_client_executable_export_count,
        .export_info = iree_hal_remote_client_executable_export_info,
        .export_parameters =
            iree_hal_remote_client_executable_export_parameters,
        .lookup_export_by_name =
            iree_hal_remote_client_executable_lookup_export_by_name,
};
