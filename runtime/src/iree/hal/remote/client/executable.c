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

typedef struct iree_hal_remote_client_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_remote_client_device_t* device;
  iree_hal_remote_resource_id_t resource_id;
  iree_host_size_t export_count;
} iree_hal_remote_client_executable_t;

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
  // Return default metadata. Full export info would require an
  // EXECUTABLE_QUERY_EXPORT RPC to the server.
  memset(out_info, 0, sizeof(*out_info));
  out_info->workgroup_size[0] = 1;
  out_info->workgroup_size[1] = 1;
  out_info->workgroup_size[2] = 1;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "export name lookup requires EXECUTABLE_QUERY_EXPORT "
                          "RPC (not yet implemented)");
}

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

  *out_executable = (iree_hal_executable_t*)executable;
  return iree_ok_status();
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
