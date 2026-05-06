// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_executable.h"

#include <string.h>

#include "iree/hal/replay/recorder_record.h"

#define IREE_HAL_REPLAY_VTABLE_DISPATCH(resource, type_prefix, method_name) \
  ((const type_prefix##_vtable_t*)((const iree_hal_resource_t*)(resource))  \
       ->vtable)                                                            \
      ->method_name

//===----------------------------------------------------------------------===//
// iree_hal_replay_recorder_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_replay_recorder_executable_t {
  // HAL resource header for the recording wrapper executable.
  iree_hal_resource_t resource;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
  // Shared recorder receiving all captured operations.
  iree_hal_replay_recorder_t* recorder;
  // Underlying executable receiving forwarded HAL calls.
  iree_hal_executable_t* base_executable;
  // Session-local device object id associated with this executable.
  iree_hal_replay_object_id_t device_id;
  // Session-local object id assigned to this executable.
  iree_hal_replay_object_id_t executable_id;
} iree_hal_replay_recorder_executable_t;

static const iree_hal_executable_vtable_t
    iree_hal_replay_recorder_executable_vtable;

static bool iree_hal_replay_recorder_executable_isa(
    iree_hal_executable_t* base_executable) {
  return iree_hal_resource_is(base_executable,
                              &iree_hal_replay_recorder_executable_vtable);
}

static iree_hal_replay_recorder_executable_t*
iree_hal_replay_recorder_executable_cast(
    iree_hal_executable_t* base_executable) {
  IREE_HAL_ASSERT_TYPE(base_executable,
                       &iree_hal_replay_recorder_executable_vtable);
  return (iree_hal_replay_recorder_executable_t*)base_executable;
}

iree_hal_executable_t* iree_hal_replay_recorder_executable_base_or_self(
    iree_hal_executable_t* executable) {
  return iree_hal_replay_recorder_executable_isa(executable)
             ? iree_hal_replay_recorder_executable_cast(executable)
                   ->base_executable
             : executable;
}

iree_hal_replay_object_id_t iree_hal_replay_recorder_executable_id_or_none(
    iree_hal_executable_t* executable) {
  return iree_hal_replay_recorder_executable_isa(executable)
             ? iree_hal_replay_recorder_executable_cast(executable)
                   ->executable_id
             : IREE_HAL_REPLAY_OBJECT_ID_NONE;
}

static iree_status_t iree_hal_replay_recorder_executable_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t executable_id,
    iree_hal_executable_t* base_executable, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_executable);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  iree_hal_replay_recorder_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void**)&executable));
  memset(executable, 0, sizeof(*executable));
  iree_hal_resource_initialize(&iree_hal_replay_recorder_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->recorder = recorder;
  iree_hal_replay_recorder_retain(executable->recorder);
  executable->base_executable = base_executable;
  iree_hal_executable_retain(executable->base_executable);
  executable->device_id = device_id;
  executable->executable_id = executable_id;

  *out_executable = (iree_hal_executable_t*)executable;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_recorder_executable_begin_operation(
    iree_hal_replay_recorder_executable_t* executable,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_pending_record_t* out_pending_record) {
  return iree_hal_replay_recorder_begin_operation(
      executable->recorder, executable->device_id, executable->executable_id,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE,
      operation_code, payload_type, out_pending_record);
}

static void iree_hal_replay_recorder_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_executable_release(executable->base_executable);
  iree_hal_replay_recorder_release(executable->recorder);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_host_size_t iree_hal_replay_recorder_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  iree_hal_replay_pending_record_t pending_record;
  iree_status_t status = iree_hal_replay_recorder_executable_begin_operation(
      executable, IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_EXPORT_COUNT,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record);
  iree_host_size_t count =
      iree_hal_executable_export_count(executable->base_executable);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_end_operation(&pending_record,
                                                    iree_ok_status());
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_replay_recorder_fail(executable->recorder, status);
  }
  return count;
}

static iree_status_t iree_hal_replay_recorder_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_executable_begin_operation(
      executable, IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_EXPORT_INFO,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_executable_export_info(executable->base_executable,
                                      export_ordinal, out_info));
}

static iree_status_t iree_hal_replay_recorder_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_executable_begin_operation(
      executable, IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_EXPORT_PARAMETERS,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record, iree_hal_executable_export_parameters(
                           executable->base_executable, export_ordinal,
                           capacity, out_parameters));
}

static iree_status_t iree_hal_replay_recorder_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_executable_begin_operation(
      executable,
      IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_LOOKUP_EXPORT_BY_NAME,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_executable_lookup_export_by_name(executable->base_executable,
                                                name, out_export_ordinal));
}

static iree_status_t iree_hal_replay_recorder_executable_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  (void)executable;
  (void)name;
  (void)queue_affinity;
  *out_buffer = NULL;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "replay recording of executable global lookup is not implemented");
}

//===----------------------------------------------------------------------===//
// iree_hal_replay_recorder_executable_cache_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_replay_recorder_executable_cache_t {
  // HAL resource header for the recording wrapper executable cache.
  iree_hal_resource_t resource;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
  // Shared recorder receiving all captured operations.
  iree_hal_replay_recorder_t* recorder;
  // Underlying executable cache receiving forwarded HAL calls.
  iree_hal_executable_cache_t* base_executable_cache;
  // Session-local device object id associated with this executable cache.
  iree_hal_replay_object_id_t device_id;
  // Session-local object id assigned to this executable cache.
  iree_hal_replay_object_id_t executable_cache_id;
} iree_hal_replay_recorder_executable_cache_t;

static const iree_hal_executable_cache_vtable_t
    iree_hal_replay_recorder_executable_cache_vtable;

static iree_hal_replay_recorder_executable_cache_t*
iree_hal_replay_recorder_executable_cache_cast(
    iree_hal_executable_cache_t* base_executable_cache) {
  IREE_HAL_ASSERT_TYPE(base_executable_cache,
                       &iree_hal_replay_recorder_executable_cache_vtable);
  return (iree_hal_replay_recorder_executable_cache_t*)base_executable_cache;
}

iree_status_t iree_hal_replay_recorder_executable_cache_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t executable_cache_id,
    iree_hal_executable_cache_t* base_executable_cache,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_executable_cache);
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;

  iree_hal_replay_recorder_executable_cache_t* executable_cache = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executable_cache), (void**)&executable_cache));
  memset(executable_cache, 0, sizeof(*executable_cache));
  iree_hal_resource_initialize(
      &iree_hal_replay_recorder_executable_cache_vtable,
      &executable_cache->resource);
  executable_cache->host_allocator = host_allocator;
  executable_cache->recorder = recorder;
  iree_hal_replay_recorder_retain(executable_cache->recorder);
  executable_cache->base_executable_cache = base_executable_cache;
  iree_hal_executable_cache_retain(executable_cache->base_executable_cache);
  executable_cache->device_id = device_id;
  executable_cache->executable_cache_id = executable_cache_id;

  *out_executable_cache = (iree_hal_executable_cache_t*)executable_cache;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_recorder_executable_cache_begin_operation(
    iree_hal_replay_recorder_executable_cache_t* executable_cache,
    iree_hal_replay_object_id_t related_object_id,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_pending_record_t* out_pending_record) {
  return iree_hal_replay_recorder_begin_operation(
      executable_cache->recorder, executable_cache->device_id,
      executable_cache->executable_cache_id, related_object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE_CACHE, operation_code,
      payload_type, out_pending_record);
}

static void iree_hal_replay_recorder_executable_cache_destroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  iree_hal_replay_recorder_executable_cache_t* executable_cache =
      iree_hal_replay_recorder_executable_cache_cast(base_executable_cache);
  iree_allocator_t host_allocator = executable_cache->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_executable_cache_release(executable_cache->base_executable_cache);
  iree_hal_replay_recorder_release(executable_cache->recorder);
  iree_allocator_free(host_allocator, executable_cache);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_replay_recorder_executable_cache_infer_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  iree_hal_replay_recorder_executable_cache_t* executable_cache =
      iree_hal_replay_recorder_executable_cache_cast(base_executable_cache);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_recorder_executable_cache_begin_operation(
          executable_cache, IREE_HAL_REPLAY_OBJECT_ID_NONE,
          IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_CACHE_INFER_FORMAT,
          IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_executable_cache_infer_format(
          executable_cache->base_executable_cache, caching_mode,
          executable_data, executable_format_capacity, executable_format,
          out_inferred_size));
}

static bool iree_hal_replay_recorder_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  iree_hal_replay_recorder_executable_cache_t* executable_cache =
      iree_hal_replay_recorder_executable_cache_cast(base_executable_cache);
  return iree_hal_executable_cache_can_prepare_format(
      executable_cache->base_executable_cache, caching_mode, executable_format);
}

static iree_status_t iree_hal_replay_recorder_prepare_payload_iovecs(
    const iree_hal_executable_params_t* executable_params,
    iree_const_byte_span_t executable_metadata,
    iree_hal_replay_executable_prepare_payload_t* out_payload,
    iree_const_byte_span_t out_iovecs[5]) {
  memset(out_payload, 0, sizeof(*out_payload));
  out_payload->queue_affinity = executable_params->queue_affinity;
  out_payload->executable_data_length =
      executable_params->executable_data.data_length;
  out_payload->constant_count = executable_params->constant_count;
  out_payload->caching_mode = executable_params->caching_mode;
  out_payload->executable_format_length =
      executable_params->executable_format.size;
  if (IREE_UNLIKELY(executable_metadata.data_length > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable metadata byte length overflow");
  }
  out_payload->executable_metadata_length =
      (uint32_t)executable_metadata.data_length;

  if (IREE_UNLIKELY(executable_params->constant_count > 0 &&
                    !executable_params->constants)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable constants are required");
  }
  iree_host_size_t constant_bytes = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(executable_params->constant_count,
                                      sizeof(uint32_t), &constant_bytes))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable constant byte length overflow");
  }

  out_iovecs[0] = iree_make_const_byte_span(out_payload, sizeof(*out_payload));
  out_iovecs[1] =
      iree_make_const_byte_span(executable_params->executable_format.data,
                                executable_params->executable_format.size);
  out_iovecs[2] = executable_params->executable_data;
  out_iovecs[3] =
      iree_make_const_byte_span(executable_params->constants, constant_bytes);
  out_iovecs[4] = executable_metadata;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_recorder_capture_executable_metadata(
    iree_hal_executable_t* executable, iree_allocator_t host_allocator,
    iree_byte_span_t* out_storage, iree_const_byte_span_t* out_metadata) {
  IREE_ASSERT_ARGUMENT(out_storage);
  IREE_ASSERT_ARGUMENT(out_metadata);
  *out_storage = iree_byte_span_empty();
  *out_metadata = iree_const_byte_span_empty();

  const iree_host_size_t export_count =
      iree_hal_executable_export_count(executable);
  iree_host_size_t export_info_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          export_count, sizeof(iree_hal_executable_export_info_t),
          &export_info_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable export metadata size overflow");
  }

  iree_hal_executable_export_info_t* export_infos = NULL;
  if (export_info_size > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, export_info_size,
                                               (void**)&export_infos));
  }
  iree_status_t status = iree_ok_status();
  iree_host_size_t parameter_count = 0;
  for (iree_host_size_t i = 0; i < export_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_executable_export_info(
        executable, (iree_hal_executable_export_ordinal_t)i, &export_infos[i]);
    if (iree_status_is_ok(status)) {
      if (IREE_UNLIKELY(!iree_host_size_checked_add(
              parameter_count, export_infos[i].parameter_count,
              &parameter_count))) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "executable parameter metadata count overflow");
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, export_infos);
    return status;
  }

  iree_hal_executable_export_parameter_t* parameters = NULL;
  iree_host_size_t parameter_info_size = 0;
  if (parameter_count > 0) {
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            parameter_count, sizeof(iree_hal_executable_export_parameter_t),
            &parameter_info_size))) {
      iree_allocator_free(host_allocator, export_infos);
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "executable parameter metadata size overflow");
    }
    status = iree_allocator_malloc(host_allocator, parameter_info_size,
                                   (void**)&parameters);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(host_allocator, export_infos);
      return status;
    }
  }

  iree_host_size_t parameter_index = 0;
  for (iree_host_size_t i = 0; i < export_count && iree_status_is_ok(status);
       ++i) {
    const iree_host_size_t export_parameter_count =
        export_infos[i].parameter_count;
    if (export_parameter_count == 0) continue;
    status = iree_hal_executable_export_parameters(
        executable, (iree_hal_executable_export_ordinal_t)i,
        export_parameter_count, parameters + parameter_index);
    parameter_index += export_parameter_count;
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, parameters);
    iree_allocator_free(host_allocator, export_infos);
    return status;
  }

  iree_host_size_t metadata_size = 0;
  iree_host_size_t export_metadata_size = 0;
  iree_host_size_t parameter_metadata_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
                        export_count,
                        sizeof(iree_hal_replay_executable_export_metadata_t),
                        &export_metadata_size) ||
                    !iree_host_size_checked_mul(
                        parameter_count,
                        sizeof(iree_hal_replay_executable_parameter_metadata_t),
                        &parameter_metadata_size) ||
                    !iree_host_size_checked_add(
                        sizeof(iree_hal_replay_executable_metadata_header_t),
                        export_metadata_size, &metadata_size) ||
                    !iree_host_size_checked_add(metadata_size,
                                                parameter_metadata_size,
                                                &metadata_size))) {
    iree_allocator_free(host_allocator, parameters);
    iree_allocator_free(host_allocator, export_infos);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable replay metadata size overflow");
  }

  uint8_t* metadata_storage = NULL;
  status = iree_allocator_malloc(host_allocator, metadata_size,
                                 (void**)&metadata_storage);
  if (iree_status_is_ok(status)) {
    memset(metadata_storage, 0, metadata_size);
    iree_hal_replay_executable_metadata_header_t* header =
        (iree_hal_replay_executable_metadata_header_t*)metadata_storage;
    header->export_count = export_count;
    header->parameter_count = parameter_count;
    iree_hal_replay_executable_export_metadata_t* export_metadata =
        (iree_hal_replay_executable_export_metadata_t*)(metadata_storage +
                                                        sizeof(*header));
    iree_hal_replay_executable_parameter_metadata_t* parameter_metadata =
        (iree_hal_replay_executable_parameter_metadata_t*)(metadata_storage +
                                                           sizeof(*header) +
                                                           export_metadata_size);

    parameter_index = 0;
    for (iree_host_size_t i = 0; i < export_count; ++i) {
      export_metadata[i].flags = export_infos[i].flags;
      export_metadata[i].workgroup_size[0] = export_infos[i].workgroup_size[0];
      export_metadata[i].workgroup_size[1] = export_infos[i].workgroup_size[1];
      export_metadata[i].workgroup_size[2] = export_infos[i].workgroup_size[2];
      export_metadata[i].constant_count = export_infos[i].constant_count;
      export_metadata[i].binding_count = export_infos[i].binding_count;
      export_metadata[i].parameter_count = export_infos[i].parameter_count;
      for (iree_host_size_t j = 0; j < export_infos[i].parameter_count; ++j) {
        const iree_hal_executable_export_parameter_t* parameter =
            &parameters[parameter_index++];
        parameter_metadata->offset = parameter->offset;
        parameter_metadata->flags = parameter->flags;
        parameter_metadata->type = parameter->type;
        parameter_metadata->size = parameter->size;
        ++parameter_metadata;
      }
    }
    *out_storage = iree_make_byte_span(metadata_storage, metadata_size);
    *out_metadata = iree_make_const_byte_span(metadata_storage, metadata_size);
  }
  iree_allocator_free(host_allocator, parameters);
  iree_allocator_free(host_allocator, export_infos);
  return status;
}

static iree_status_t
iree_hal_replay_recorder_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_replay_recorder_executable_cache_t* executable_cache =
      iree_hal_replay_recorder_executable_cache_cast(base_executable_cache);
  *out_executable = NULL;

  iree_hal_replay_object_id_t executable_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_reserve_object_id(
      executable_cache->recorder, &executable_id));

  iree_hal_replay_executable_prepare_payload_t operation_payload;
  iree_const_byte_span_t operation_iovecs[5];
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_prepare_payload_iovecs(
      executable_params, iree_const_byte_span_empty(), &operation_payload,
      operation_iovecs));

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_recorder_executable_cache_begin_operation(
          executable_cache, executable_id,
          IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_CACHE_PREPARE_EXECUTABLE,
          IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_PREPARE, &pending_record));

  iree_hal_executable_t* base_executable = NULL;
  iree_hal_executable_t* replay_executable = NULL;
  iree_status_t status = iree_hal_executable_cache_prepare_executable(
      executable_cache->base_executable_cache, executable_params,
      &base_executable);
  iree_byte_span_t executable_metadata_storage = iree_byte_span_empty();
  iree_const_byte_span_t executable_metadata = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_capture_executable_metadata(
        base_executable, executable_cache->host_allocator,
        &executable_metadata_storage, &executable_metadata);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_executable_create_proxy(
        executable_cache->recorder, executable_cache->device_id, executable_id,
        base_executable, executable_cache->host_allocator, &replay_executable);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_prepare_payload_iovecs(
        executable_params, executable_metadata, &operation_payload,
        operation_iovecs);
  }
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, IREE_ARRAYSIZE(operation_iovecs),
      operation_iovecs, IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE, executable_id,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, 0, NULL);

  if (iree_status_is_ok(status)) {
    *out_executable = replay_executable;
  } else {
    iree_hal_executable_release(replay_executable);
  }
  iree_allocator_free(executable_cache->host_allocator,
                      executable_metadata_storage.data);
  iree_hal_executable_release(base_executable);
  return status;
}

static const iree_hal_executable_vtable_t
    iree_hal_replay_recorder_executable_vtable = {
        .destroy = iree_hal_replay_recorder_executable_destroy,
        .export_count = iree_hal_replay_recorder_executable_export_count,
        .export_info = iree_hal_replay_recorder_executable_export_info,
        .export_parameters =
            iree_hal_replay_recorder_executable_export_parameters,
        .lookup_export_by_name =
            iree_hal_replay_recorder_executable_lookup_export_by_name,
        .lookup_global_by_name =
            iree_hal_replay_recorder_executable_lookup_global_by_name,
};

static const iree_hal_executable_cache_vtable_t
    iree_hal_replay_recorder_executable_cache_vtable = {
        .destroy = iree_hal_replay_recorder_executable_cache_destroy,
        .infer_format = iree_hal_replay_recorder_executable_cache_infer_format,
        .can_prepare_format =
            iree_hal_replay_recorder_executable_cache_can_prepare_format,
        .prepare_executable =
            iree_hal_replay_recorder_executable_cache_prepare_executable,
};
