// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/att/bundle.h"

#include <string.h>

#include "iree/tooling/profile/att/util.h"
#include "iree/tooling/profile/reader.h"

void iree_profile_att_profile_initialize(
    iree_allocator_t host_allocator, iree_profile_att_profile_t* out_profile) {
  memset(out_profile, 0, sizeof(*out_profile));
  out_profile->host_allocator = host_allocator;
}

void iree_profile_att_profile_deinitialize(
    iree_profile_att_profile_t* profile) {
  iree_allocator_t host_allocator = profile->host_allocator;
  iree_allocator_free(host_allocator, profile->traces);
  iree_allocator_free(host_allocator, profile->dispatches);
  iree_allocator_free(host_allocator, profile->exports);
  iree_allocator_free(host_allocator, profile->code_object_loads);
  iree_allocator_free(host_allocator, profile->code_objects);
  iree_io_file_contents_free(profile->file_contents);
  memset(profile, 0, sizeof(*profile));
}

static iree_status_t iree_profile_att_append_code_object(
    iree_profile_att_profile_t* profile,
    iree_profile_att_code_object_t code_object) {
  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      profile->host_allocator, profile->code_object_count + 1,
      sizeof(profile->code_objects[0]), &profile->code_object_capacity,
      (void**)&profile->code_objects));
  profile->code_objects[profile->code_object_count++] = code_object;
  return iree_ok_status();
}

static iree_status_t iree_profile_att_append_code_object_load(
    iree_profile_att_profile_t* profile,
    iree_profile_att_code_object_load_t code_object_load) {
  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      profile->host_allocator, profile->code_object_load_count + 1,
      sizeof(profile->code_object_loads[0]),
      &profile->code_object_load_capacity,
      (void**)&profile->code_object_loads));
  profile->code_object_loads[profile->code_object_load_count++] =
      code_object_load;
  return iree_ok_status();
}

static iree_status_t iree_profile_att_append_export(
    iree_profile_att_profile_t* profile,
    iree_profile_att_export_t export_info) {
  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      profile->host_allocator, profile->export_count + 1,
      sizeof(profile->exports[0]), &profile->export_capacity,
      (void**)&profile->exports));
  profile->exports[profile->export_count++] = export_info;
  return iree_ok_status();
}

static iree_status_t iree_profile_att_append_dispatch(
    iree_profile_att_profile_t* profile, iree_profile_att_dispatch_t dispatch) {
  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      profile->host_allocator, profile->dispatch_count + 1,
      sizeof(profile->dispatches[0]), &profile->dispatch_capacity,
      (void**)&profile->dispatches));
  profile->dispatches[profile->dispatch_count++] = dispatch;
  return iree_ok_status();
}

static iree_status_t iree_profile_att_append_trace(
    iree_profile_att_profile_t* profile, iree_profile_att_trace_t trace) {
  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      profile->host_allocator, profile->trace_count + 1,
      sizeof(profile->traces[0]), &profile->trace_capacity,
      (void**)&profile->traces));
  profile->traces[profile->trace_count++] = trace;
  return iree_ok_status();
}

static iree_status_t iree_profile_att_parse_code_objects(
    iree_profile_att_profile_t* profile,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_executable_code_object_record_t),
      &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_executable_code_object_record_t code_object_record;
    memcpy(&code_object_record, typed_record.contents.data,
           sizeof(code_object_record));
    if ((iree_host_size_t)code_object_record.data_length !=
        typed_record.inline_payload.data_length) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "profile executable code-object chunk has an invalid record");
    }
    if (iree_status_is_ok(status)) {
      iree_profile_att_code_object_t code_object = {
          .executable_id = code_object_record.executable_id,
          .code_object_id = code_object_record.code_object_id,
          .data = typed_record.inline_payload,
      };
      status = iree_profile_att_append_code_object(profile, code_object);
    }
  }
  return status;
}

static iree_status_t iree_profile_att_parse_code_object_loads(
    iree_profile_att_profile_t* profile,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_executable_code_object_load_record_t),
      &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_executable_code_object_load_record_t load_record;
    memcpy(&load_record, typed_record.contents.data, sizeof(load_record));
    if (typed_record.record_length != sizeof(load_record)) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "profile executable code-object load record has invalid length");
    }
    if (iree_status_is_ok(status)) {
      iree_profile_att_code_object_load_t code_object_load = {
          .physical_device_ordinal = load_record.physical_device_ordinal,
          .executable_id = load_record.executable_id,
          .code_object_id = load_record.code_object_id,
          .load_delta = load_record.load_delta,
          .load_size = load_record.load_size,
      };
      status =
          iree_profile_att_append_code_object_load(profile, code_object_load);
    }
  }
  return status;
}

static iree_status_t iree_profile_att_parse_exports(
    iree_profile_att_profile_t* profile,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_executable_export_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_executable_export_record_t export_record;
    memcpy(&export_record, typed_record.contents.data, sizeof(export_record));
    if ((iree_host_size_t)export_record.name_length !=
        typed_record.inline_payload.data_length) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "profile executable export chunk has an invalid record");
    }
    if (iree_status_is_ok(status)) {
      iree_profile_att_export_t export_info = {
          .executable_id = export_record.executable_id,
          .export_ordinal = export_record.export_ordinal,
          .name = iree_make_string_view(
              (const char*)typed_record.inline_payload.data,
              typed_record.inline_payload.data_length),
      };
      status = iree_profile_att_append_export(profile, export_info);
    }
  }
  return status;
}

static iree_status_t iree_profile_att_parse_dispatches(
    iree_profile_att_profile_t* profile,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_dispatch_event_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_profile_att_dispatch_t dispatch = {
        .physical_device_ordinal = record->header.physical_device_ordinal,
        .queue_ordinal = record->header.queue_ordinal,
    };
    memcpy(&dispatch.record, typed_record.contents.data,
           sizeof(dispatch.record));
    if (typed_record.record_length != sizeof(dispatch.record)) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "profile dispatch event record has invalid length");
    }
    if (iree_status_is_ok(status)) {
      status = iree_profile_att_append_dispatch(profile, dispatch);
    }
  }
  return status;
}

static iree_status_t iree_profile_att_parse_trace(
    iree_profile_att_profile_t* profile,
    const iree_hal_profile_file_record_t* record) {
  iree_hal_profile_executable_trace_record_t trace_record;
  iree_const_byte_span_t trace_data = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_profile_executable_trace_record_parse(
      record, &trace_record, &trace_data));
  iree_profile_att_trace_t trace = {
      .record = trace_record,
      .data = trace_data,
  };
  return iree_profile_att_append_trace(profile, trace);
}

iree_status_t iree_profile_att_profile_parse_record(
    iree_profile_att_profile_t* profile,
    const iree_hal_profile_file_record_t* record) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(
          record->content_type,
          IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECTS)) {
    return iree_profile_att_parse_code_objects(profile, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECT_LOADS)) {
    return iree_profile_att_parse_code_object_loads(profile, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    return iree_profile_att_parse_exports(profile, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    return iree_profile_att_parse_dispatches(profile, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACES)) {
    return iree_profile_att_parse_trace(profile, record);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_att_parse_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  return iree_profile_att_profile_parse_record(
      (iree_profile_att_profile_t*)user_data, record);
}

iree_status_t iree_profile_att_profile_parse_file(
    iree_string_view_t path, iree_profile_att_profile_t* profile) {
  iree_profile_file_t profile_file;
  IREE_RETURN_IF_ERROR(
      iree_profile_file_open(path, profile->host_allocator, &profile_file));
  // Trace payloads borrow from the mapped bundle, so the ATT profile retains
  // the mapping instead of closing the generic reader at the end of parsing.
  profile->file_contents = profile_file.contents;
  iree_profile_file_record_callback_t record_callback = {
      .fn = iree_profile_att_parse_record,
      .user_data = profile,
  };
  return iree_profile_file_for_each_record(&profile_file, record_callback);
}

const iree_profile_att_code_object_t* iree_profile_att_profile_find_code_object(
    const iree_profile_att_profile_t* profile, uint64_t executable_id,
    uint64_t code_object_id) {
  for (iree_host_size_t i = 0; i < profile->code_object_count; ++i) {
    const iree_profile_att_code_object_t* code_object =
        &profile->code_objects[i];
    if (code_object->executable_id == executable_id &&
        code_object->code_object_id == code_object_id) {
      return code_object;
    }
  }
  return NULL;
}

const iree_profile_att_export_t* iree_profile_att_profile_find_export(
    const iree_profile_att_profile_t* profile, uint64_t executable_id,
    uint32_t export_ordinal) {
  for (iree_host_size_t i = 0; i < profile->export_count; ++i) {
    const iree_profile_att_export_t* export_info = &profile->exports[i];
    if (export_info->executable_id == executable_id &&
        export_info->export_ordinal == export_ordinal) {
      return export_info;
    }
  }
  return NULL;
}

const iree_profile_att_dispatch_t* iree_profile_att_profile_find_dispatch(
    const iree_profile_att_profile_t* profile,
    const iree_profile_att_trace_t* trace) {
  for (iree_host_size_t i = 0; i < profile->dispatch_count; ++i) {
    const iree_profile_att_dispatch_t* dispatch = &profile->dispatches[i];
    if (dispatch->record.event_id == trace->record.dispatch_event_id &&
        dispatch->record.submission_id == trace->record.submission_id &&
        dispatch->record.command_buffer_id == trace->record.command_buffer_id &&
        dispatch->record.command_index == trace->record.command_index &&
        dispatch->record.executable_id == trace->record.executable_id &&
        dispatch->record.export_ordinal == trace->record.export_ordinal) {
      return dispatch;
    }
  }
  return NULL;
}
