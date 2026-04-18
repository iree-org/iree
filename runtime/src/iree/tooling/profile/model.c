// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/model.h"

#include <string.h>

#include "iree/tooling/profile/reader.h"

void iree_profile_model_initialize(iree_allocator_t host_allocator,
                                   iree_profile_model_t* out_model) {
  memset(out_model, 0, sizeof(*out_model));
  out_model->host_allocator = host_allocator;
}

void iree_profile_model_deinitialize(iree_profile_model_t* model) {
  iree_allocator_free(model->host_allocator, model->host_execution_events);
  iree_allocator_free(model->host_allocator, model->queue_device_events);
  iree_allocator_free(model->host_allocator, model->queue_events);
  iree_allocator_free(model->host_allocator, model->devices);
  iree_allocator_free(model->host_allocator, model->queues);
  iree_allocator_free(model->host_allocator, model->command_operations);
  iree_allocator_free(model->host_allocator, model->command_buffers);
  iree_allocator_free(model->host_allocator, model->exports);
  iree_allocator_free(model->host_allocator, model->executables);
  memset(model, 0, sizeof(*model));
}

iree_status_t iree_profile_model_ensure_device(
    iree_profile_model_t* model, uint32_t physical_device_ordinal,
    iree_profile_model_device_t** out_device) {
  *out_device = NULL;

  for (iree_host_size_t i = 0; i < model->device_count; ++i) {
    if (model->devices[i].physical_device_ordinal == physical_device_ordinal) {
      *out_device = &model->devices[i];
      return iree_ok_status();
    }
  }

  if (model->device_count + 1 > model->device_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)4, model->device_count + 1),
        sizeof(model->devices[0]), &model->device_capacity,
        (void**)&model->devices));
  }

  iree_profile_model_device_t* device = &model->devices[model->device_count++];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  *out_device = device;
  return iree_ok_status();
}

static void iree_profile_model_record_clock_sample(
    iree_profile_model_device_t* device,
    const iree_hal_profile_clock_correlation_record_t* record) {
  if (device->clock_sample_count == 0) {
    device->first_clock_sample = *record;
  }
  device->last_clock_sample = *record;
  ++device->clock_sample_count;
}

const iree_profile_model_device_t* iree_profile_model_find_device(
    const iree_profile_model_t* model, uint32_t physical_device_ordinal) {
  for (iree_host_size_t i = 0; i < model->device_count; ++i) {
    if (model->devices[i].physical_device_ordinal == physical_device_ordinal) {
      return &model->devices[i];
    }
  }
  return NULL;
}

const iree_profile_model_queue_t* iree_profile_model_find_queue(
    const iree_profile_model_t* model, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id) {
  for (iree_host_size_t i = 0; i < model->queue_count; ++i) {
    const iree_profile_model_queue_t* queue_info = &model->queues[i];
    if (queue_info->record.physical_device_ordinal == physical_device_ordinal &&
        queue_info->record.queue_ordinal == queue_ordinal &&
        queue_info->record.stream_id == stream_id) {
      return queue_info;
    }
  }
  return NULL;
}

const iree_profile_model_executable_t* iree_profile_model_find_executable(
    const iree_profile_model_t* model, uint64_t executable_id) {
  for (iree_host_size_t i = 0; i < model->executable_count; ++i) {
    const iree_profile_model_executable_t* executable_info =
        &model->executables[i];
    if (executable_info->record.executable_id == executable_id) {
      return executable_info;
    }
  }
  return NULL;
}

const iree_profile_model_export_t* iree_profile_model_find_export(
    const iree_profile_model_t* model, uint64_t executable_id,
    uint32_t export_ordinal) {
  for (iree_host_size_t i = 0; i < model->export_count; ++i) {
    const iree_profile_model_export_t* export_info = &model->exports[i];
    if (export_info->executable_id == executable_id &&
        export_info->export_ordinal == export_ordinal) {
      return export_info;
    }
  }
  return NULL;
}

const iree_profile_model_command_buffer_t*
iree_profile_model_find_command_buffer(const iree_profile_model_t* model,
                                       uint64_t command_buffer_id) {
  for (iree_host_size_t i = 0; i < model->command_buffer_count; ++i) {
    const iree_profile_model_command_buffer_t* command_buffer_info =
        &model->command_buffers[i];
    if (command_buffer_info->record.command_buffer_id == command_buffer_id) {
      return command_buffer_info;
    }
  }
  return NULL;
}

bool iree_profile_model_device_try_fit_clock(
    const iree_profile_model_device_t* device, double* out_ns_per_tick,
    double* out_tick_frequency_hz) {
  *out_ns_per_tick = 0.0;
  *out_tick_frequency_hz = 0.0;
  if (!device || device->clock_sample_count < 2) return false;

  const iree_hal_profile_clock_correlation_record_t* first =
      &device->first_clock_sample;
  const iree_hal_profile_clock_correlation_record_t* last =
      &device->last_clock_sample;
  if (!iree_all_bits_set(
          first->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP) ||
      !iree_all_bits_set(
          last->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP)) {
    return false;
  }
  if (last->device_tick <= first->device_tick ||
      last->host_cpu_timestamp_ns <= first->host_cpu_timestamp_ns) {
    return false;
  }

  const double device_delta_ticks =
      (double)(last->device_tick - first->device_tick);
  const double host_delta_ns =
      (double)(last->host_cpu_timestamp_ns - first->host_cpu_timestamp_ns);
  *out_ns_per_tick = host_delta_ns / device_delta_ticks;
  *out_tick_frequency_hz = 1000000000.0 / *out_ns_per_tick;
  return true;
}

static iree_string_view_t iree_profile_model_format_numeric_key(
    uint32_t physical_device_ordinal, uint64_t executable_id,
    uint32_t export_ordinal, char* buffer, iree_host_size_t buffer_capacity) {
  if (buffer_capacity == 0) return iree_string_view_empty();
  int result = 0;
  if (physical_device_ordinal == UINT32_MAX) {
    result = snprintf(buffer, buffer_capacity, "executable%" PRIu64 "#%u",
                      executable_id, export_ordinal);
  } else {
    result =
        snprintf(buffer, buffer_capacity, "device%u/executable%" PRIu64 "#%u",
                 physical_device_ordinal, executable_id, export_ordinal);
  }
  if (result < 0) return iree_string_view_empty();
  iree_host_size_t length = (iree_host_size_t)result;
  if (length >= buffer_capacity) length = buffer_capacity - 1;
  return iree_make_string_view(buffer, length);
}

iree_string_view_t iree_profile_model_format_export_key(
    const iree_profile_model_export_t* export_info,
    uint32_t physical_device_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity) {
  if (!iree_string_view_is_empty(export_info->name)) {
    return export_info->name;
  }
  return iree_profile_model_format_numeric_key(
      physical_device_ordinal, export_info->executable_id,
      export_info->export_ordinal, numeric_buffer, numeric_buffer_capacity);
}

iree_status_t iree_profile_model_resolve_dispatch_key(
    const iree_profile_model_t* model, uint32_t physical_device_ordinal,
    uint64_t executable_id, uint32_t export_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity, iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();
  const iree_profile_model_executable_t* executable_info =
      iree_profile_model_find_executable(model, executable_id);
  if (!executable_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing executable metadata "
        "device=%u executable=%" PRIu64 " export=%u",
        physical_device_ordinal, executable_id, export_ordinal);
  }
  const iree_profile_model_export_t* export_info =
      iree_profile_model_find_export(model, executable_id, export_ordinal);
  if (!export_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing executable export metadata "
        "device=%u executable=%" PRIu64 " export=%u",
        physical_device_ordinal, executable_id, export_ordinal);
  }
  *out_key = iree_profile_model_format_export_key(
      export_info, physical_device_ordinal, numeric_buffer,
      numeric_buffer_capacity);
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_export(
    iree_profile_model_t* model,
    const iree_profile_model_export_t* export_info) {
  if (model->export_count + 1 > model->export_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)16, model->export_count + 1),
        sizeof(model->exports[0]), &model->export_capacity,
        (void**)&model->exports));
  }
  model->exports[model->export_count++] = *export_info;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_executable(
    iree_profile_model_t* model,
    const iree_hal_profile_executable_record_t* record) {
  if (model->executable_count + 1 > model->executable_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)8, model->executable_count + 1),
        sizeof(model->executables[0]), &model->executable_capacity,
        (void**)&model->executables));
  }
  iree_profile_model_executable_t* executable_info =
      &model->executables[model->executable_count++];
  executable_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_command_buffer(
    iree_profile_model_t* model,
    const iree_hal_profile_command_buffer_record_t* record) {
  if (model->command_buffer_count + 1 > model->command_buffer_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)16, model->command_buffer_count + 1),
        sizeof(model->command_buffers[0]), &model->command_buffer_capacity,
        (void**)&model->command_buffers));
  }
  iree_profile_model_command_buffer_t* command_buffer_info =
      &model->command_buffers[model->command_buffer_count++];
  command_buffer_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_command_operation(
    iree_profile_model_t* model,
    const iree_hal_profile_command_operation_record_t* record) {
  if (model->command_operation_count + 1 > model->command_operation_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)64, model->command_operation_count + 1),
        sizeof(model->command_operations[0]),
        &model->command_operation_capacity,
        (void**)&model->command_operations));
  }
  iree_profile_model_command_operation_t* operation_info =
      &model->command_operations[model->command_operation_count++];
  operation_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_queue(
    iree_profile_model_t* model,
    const iree_hal_profile_queue_record_t* record) {
  if (model->queue_count + 1 > model->queue_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)4, model->queue_count + 1),
        sizeof(model->queues[0]), &model->queue_capacity,
        (void**)&model->queues));
  }
  iree_profile_model_queue_t* queue_info = &model->queues[model->queue_count++];
  queue_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_queue_event(
    iree_profile_model_t* model, const iree_hal_profile_queue_event_t* record) {
  if (model->queue_event_count + 1 > model->queue_event_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)64, model->queue_event_count + 1),
        sizeof(model->queue_events[0]), &model->queue_event_capacity,
        (void**)&model->queue_events));
  }
  iree_profile_model_queue_event_t* event_info =
      &model->queue_events[model->queue_event_count++];
  event_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_queue_device_event(
    iree_profile_model_t* model,
    const iree_hal_profile_queue_device_event_t* record) {
  if (model->queue_device_event_count + 1 >
      model->queue_device_event_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)64, model->queue_device_event_count + 1),
        sizeof(model->queue_device_events[0]),
        &model->queue_device_event_capacity,
        (void**)&model->queue_device_events));
  }
  iree_profile_model_queue_device_event_t* event_info =
      &model->queue_device_events[model->queue_device_event_count++];
  event_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_append_host_execution_event(
    iree_profile_model_t* model,
    const iree_hal_profile_host_execution_event_t* record) {
  if (model->host_execution_event_count + 1 >
      model->host_execution_event_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        model->host_allocator,
        iree_max((iree_host_size_t)64, model->host_execution_event_count + 1),
        sizeof(model->host_execution_events[0]),
        &model->host_execution_event_capacity,
        (void**)&model->host_execution_events));
  }
  iree_profile_model_host_execution_event_t* event_info =
      &model->host_execution_events[model->host_execution_event_count++];
  event_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_model_process_queue_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_queue_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_queue_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    status = iree_profile_model_append_queue(model, &record_value);
  }
  return status;
}

static iree_status_t iree_profile_model_process_executable_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_executable_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_executable_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    status = iree_profile_model_append_executable(model, &record_value);
  }
  return status;
}

static iree_status_t iree_profile_model_process_export_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
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

    iree_hal_profile_executable_export_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    if ((iree_host_size_t)record_value.name_length !=
        typed_record.inline_payload.data_length) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "executable export name length is inconsistent");
    }
    if (iree_status_is_ok(status)) {
      iree_profile_model_export_t export_info = {
          .executable_id = record_value.executable_id,
          .flags = record_value.flags,
          .export_ordinal = record_value.export_ordinal,
          .constant_count = record_value.constant_count,
          .binding_count = record_value.binding_count,
          .parameter_count = record_value.parameter_count,
          .workgroup_size = {record_value.workgroup_size[0],
                             record_value.workgroup_size[1],
                             record_value.workgroup_size[2]},
          .pipeline_hash = {record_value.pipeline_hash[0],
                            record_value.pipeline_hash[1]},
          .name = iree_make_string_view(
              (const char*)typed_record.inline_payload.data,
              typed_record.inline_payload.data_length),
      };
      status = iree_profile_model_append_export(model, &export_info);
    }
  }
  return status;
}

static iree_status_t iree_profile_model_process_command_buffer_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_command_buffer_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_command_buffer_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    status = iree_profile_model_append_command_buffer(model, &record_value);
  }
  return status;
}

static iree_status_t iree_profile_model_process_command_operation_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_command_operation_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_command_operation_record_t record_value;
    memcpy(&record_value, typed_record.contents.data, sizeof(record_value));
    if (!iree_profile_model_find_command_buffer(
            model, record_value.command_buffer_id)) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "command operation references missing command-buffer metadata "
          "command_buffer=%" PRIu64 " command_index=%u",
          record_value.command_buffer_id, record_value.command_index);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_profile_model_append_command_operation(model, &record_value);
    }
  }
  return status;
}

static iree_status_t iree_profile_model_process_clock_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_clock_correlation_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_clock_correlation_record_t clock_record;
    memcpy(&clock_record, typed_record.contents.data, sizeof(clock_record));

    iree_profile_model_device_t* device = NULL;
    status = iree_profile_model_ensure_device(
        model, clock_record.physical_device_ordinal, &device);
    if (iree_status_is_ok(status)) {
      iree_profile_model_record_clock_sample(device, &clock_record);
    }
  }
  return status;
}

iree_status_t iree_profile_model_process_metadata_record(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(record->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
    return iree_profile_model_process_queue_records(model, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
    return iree_profile_model_process_executable_records(model, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    return iree_profile_model_process_export_records(model, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS)) {
    return iree_profile_model_process_command_buffer_records(model, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS)) {
    return iree_profile_model_process_command_operation_records(model, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    return iree_profile_model_process_clock_records(model, record);
  }
  return iree_ok_status();
}

const char* iree_profile_command_operation_type_name(
    iree_hal_profile_command_operation_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BARRIER:
      return "barrier";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH:
      return "dispatch";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL:
      return "fill";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COPY:
      return "copy";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_UPDATE:
      return "update";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_PROFILE_MARKER:
      return "profile_marker";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BRANCH:
      return "branch";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COND_BRANCH:
      return "cond_branch";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_RETURN:
      return "return";
    default:
      return "unknown";
  }
}

const char* iree_profile_queue_event_type_name(
    iree_hal_profile_queue_event_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER:
      return "barrier";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH:
      return "dispatch";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE:
      return "execute";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY:
      return "copy";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL:
      return "fill";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE:
      return "update";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ:
      return "read";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE:
      return "write";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA:
      return "alloca";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA:
      return "dealloca";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL:
      return "host_call";
    default:
      return "unknown";
  }
}

const char* iree_profile_queue_dependency_strategy_name(
    iree_hal_profile_queue_dependency_strategy_t strategy) {
  switch (strategy) {
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE:
      return "none";
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE:
      return "inline";
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER:
      return "device_barrier";
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER:
      return "software_defer";
    default:
      return "unknown";
  }
}

const char* iree_profile_event_relationship_type_name(
    iree_hal_profile_event_relationship_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_DISPATCH:
      return "queue_submission_dispatch";
    case IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_QUEUE_DEVICE_EVENT:
      return "queue_submission_queue_device_event";
    case IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_EVENT_HOST_EXECUTION_EVENT:
      return "queue_event_host_execution_event";
    case IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_HOST_EXECUTION_EVENT:
      return "queue_submission_host_execution_event";
    default:
      return "unknown";
  }
}

const char* iree_profile_event_endpoint_type_name(
    iree_hal_profile_event_endpoint_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_SUBMISSION:
      return "queue_submission";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_EVENT:
      return "queue_event";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_DISPATCH_EVENT:
      return "dispatch_event";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_COMMAND_OPERATION:
      return "command_operation";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_MEMORY_EVENT:
      return "memory_event";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_ARTIFACT:
      return "artifact";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_DEVICE_EVENT:
      return "queue_device_event";
    case IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_HOST_EXECUTION_EVENT:
      return "host_execution_event";
    default:
      return "unknown";
  }
}

static bool iree_profile_model_queue_operation_matches(
    iree_hal_profile_queue_event_type_t type, uint64_t submission_id,
    int64_t id_filter, iree_string_view_t filter) {
  if (id_filter >= 0 && submission_id != (uint64_t)id_filter) {
    return false;
  }
  iree_string_view_t type_name =
      iree_make_cstring_view(iree_profile_queue_event_type_name(type));
  return iree_profile_key_matches(type_name, filter);
}

static iree_status_t iree_profile_model_verify_event_queue(
    const iree_profile_model_t* model, const char* event_kind,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    uint64_t stream_id, uint64_t submission_id) {
  const iree_profile_model_queue_t* queue_info = iree_profile_model_find_queue(
      model, physical_device_ordinal, queue_ordinal, stream_id);
  if (!queue_info) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "%s references missing queue metadata "
                            "device=%u queue=%u stream=%" PRIu64
                            " submission=%" PRIu64,
                            event_kind, physical_device_ordinal, queue_ordinal,
                            stream_id, submission_id);
  }
  return iree_ok_status();
}

iree_status_t iree_profile_model_process_queue_event_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record,
    iree_string_view_t filter, int64_t id_filter) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(record->content_type,
                              IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
    return iree_ok_status();
  }

  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_queue_event_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_queue_event_t event;
    memcpy(&event, typed_record.contents.data, sizeof(event));
    ++model->total_queue_event_count;
    if (iree_profile_model_queue_operation_matches(
            event.type, event.submission_id, id_filter, filter)) {
      ++model->matched_queue_event_count;
      status = iree_profile_model_verify_event_queue(
          model, "queue event", event.physical_device_ordinal,
          event.queue_ordinal, event.stream_id, event.submission_id);
      if (iree_status_is_ok(status)) {
        status = iree_profile_model_append_queue_event(model, &event);
      }
    }
  }
  return status;
}

iree_status_t iree_profile_model_process_queue_device_event_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record,
    iree_string_view_t filter, int64_t id_filter) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(
          record->content_type,
          IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS)) {
    return iree_ok_status();
  }

  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_queue_device_event_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_queue_device_event_t event;
    memcpy(&event, typed_record.contents.data, sizeof(event));
    ++model->total_queue_device_event_count;
    if (iree_profile_model_queue_operation_matches(
            event.type, event.submission_id, id_filter, filter)) {
      ++model->matched_queue_device_event_count;
      iree_profile_model_device_t* device = NULL;
      status = iree_profile_model_ensure_device(
          model, event.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        status = iree_profile_model_verify_event_queue(
            model, "queue device event", event.physical_device_ordinal,
            event.queue_ordinal, event.stream_id, event.submission_id);
      }
      if (iree_status_is_ok(status)) {
        status = iree_profile_model_append_queue_device_event(model, &event);
      }
    }
  }
  return status;
}

iree_status_t iree_profile_model_process_host_execution_event_records(
    iree_profile_model_t* model, const iree_hal_profile_file_record_t* record,
    iree_string_view_t filter, int64_t id_filter) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(
          record->content_type,
          IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS)) {
    return iree_ok_status();
  }

  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_host_execution_event_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_host_execution_event_t event;
    memcpy(&event, typed_record.contents.data, sizeof(event));
    ++model->total_host_execution_event_count;
    if (iree_profile_model_queue_operation_matches(
            event.type, event.submission_id, id_filter, filter)) {
      ++model->matched_host_execution_event_count;
      status = iree_profile_model_verify_event_queue(
          model, "host execution event", event.physical_device_ordinal,
          event.queue_ordinal, event.stream_id, event.submission_id);
      if (iree_status_is_ok(status)) {
        status = iree_profile_model_append_host_execution_event(model, &event);
      }
    }
  }
  return status;
}

double iree_profile_model_span_ticks(uint64_t earliest_start_tick,
                                     uint64_t latest_end_tick) {
  if (earliest_start_tick == UINT64_MAX || latest_end_tick == 0 ||
      latest_end_tick < earliest_start_tick) {
    return 0.0;
  }
  return (double)(latest_end_tick - earliest_start_tick);
}

iree_status_t iree_profile_model_resolve_command_operation_key(
    const iree_profile_model_t* model,
    const iree_hal_profile_command_operation_record_t* operation,
    char* numeric_buffer, iree_host_size_t numeric_buffer_capacity,
    iree_string_view_t* out_key) {
  *out_key = iree_make_cstring_view(
      iree_profile_command_operation_type_name(operation->type));
  if (operation->type != IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH ||
      operation->executable_id == 0 ||
      operation->export_ordinal == UINT32_MAX) {
    return iree_ok_status();
  }

  const iree_profile_model_command_buffer_t* command_buffer =
      iree_profile_model_find_command_buffer(model,
                                             operation->command_buffer_id);
  if (!command_buffer) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "command operation references missing command-buffer metadata "
        "command_buffer=%" PRIu64 " command_index=%u",
        operation->command_buffer_id, operation->command_index);
  }
  return iree_profile_model_resolve_dispatch_key(
      model, command_buffer->record.physical_device_ordinal,
      operation->executable_id, operation->export_ordinal, numeric_buffer,
      numeric_buffer_capacity, out_key);
}
