// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/profile.h"

#include <inttypes.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_local_profile_recorder_t
//===----------------------------------------------------------------------===//

#define IREE_HAL_LOCAL_PROFILE_DEFAULT_EVENT_CAPACITY 4096

struct iree_hal_local_profile_recorder_t {
  // Host allocator used for recorder-owned storage.
  iree_allocator_t host_allocator;

  // Copied human-readable producer name used on emitted chunks.
  iree_string_view_t name;

  // Allocated storage backing |name| when the caller provided one.
  char* name_storage;

  // Owned profiling options for the active session.
  iree_hal_device_profiling_options_t options;

  // Opaque storage backing borrowed pointers in |options|, or NULL.
  iree_hal_device_profiling_options_storage_t* options_storage;

  // Process-local profiling session identifier.
  uint64_t session_id;

  // True while the sink session has begun and has not been ended.
  bool active;

  // Buffered host queue event records.
  iree_hal_profile_queue_event_t* queue_events;

  // Maximum entries in |queue_events|.
  iree_host_size_t queue_event_capacity;

  // Number of valid buffered entries in |queue_events|.
  iree_host_size_t queue_event_count;

  // Next nonzero queue event id assigned by this stream.
  uint64_t next_queue_event_id;

  // Buffered host execution span records.
  iree_hal_profile_host_execution_event_t* host_execution_events;

  // Maximum entries in |host_execution_events|.
  iree_host_size_t host_execution_event_capacity;

  // Number of valid buffered entries in |host_execution_events|.
  iree_host_size_t host_execution_event_count;

  // Next nonzero host execution event id assigned by this stream.
  uint64_t next_host_execution_event_id;

  // Buffered explicit profile relationship records.
  iree_hal_profile_event_relationship_record_t* event_relationships;

  // Maximum entries in |event_relationships|.
  iree_host_size_t event_relationship_capacity;

  // Number of valid buffered entries in |event_relationships|.
  iree_host_size_t event_relationship_count;

  // Next nonzero relationship id assigned by this stream.
  uint64_t next_event_relationship_id;
};

static iree_status_t iree_hal_local_profile_recorder_validate_records(
    const iree_hal_local_profile_recorder_options_t* recorder_options) {
  if (recorder_options->device_record_count == 0 ||
      !recorder_options->device_records) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local profiling requires at least one device metadata record");
  }
  if (recorder_options->queue_record_count == 0 ||
      !recorder_options->queue_records) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local profiling requires at least one queue metadata record");
  }
  for (iree_host_size_t i = 0; i < recorder_options->device_record_count; ++i) {
    const iree_hal_profile_device_record_t* record =
        &recorder_options->device_records[i];
    if (record->record_length != sizeof(*record)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "local profiling device metadata record %" PRIhsz
                              " has unsupported length %" PRIu32,
                              i, record->record_length);
    }
    if (record->physical_device_ordinal == UINT32_MAX ||
        record->queue_count == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "local profiling device metadata record %" PRIhsz
                              " has invalid queue identity",
                              i);
    }
  }
  for (iree_host_size_t i = 0; i < recorder_options->queue_record_count; ++i) {
    const iree_hal_profile_queue_record_t* record =
        &recorder_options->queue_records[i];
    if (record->record_length != sizeof(*record)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "local profiling queue metadata record %" PRIhsz
                              " has unsupported length %" PRIu32,
                              i, record->record_length);
    }
    if (record->physical_device_ordinal == UINT32_MAX ||
        record->queue_ordinal == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "local profiling queue metadata record %" PRIhsz
                              " has invalid queue identity",
                              i);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_local_profile_recorder_validate_profiling_options(
    const iree_hal_device_profiling_options_t* profiling_options) {
  const iree_hal_device_profiling_data_families_t supported_data_families =
      iree_hal_local_profile_recorder_supported_data_families();
  const iree_hal_device_profiling_data_families_t unsupported_data_families =
      profiling_options->data_families & ~supported_data_families;
  if (unsupported_data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported local profiling data families 0x%" PRIx64,
        unsupported_data_families);
  }
  if (profiling_options->data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE &&
      !profiling_options->sink) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local profiling with requested data families requires a profile sink");
  }
  if (profiling_options->capture_filter.reserved0 != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local profiling capture filter reserved fields must be zero");
  }
  if (!iree_hal_profile_capture_filter_is_default(
          &profiling_options->capture_filter)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "local profiling does not support capture filters yet");
  }
  if (profiling_options->counter_set_count != 0 ||
      profiling_options->counter_sets) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "local profiling does not support counter set selections");
  }
  return iree_ok_status();
}

static iree_host_size_t iree_hal_local_profile_recorder_normalize_capacity(
    iree_host_size_t requested_capacity) {
  return requested_capacity != 0
             ? requested_capacity
             : IREE_HAL_LOCAL_PROFILE_DEFAULT_EVENT_CAPACITY;
}

static iree_status_t iree_hal_local_profile_recorder_allocate_events(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_recorder_options_t* recorder_options) {
  if (iree_hal_device_profiling_options_requests_data(
          &recorder->options, IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS)) {
    recorder->queue_event_capacity =
        iree_hal_local_profile_recorder_normalize_capacity(
            recorder_options->queue_event_capacity);
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        recorder->host_allocator, recorder->queue_event_capacity,
        sizeof(*recorder->queue_events), (void**)&recorder->queue_events));
  }
  if (iree_hal_device_profiling_options_requests_data(
          &recorder->options,
          IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS)) {
    recorder->host_execution_event_capacity =
        iree_hal_local_profile_recorder_normalize_capacity(
            recorder_options->host_execution_event_capacity);
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        recorder->host_allocator, recorder->host_execution_event_capacity,
        sizeof(*recorder->host_execution_events),
        (void**)&recorder->host_execution_events));
  }
  if (recorder->queue_events && recorder->host_execution_events) {
    if (recorder_options->event_relationship_capacity != 0) {
      recorder->event_relationship_capacity =
          recorder_options->event_relationship_capacity;
    } else {
      recorder->event_relationship_capacity =
          recorder->queue_event_capacity <
                  recorder->host_execution_event_capacity
              ? recorder->queue_event_capacity
              : recorder->host_execution_event_capacity;
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        recorder->host_allocator, recorder->event_relationship_capacity,
        sizeof(*recorder->event_relationships),
        (void**)&recorder->event_relationships));
  }
  recorder->next_queue_event_id = 1;
  recorder->next_host_execution_event_id = 1;
  recorder->next_event_relationship_id = 1;
  return iree_ok_status();
}

static iree_status_t iree_hal_local_profile_recorder_copy_name(
    iree_hal_local_profile_recorder_t* recorder, iree_string_view_t name) {
  if (iree_string_view_is_empty(name)) {
    recorder->name = IREE_SV("local");
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_clone(
      recorder->host_allocator, iree_make_const_byte_span(name.data, name.size),
      (void**)&recorder->name_storage));
  recorder->name = iree_make_string_view(recorder->name_storage, name.size);
  return iree_ok_status();
}

static iree_hal_profile_chunk_metadata_t
iree_hal_local_profile_recorder_metadata(
    const iree_hal_local_profile_recorder_t* recorder,
    iree_string_view_t content_type) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = content_type;
  metadata.name = recorder->name;
  metadata.session_id = recorder->session_id;
  return metadata;
}

static iree_status_t iree_hal_local_profile_recorder_write_records(
    iree_hal_local_profile_recorder_t* recorder,
    iree_string_view_t content_type, const void* records,
    iree_host_size_t record_count, iree_host_size_t record_size) {
  if (record_count == 0) return iree_ok_status();
  iree_host_size_t byte_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(record_count, record_size,
                                                &byte_length))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local profiling chunk size overflow for %" PRIhsz
                            " records",
                            record_count);
  }
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_local_profile_recorder_metadata(recorder, content_type);
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span(records, byte_length);
  return iree_hal_profile_sink_write(recorder->options.sink, &metadata, 1,
                                     &iovec);
}

static iree_status_t iree_hal_local_profile_recorder_write_metadata(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_recorder_options_t* recorder_options) {
  IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_write_records(
      recorder, IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES,
      recorder_options->device_records, recorder_options->device_record_count,
      sizeof(*recorder_options->device_records)));
  return iree_hal_local_profile_recorder_write_records(
      recorder, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES,
      recorder_options->queue_records, recorder_options->queue_record_count,
      sizeof(*recorder_options->queue_records));
}

iree_status_t iree_hal_local_profile_recorder_create(
    const iree_hal_local_profile_recorder_options_t* recorder_options,
    const iree_hal_device_profiling_options_t* profiling_options,
    iree_allocator_t host_allocator,
    iree_hal_local_profile_recorder_t** out_recorder) {
  IREE_ASSERT_ARGUMENT(recorder_options);
  IREE_ASSERT_ARGUMENT(profiling_options);
  IREE_ASSERT_ARGUMENT(out_recorder);
  *out_recorder = NULL;

  if (profiling_options->data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_local_profile_recorder_validate_profiling_options(
          profiling_options));
  IREE_RETURN_IF_ERROR(
      iree_hal_local_profile_recorder_validate_records(recorder_options));

  iree_hal_local_profile_recorder_t* recorder = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*recorder),
                                             (void**)&recorder));
  recorder->host_allocator = host_allocator;
  recorder->session_id = recorder_options->session_id;

  bool sink_session_begun = false;
  iree_status_t status = iree_hal_local_profile_recorder_copy_name(
      recorder, recorder_options->name);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_profiling_options_clone(
        profiling_options, host_allocator, &recorder->options,
        &recorder->options_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_local_profile_recorder_allocate_events(recorder,
                                                             recorder_options);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_local_profile_recorder_metadata(
            recorder, IREE_HAL_PROFILE_CONTENT_TYPE_SESSION);
    status =
        iree_hal_profile_sink_begin_session(recorder->options.sink, &metadata);
    sink_session_begun = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_local_profile_recorder_write_metadata(recorder,
                                                            recorder_options);
  }

  if (iree_status_is_ok(status)) {
    recorder->active = true;
    *out_recorder = recorder;
  } else {
    if (sink_session_begun) {
      iree_hal_profile_chunk_metadata_t metadata =
          iree_hal_local_profile_recorder_metadata(
              recorder, IREE_HAL_PROFILE_CONTENT_TYPE_SESSION);
      status = iree_status_join(status, iree_hal_profile_sink_end_session(
                                            recorder->options.sink, &metadata,
                                            iree_status_code(status)));
    }
    iree_hal_local_profile_recorder_destroy(recorder);
  }
  return status;
}

void iree_hal_local_profile_recorder_destroy(
    iree_hal_local_profile_recorder_t* recorder) {
  if (!recorder) return;
  IREE_ASSERT(!recorder->active,
              "active local profile recorders must be ended before destroy");
  iree_allocator_t host_allocator = recorder->host_allocator;
  iree_allocator_free(host_allocator, recorder->event_relationships);
  iree_allocator_free(host_allocator, recorder->host_execution_events);
  iree_allocator_free(host_allocator, recorder->queue_events);
  iree_hal_device_profiling_options_storage_free(recorder->options_storage,
                                                 host_allocator);
  iree_allocator_free(host_allocator, recorder->name_storage);
  iree_allocator_free(host_allocator, recorder);
}

bool iree_hal_local_profile_recorder_is_enabled(
    const iree_hal_local_profile_recorder_t* recorder,
    iree_hal_device_profiling_data_families_t data_families) {
  return recorder && recorder->active &&
         iree_hal_device_profiling_options_requests_data(&recorder->options,
                                                         data_families);
}

static iree_status_t iree_hal_local_profile_recorder_validate_queue_scope(
    const iree_hal_local_profile_queue_scope_t* scope) {
  if (scope->physical_device_ordinal == UINT32_MAX ||
      scope->queue_ordinal == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local profile event requires a queue scope");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_local_profile_recorder_append_queue_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_queue_event_info_t* event_info,
    uint64_t* out_event_id) {
  IREE_ASSERT_ARGUMENT(event_info);
  if (out_event_id) *out_event_id = 0;
  if (!iree_hal_local_profile_recorder_is_enabled(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_local_profile_recorder_validate_queue_scope(&event_info->scope));
  if (event_info->type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local profile queue event requires a type");
  }
  if (recorder->queue_event_count >= recorder->queue_event_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "local profile queue event buffer is full");
  }

  iree_hal_profile_queue_event_t* event =
      &recorder->queue_events[recorder->queue_event_count++];
  *event = iree_hal_profile_queue_event_default();
  event->type = event_info->type;
  event->flags = event_info->flags;
  event->dependency_strategy = event_info->dependency_strategy;
  event->event_id = recorder->next_queue_event_id++;
  event->host_time_ns = event_info->host_time_ns != 0 ? event_info->host_time_ns
                                                      : iree_time_now();
  event->submission_id = event_info->submission_id;
  event->command_buffer_id = event_info->command_buffer_id;
  event->allocation_id = event_info->allocation_id;
  event->stream_id = event_info->scope.stream_id;
  event->physical_device_ordinal = event_info->scope.physical_device_ordinal;
  event->queue_ordinal = event_info->scope.queue_ordinal;
  event->wait_count = event_info->wait_count;
  event->signal_count = event_info->signal_count;
  event->barrier_count = event_info->barrier_count;
  event->operation_count = event_info->operation_count;
  event->payload_length = event_info->payload_length;
  if (out_event_id) *out_event_id = event->event_id;
  return iree_ok_status();
}

static iree_status_t iree_hal_local_profile_recorder_append_relationship(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_queue_scope_t* scope, uint64_t queue_event_id,
    uint64_t host_execution_event_id) {
  if (!recorder->event_relationships || queue_event_id == 0 ||
      host_execution_event_id == 0) {
    return iree_ok_status();
  }
  if (recorder->event_relationship_count >=
      recorder->event_relationship_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "local profile relationship buffer is full");
  }

  iree_hal_profile_event_relationship_record_t* relationship =
      &recorder->event_relationships[recorder->event_relationship_count++];
  *relationship = iree_hal_profile_event_relationship_record_default();
  relationship->type =
      IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_EVENT_HOST_EXECUTION_EVENT;
  relationship->relationship_id = recorder->next_event_relationship_id++;
  relationship->source_type = IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_EVENT;
  relationship->target_type =
      IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_HOST_EXECUTION_EVENT;
  relationship->physical_device_ordinal = scope->physical_device_ordinal;
  relationship->queue_ordinal = scope->queue_ordinal;
  relationship->stream_id = scope->stream_id;
  relationship->source_id = queue_event_id;
  relationship->target_id = host_execution_event_id;
  return iree_ok_status();
}

iree_status_t iree_hal_local_profile_recorder_append_host_execution_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_host_execution_event_info_t* event_info,
    uint64_t* out_event_id) {
  IREE_ASSERT_ARGUMENT(event_info);
  if (out_event_id) *out_event_id = 0;
  if (!iree_hal_local_profile_recorder_is_enabled(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_local_profile_recorder_validate_queue_scope(&event_info->scope));
  if (event_info->type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local profile host execution event requires a "
                            "queue operation type");
  }

  iree_time_t start_time_ns = event_info->start_host_time_ns;
  iree_time_t end_time_ns = event_info->end_host_time_ns;
  if (start_time_ns == 0) start_time_ns = iree_time_now();
  if (end_time_ns == 0) end_time_ns = iree_time_now();
  if (end_time_ns < start_time_ns) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local profile host execution event end time precedes start time");
  }
  if (recorder->host_execution_event_count >=
      recorder->host_execution_event_capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "local profile host execution event buffer is full");
  }
  if (recorder->event_relationships &&
      event_info->related_queue_event_id != 0 &&
      recorder->event_relationship_count >=
          recorder->event_relationship_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "local profile relationship buffer is full");
  }

  iree_hal_profile_host_execution_event_t* event =
      &recorder->host_execution_events[recorder->host_execution_event_count++];
  *event = iree_hal_profile_host_execution_event_default();
  event->type = event_info->type;
  event->flags = event_info->flags;
  event->status_code = event_info->status_code;
  event->event_id = recorder->next_host_execution_event_id++;
  event->submission_id = event_info->submission_id;
  event->command_buffer_id = event_info->command_buffer_id;
  event->executable_id = event_info->executable_id;
  event->allocation_id = event_info->allocation_id;
  event->stream_id = event_info->scope.stream_id;
  event->physical_device_ordinal = event_info->scope.physical_device_ordinal;
  event->queue_ordinal = event_info->scope.queue_ordinal;
  event->command_index = event_info->command_index;
  event->export_ordinal = event_info->export_ordinal;
  memcpy(event->workgroup_count, event_info->workgroup_count,
         sizeof(event->workgroup_count));
  memcpy(event->workgroup_size, event_info->workgroup_size,
         sizeof(event->workgroup_size));
  event->start_host_time_ns = start_time_ns;
  event->end_host_time_ns = end_time_ns;
  event->payload_length = event_info->payload_length;
  event->operation_count = event_info->operation_count;
  if (out_event_id) *out_event_id = event->event_id;

  return iree_hal_local_profile_recorder_append_relationship(
      recorder, &event_info->scope, event_info->related_queue_event_id,
      event->event_id);
}

iree_status_t iree_hal_local_profile_recorder_flush(
    iree_hal_local_profile_recorder_t* recorder) {
  if (!recorder || !recorder->active) return iree_ok_status();

  iree_status_t status = iree_hal_local_profile_recorder_write_records(
      recorder, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS,
      recorder->queue_events, recorder->queue_event_count,
      sizeof(*recorder->queue_events));
  if (iree_status_is_ok(status)) {
    recorder->queue_event_count = 0;
    status = iree_hal_local_profile_recorder_write_records(
        recorder, IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS,
        recorder->host_execution_events, recorder->host_execution_event_count,
        sizeof(*recorder->host_execution_events));
  }
  if (iree_status_is_ok(status)) {
    recorder->host_execution_event_count = 0;
    status = iree_hal_local_profile_recorder_write_records(
        recorder, IREE_HAL_PROFILE_CONTENT_TYPE_EVENT_RELATIONSHIPS,
        recorder->event_relationships, recorder->event_relationship_count,
        sizeof(*recorder->event_relationships));
  }
  if (iree_status_is_ok(status)) {
    recorder->event_relationship_count = 0;
  }
  return status;
}

iree_status_t iree_hal_local_profile_recorder_end(
    iree_hal_local_profile_recorder_t* recorder) {
  if (!recorder || !recorder->active) return iree_ok_status();

  iree_status_t status = iree_hal_local_profile_recorder_flush(recorder);
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_local_profile_recorder_metadata(
          recorder, IREE_HAL_PROFILE_CONTENT_TYPE_SESSION);
  status = iree_status_join(
      status, iree_hal_profile_sink_end_session(
                  recorder->options.sink, &metadata, iree_status_code(status)));
  recorder->active = false;
  return status;
}
