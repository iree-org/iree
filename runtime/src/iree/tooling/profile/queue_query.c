// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/queue_query.h"

#include <string.h>

#include "iree/tooling/profile/reader.h"

void iree_profile_queue_event_query_initialize(
    iree_allocator_t host_allocator,
    iree_profile_queue_event_query_t* out_query) {
  memset(out_query, 0, sizeof(*out_query));
  out_query->host_allocator = host_allocator;
}

void iree_profile_queue_event_query_deinitialize(
    iree_profile_queue_event_query_t* query) {
  iree_allocator_free(query->host_allocator, query->host_execution_events);
  iree_allocator_free(query->host_allocator, query->queue_device_events);
  iree_allocator_free(query->host_allocator, query->queue_events);
  memset(query, 0, sizeof(*query));
}

static bool iree_profile_queue_operation_matches(
    iree_hal_profile_queue_event_type_t type, uint64_t submission_id,
    int64_t id_filter, iree_string_view_t filter) {
  if (id_filter >= 0 && submission_id != (uint64_t)id_filter) {
    return false;
  }
  iree_string_view_t type_name =
      iree_make_cstring_view(iree_profile_queue_event_type_name(type));
  return iree_profile_key_matches(type_name, filter);
}

static iree_status_t iree_profile_queue_event_query_verify_queue(
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

static iree_status_t iree_profile_queue_event_query_append_event(
    iree_profile_queue_event_query_t* query,
    const iree_hal_profile_queue_event_t* record) {
  if (query->queue_event_count + 1 > query->queue_event_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        query->host_allocator,
        iree_max((iree_host_size_t)64, query->queue_event_count + 1),
        sizeof(query->queue_events[0]), &query->queue_event_capacity,
        (void**)&query->queue_events));
  }
  iree_profile_queue_event_row_t* event_info =
      &query->queue_events[query->queue_event_count++];
  event_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_queue_event_query_append_device_event(
    iree_profile_queue_event_query_t* query,
    const iree_hal_profile_queue_device_event_t* record) {
  if (query->queue_device_event_count + 1 >
      query->queue_device_event_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        query->host_allocator,
        iree_max((iree_host_size_t)64, query->queue_device_event_count + 1),
        sizeof(query->queue_device_events[0]),
        &query->queue_device_event_capacity,
        (void**)&query->queue_device_events));
  }
  iree_profile_queue_device_event_row_t* event_info =
      &query->queue_device_events[query->queue_device_event_count++];
  event_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_queue_event_query_append_host_execution(
    iree_profile_queue_event_query_t* query,
    const iree_hal_profile_host_execution_event_t* record) {
  if (query->host_execution_event_count + 1 >
      query->host_execution_event_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        query->host_allocator,
        iree_max((iree_host_size_t)64, query->host_execution_event_count + 1),
        sizeof(query->host_execution_events[0]),
        &query->host_execution_event_capacity,
        (void**)&query->host_execution_events));
  }
  iree_profile_host_execution_event_row_t* event_info =
      &query->host_execution_events[query->host_execution_event_count++];
  event_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_queue_event_query_process_queue_events(
    iree_profile_queue_event_query_t* query, const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter) {
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
    ++query->total_queue_event_count;
    if (iree_profile_queue_operation_matches(event.type, event.submission_id,
                                             id_filter, filter)) {
      ++query->matched_queue_event_count;
      status = iree_profile_queue_event_query_verify_queue(
          model, "queue event", event.physical_device_ordinal,
          event.queue_ordinal, event.stream_id, event.submission_id);
      if (iree_status_is_ok(status)) {
        status = iree_profile_queue_event_query_append_event(query, &event);
      }
    }
  }
  return status;
}

static iree_status_t iree_profile_queue_event_query_process_device_events(
    iree_profile_queue_event_query_t* query, const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter) {
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
    ++query->total_queue_device_event_count;
    if (iree_profile_queue_operation_matches(event.type, event.submission_id,
                                             id_filter, filter)) {
      ++query->matched_queue_device_event_count;
      status = iree_profile_queue_event_query_verify_queue(
          model, "queue device event", event.physical_device_ordinal,
          event.queue_ordinal, event.stream_id, event.submission_id);
      if (iree_status_is_ok(status)) {
        status =
            iree_profile_queue_event_query_append_device_event(query, &event);
      }
    }
  }
  return status;
}

static iree_status_t iree_profile_queue_event_query_process_host_executions(
    iree_profile_queue_event_query_t* query, const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter) {
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
    ++query->total_host_execution_event_count;
    if (iree_profile_queue_operation_matches(event.type, event.submission_id,
                                             id_filter, filter)) {
      ++query->matched_host_execution_event_count;
      status = iree_profile_queue_event_query_verify_queue(
          model, "host execution event", event.physical_device_ordinal,
          event.queue_ordinal, event.stream_id, event.submission_id);
      if (iree_status_is_ok(status)) {
        status =
            iree_profile_queue_event_query_append_host_execution(query, &event);
      }
    }
  }
  return status;
}

iree_status_t iree_profile_queue_event_query_process_record(
    iree_profile_queue_event_query_t* query, const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(record->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
    return iree_profile_queue_event_query_process_queue_events(
        query, model, record, filter, id_filter);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS)) {
    return iree_profile_queue_event_query_process_device_events(
        query, model, record, filter, id_filter);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS)) {
    return iree_profile_queue_event_query_process_host_executions(
        query, model, record, filter, id_filter);
  }
  return iree_ok_status();
}
