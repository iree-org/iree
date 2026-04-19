// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/summary.h"

#include <stddef.h>
#include <string.h>

#include "iree/tooling/profile/model.h"
#include "iree/tooling/profile/reader.h"

void iree_profile_summary_initialize(iree_allocator_t host_allocator,
                                     iree_profile_summary_t* out_summary) {
  memset(out_summary, 0, sizeof(*out_summary));
  out_summary->host_allocator = host_allocator;
}

void iree_profile_summary_deinitialize(iree_profile_summary_t* summary) {
  iree_allocator_free(summary->host_allocator, summary->devices);
  memset(summary, 0, sizeof(*summary));
}

static iree_status_t iree_profile_summary_get_device(
    iree_profile_summary_t* summary, uint32_t physical_device_ordinal,
    iree_profile_device_summary_t** out_device) {
  *out_device = NULL;

  for (iree_host_size_t i = 0; i < summary->device_count; ++i) {
    if (summary->devices[i].physical_device_ordinal ==
        physical_device_ordinal) {
      *out_device = &summary->devices[i];
      return iree_ok_status();
    }
  }

  if (summary->device_count + 1 > summary->device_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        summary->host_allocator,
        iree_max((iree_host_size_t)4, summary->device_count + 1),
        sizeof(summary->devices[0]), &summary->device_capacity,
        (void**)&summary->devices));
  }

  iree_profile_device_summary_t* device =
      &summary->devices[summary->device_count++];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  device->minimum_clock_uncertainty_ns = INT64_MAX;
  device->earliest_dispatch_start_tick = UINT64_MAX;
  device->minimum_dispatch_ticks = UINT64_MAX;
  *out_device = device;
  return iree_ok_status();
}

static void iree_profile_summary_record_clock_sample(
    iree_profile_device_summary_t* device,
    const iree_hal_profile_clock_correlation_record_t* record) {
  if (device->clock_sample_count == 0) {
    device->first_clock_sample = *record;
  }
  device->last_clock_sample = *record;
  ++device->clock_sample_count;

  if (iree_all_bits_set(
          record->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET) &&
      record->host_time_end_ns >= record->host_time_begin_ns) {
    const int64_t uncertainty_ns =
        record->host_time_end_ns - record->host_time_begin_ns;
    device->minimum_clock_uncertainty_ns =
        iree_min(device->minimum_clock_uncertainty_ns, uncertainty_ns);
    device->maximum_clock_uncertainty_ns =
        iree_max(device->maximum_clock_uncertainty_ns, uncertainty_ns);
  }
}

static bool iree_profile_summary_accumulate_ticks(uint64_t* total_ticks,
                                                  uint64_t duration_ticks) {
  if (duration_ticks > UINT64_MAX - *total_ticks) return false;
  *total_ticks += duration_ticks;
  return true;
}

static bool iree_profile_summary_record_dispatch_event(
    iree_profile_device_summary_t* device,
    const iree_hal_profile_dispatch_event_t* record) {
  ++device->dispatch_event_count;
  if (record->start_tick == 0 || record->end_tick == 0 ||
      record->end_tick < record->start_tick) {
    ++device->invalid_dispatch_event_count;
    return true;
  }

  const uint64_t duration_ticks = record->end_tick - record->start_tick;
  if (!iree_profile_summary_accumulate_ticks(&device->total_dispatch_ticks,
                                             duration_ticks)) {
    return false;
  }
  device->earliest_dispatch_start_tick =
      iree_min(device->earliest_dispatch_start_tick, record->start_tick);
  device->latest_dispatch_end_tick =
      iree_max(device->latest_dispatch_end_tick, record->end_tick);
  device->minimum_dispatch_ticks =
      iree_min(device->minimum_dispatch_ticks, duration_ticks);
  device->maximum_dispatch_ticks =
      iree_max(device->maximum_dispatch_ticks, duration_ticks);
  return true;
}

static iree_status_t iree_profile_summary_count_typed_records(
    const iree_hal_profile_file_record_t* record,
    iree_host_size_t minimum_record_length, uint64_t* record_count) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(record, minimum_record_length,
                                                &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;
    ++*record_count;
  }
  return status;
}

static iree_status_t iree_profile_summary_process_device_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_device_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_device_record_t device_record;
    memcpy(&device_record, typed_record.contents.data, sizeof(device_record));

    iree_profile_device_summary_t* device = NULL;
    status = iree_profile_summary_get_device(
        summary, device_record.physical_device_ordinal, &device);
    if (iree_status_is_ok(status)) {
      ++device->device_record_count;
      device->queue_count = device_record.queue_count;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_queue_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
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

    iree_hal_profile_queue_record_t queue_record;
    memcpy(&queue_record, typed_record.contents.data, sizeof(queue_record));

    iree_profile_device_summary_t* device = NULL;
    status = iree_profile_summary_get_device(
        summary, queue_record.physical_device_ordinal, &device);
    if (iree_status_is_ok(status)) {
      ++device->queue_record_count;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_executable_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  return iree_profile_summary_count_typed_records(
      record, sizeof(iree_hal_profile_executable_record_t),
      &summary->executable_record_count);
}

static iree_status_t
iree_profile_summary_process_executable_code_object_records(
    iree_profile_summary_t* summary,
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
          "profile executable code-object data length is inconsistent with "
          "record length");
    }
    if (iree_status_is_ok(status) &&
        code_object_record.data_length >
            UINT64_MAX - summary->executable_code_object_data_bytes) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "profile executable code-object byte counter overflowed");
    }
    if (iree_status_is_ok(status)) {
      ++summary->executable_code_object_record_count;
      summary->executable_code_object_data_bytes +=
          code_object_record.data_length;
    }
  }
  return status;
}

static iree_status_t
iree_profile_summary_process_executable_code_object_load_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  return iree_profile_summary_count_typed_records(
      record, sizeof(iree_hal_profile_executable_code_object_load_record_t),
      &summary->executable_code_object_load_record_count);
}

static iree_status_t iree_profile_summary_process_executable_export_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  return iree_profile_summary_count_typed_records(
      record, sizeof(iree_hal_profile_executable_export_record_t),
      &summary->executable_export_record_count);
}

static iree_status_t iree_profile_summary_process_command_buffer_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  return iree_profile_summary_count_typed_records(
      record, sizeof(iree_hal_profile_command_buffer_record_t),
      &summary->command_buffer_record_count);
}

static iree_status_t iree_profile_summary_process_command_operation_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  return iree_profile_summary_count_typed_records(
      record, sizeof(iree_hal_profile_command_operation_record_t),
      &summary->command_operation_record_count);
}

static iree_status_t iree_profile_summary_process_clock_correlation_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
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

    iree_profile_device_summary_t* device = NULL;
    status = iree_profile_summary_get_device(
        summary, clock_record.physical_device_ordinal, &device);
    if (iree_status_is_ok(status)) {
      iree_profile_summary_record_clock_sample(device, &clock_record);
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_dispatch_event_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_device_summary_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_summary_get_device(
      summary, record->header.physical_device_ordinal, &device));

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

    iree_hal_profile_dispatch_event_t dispatch_record;
    memcpy(&dispatch_record, typed_record.contents.data,
           sizeof(dispatch_record));
    if (!iree_profile_summary_record_dispatch_event(device, &dispatch_record)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "device summary dispatch tick total overflow device=%u",
          record->header.physical_device_ordinal);
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_memory_event_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  return iree_profile_summary_count_typed_records(
      record, sizeof(iree_hal_profile_memory_event_t),
      &summary->memory_event_record_count);
}

static iree_status_t iree_profile_summary_process_counter_set_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_counter_set_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_counter_set_record_t counter_set_record;
    memcpy(&counter_set_record, typed_record.contents.data,
           sizeof(counter_set_record));
    if ((iree_host_size_t)counter_set_record.name_length !=
        typed_record.inline_payload.data_length) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "counter set name length is inconsistent with record length");
    }
    if (iree_status_is_ok(status)) {
      ++summary->counter_set_record_count;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_counter_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_counter_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_counter_record_t counter_record;
    memcpy(&counter_record, typed_record.contents.data, sizeof(counter_record));
    iree_host_size_t trailing_length = 0;
    if (!iree_host_size_checked_add(counter_record.block_name_length,
                                    counter_record.name_length,
                                    &trailing_length) ||
        !iree_host_size_checked_add(trailing_length,
                                    counter_record.description_length,
                                    &trailing_length) ||
        trailing_length != typed_record.inline_payload.data_length) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "counter string lengths are inconsistent with record length");
    }
    if (iree_status_is_ok(status)) {
      ++summary->counter_record_count;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_counter_sample_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_counter_sample_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_counter_sample_record_t sample_record;
    memcpy(&sample_record, typed_record.contents.data, sizeof(sample_record));
    iree_host_size_t values_length = 0;
    if (!iree_host_size_checked_mul(sample_record.sample_value_count,
                                    sizeof(uint64_t), &values_length) ||
        values_length != typed_record.inline_payload.data_length) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "counter sample value count is inconsistent with record length");
    }
    if (iree_status_is_ok(status)) {
      ++summary->counter_sample_record_count;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_executable_trace_record(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_t typed_record;
  IREE_RETURN_IF_ERROR(iree_profile_typed_record_parse(
      record, 0, sizeof(iree_hal_profile_executable_trace_record_t), 0,
      &typed_record));
  iree_hal_profile_executable_trace_record_t trace_record;
  memcpy(&trace_record, typed_record.contents.data, sizeof(trace_record));
  if ((iree_host_size_t)trace_record.data_length !=
      typed_record.following_payload.data_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile executable trace chunk data length is inconsistent with "
        "payload length");
  }

  if (trace_record.data_length >
      UINT64_MAX - summary->executable_trace_data_bytes) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile executable trace byte counter overflowed");
  }

  ++summary->executable_trace_record_count;
  summary->executable_trace_data_bytes += trace_record.data_length;
  return iree_ok_status();
}

static iree_status_t iree_profile_summary_process_queue_event_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  return iree_profile_summary_count_typed_records(
      record, sizeof(iree_hal_profile_queue_event_t),
      &summary->queue_event_record_count);
}

static iree_status_t iree_profile_summary_process_queue_device_event_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  return iree_profile_summary_count_typed_records(
      record, sizeof(iree_hal_profile_queue_device_event_t),
      &summary->queue_device_event_record_count);
}

static iree_status_t iree_profile_summary_process_host_execution_event_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
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
    ++summary->host_execution_event_record_count;
    if (event.start_host_time_ns < 0 ||
        event.end_host_time_ns < event.start_host_time_ns) {
      ++summary->invalid_host_execution_event_record_count;
      continue;
    }
    const uint64_t duration_ns =
        (uint64_t)(event.end_host_time_ns - event.start_host_time_ns);
    if (duration_ns > UINT64_MAX - summary->total_host_execution_duration_ns) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "profile host execution duration counter overflowed");
    }
    summary->total_host_execution_duration_ns += duration_ns;
  }
  return status;
}

static iree_status_t iree_profile_summary_process_event_relationship_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  return iree_profile_summary_count_typed_records(
      record, sizeof(iree_hal_profile_event_relationship_record_t),
      &summary->event_relationship_record_count);
}

typedef iree_status_t (*iree_profile_summary_chunk_processor_fn_t)(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record);

typedef struct iree_profile_summary_chunk_route_t {
  // Content type matched against chunk records.
  iree_string_view_t content_type;
  // Byte offset of the summary chunk counter incremented for this content type.
  iree_host_size_t chunk_count_offset;
  // Processor invoked after the chunk counter is incremented.
  iree_profile_summary_chunk_processor_fn_t process;
} iree_profile_summary_chunk_route_t;

#define IREE_PROFILE_SUMMARY_CHUNK_ROUTE(content_type, counter_field, \
                                         process_fn)                  \
  {                                                                   \
      content_type,                                                   \
      offsetof(iree_profile_summary_t, counter_field),                \
      process_fn,                                                     \
  }

static uint64_t* iree_profile_summary_chunk_counter(
    iree_profile_summary_t* summary,
    const iree_profile_summary_chunk_route_t* route) {
  return (uint64_t*)((uint8_t*)summary + route->chunk_count_offset);
}

static iree_status_t iree_profile_summary_process_chunk_record(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  const iree_profile_summary_chunk_route_t iree_profile_summary_chunk_routes[] =
      {
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES, device_chunk_count,
              iree_profile_summary_process_device_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES, queue_chunk_count,
              iree_profile_summary_process_queue_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES, executable_chunk_count,
              iree_profile_summary_process_executable_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECTS,
              executable_code_object_chunk_count,
              iree_profile_summary_process_executable_code_object_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECT_LOADS,
              executable_code_object_load_chunk_count,
              iree_profile_summary_process_executable_code_object_load_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS,
              executable_export_chunk_count,
              iree_profile_summary_process_executable_export_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS,
              command_buffer_chunk_count,
              iree_profile_summary_process_command_buffer_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS,
              command_operation_chunk_count,
              iree_profile_summary_process_command_operation_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS,
              clock_correlation_chunk_count,
              iree_profile_summary_process_clock_correlation_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS,
              dispatch_event_chunk_count,
              iree_profile_summary_process_dispatch_event_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS,
              queue_event_chunk_count,
              iree_profile_summary_process_queue_event_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS,
              queue_device_event_chunk_count,
              iree_profile_summary_process_queue_device_event_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS,
              host_execution_event_chunk_count,
              iree_profile_summary_process_host_execution_event_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS,
              memory_event_chunk_count,
              iree_profile_summary_process_memory_event_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_EVENT_RELATIONSHIPS,
              event_relationship_chunk_count,
              iree_profile_summary_process_event_relationship_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS,
              counter_set_chunk_count,
              iree_profile_summary_process_counter_set_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS, counter_chunk_count,
              iree_profile_summary_process_counter_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES,
              counter_sample_chunk_count,
              iree_profile_summary_process_counter_sample_records),
          IREE_PROFILE_SUMMARY_CHUNK_ROUTE(
              IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACES,
              executable_trace_chunk_count,
              iree_profile_summary_process_executable_trace_record),
      };

  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_profile_summary_chunk_routes); ++i) {
    const iree_profile_summary_chunk_route_t* route =
        &iree_profile_summary_chunk_routes[i];
    if (!iree_string_view_equal(record->content_type, route->content_type)) {
      continue;
    }
    ++*iree_profile_summary_chunk_counter(summary, route);
    return route->process(summary, record);
  }

  ++summary->unknown_chunk_count;
  return iree_ok_status();
}

#undef IREE_PROFILE_SUMMARY_CHUNK_ROUTE

iree_status_t iree_profile_summary_process_record(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  ++summary->file_record_count;

  switch (record->header.record_type) {
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN:
      ++summary->session_begin_count;
      return iree_ok_status();
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END:
      ++summary->session_end_count;
      return iree_ok_status();
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK:
      ++summary->chunk_count;
      break;
    default:
      ++summary->unknown_record_count;
      return iree_ok_status();
  }

  if (iree_any_bit_set(record->header.chunk_flags,
                       IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED)) {
    ++summary->truncated_chunk_count;
    summary->dropped_record_count += record->header.dropped_record_count;
  }

  return iree_profile_summary_process_chunk_record(summary, record);
}

static bool iree_profile_device_summary_try_fit_clock_exact(
    const iree_profile_device_summary_t* device,
    iree_profile_model_clock_fit_t* out_clock_fit) {
  iree_profile_model_device_t model_device;
  memset(&model_device, 0, sizeof(model_device));
  model_device.physical_device_ordinal = device->physical_device_ordinal;
  model_device.clock_sample_count = device->clock_sample_count;
  model_device.first_clock_sample = device->first_clock_sample;
  model_device.last_clock_sample = device->last_clock_sample;
  return iree_profile_model_device_try_fit_clock_exact(
      &model_device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS,
      out_clock_fit);
}

static bool iree_profile_device_summary_clock_covers_dispatches(
    const iree_profile_device_summary_t* device) {
  const uint64_t valid_dispatch_count =
      device->dispatch_event_count - device->invalid_dispatch_event_count;
  if (device->clock_sample_count < 2 || valid_dispatch_count == 0) {
    return false;
  }
  if (!iree_all_bits_set(device->first_clock_sample.flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK) ||
      !iree_all_bits_set(device->last_clock_sample.flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK)) {
    return false;
  }
  return device->first_clock_sample.device_tick <=
             device->earliest_dispatch_start_tick &&
         device->latest_dispatch_end_tick <=
             device->last_clock_sample.device_tick;
}

typedef struct iree_profile_summary_device_timing_t {
  // True when the device has an exact device-tick to host-time fit.
  bool has_clock_fit;
  // True when valid dispatch ticks are covered by the clock sample range.
  bool clock_covers_dispatches;
  // True when min/max/total dispatch ticks were converted to nanoseconds.
  bool has_dispatch_ns;
  // Valid dispatch event count for the device.
  uint64_t valid_dispatch_count;
  // Average dispatch duration in raw device ticks.
  double average_dispatch_ticks;
  // Nanoseconds per device tick when |has_clock_fit| is true.
  double ns_per_tick;
  // Device tick frequency in hertz when |has_clock_fit| is true.
  double tick_frequency_hz;
  // Minimum valid dispatch duration in nanoseconds when |has_dispatch_ns| is
  // true.
  int64_t minimum_dispatch_ns;
  // Maximum valid dispatch duration in nanoseconds when |has_dispatch_ns| is
  // true.
  int64_t maximum_dispatch_ns;
  // Total valid dispatch duration in nanoseconds when |has_dispatch_ns| is
  // true.
  int64_t total_dispatch_ns;
} iree_profile_summary_device_timing_t;

static iree_profile_summary_device_timing_t
iree_profile_summary_calculate_device_timing(
    const iree_profile_device_summary_t* device) {
  iree_profile_summary_device_timing_t timing = {0};
  timing.valid_dispatch_count =
      device->dispatch_event_count - device->invalid_dispatch_event_count;
  timing.average_dispatch_ticks =
      timing.valid_dispatch_count
          ? device->total_dispatch_ticks / (double)timing.valid_dispatch_count
          : 0.0;
  timing.clock_covers_dispatches =
      iree_profile_device_summary_clock_covers_dispatches(device);

  iree_profile_model_clock_fit_t clock_fit;
  timing.has_clock_fit =
      iree_profile_device_summary_try_fit_clock_exact(device, &clock_fit);
  timing.ns_per_tick =
      timing.has_clock_fit
          ? iree_profile_model_clock_fit_ns_per_tick(&clock_fit)
          : 0.0;
  timing.tick_frequency_hz =
      timing.has_clock_fit
          ? iree_profile_model_clock_fit_tick_frequency_hz(&clock_fit)
          : 0.0;
  timing.has_dispatch_ns =
      timing.has_clock_fit && timing.valid_dispatch_count != 0 &&
      iree_profile_model_clock_fit_scale_ticks_to_ns(
          &clock_fit, device->minimum_dispatch_ticks,
          &timing.minimum_dispatch_ns) &&
      iree_profile_model_clock_fit_scale_ticks_to_ns(
          &clock_fit, device->maximum_dispatch_ticks,
          &timing.maximum_dispatch_ns) &&
      iree_profile_model_clock_fit_scale_ticks_to_ns(
          &clock_fit, device->total_dispatch_ticks, &timing.total_dispatch_ns);
  return timing;
}

static int64_t iree_profile_summary_minimum_clock_uncertainty_ns(
    const iree_profile_device_summary_t* device) {
  return device->minimum_clock_uncertainty_ns == INT64_MAX
             ? 0
             : device->minimum_clock_uncertainty_ns;
}

static void iree_profile_print_summary_text_records(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file,
          "records: file=%" PRIu64 " session_begin=%" PRIu64 " chunks=%" PRIu64
          " session_end=%" PRIu64 " unknown=%" PRIu64 "\n",
          summary->file_record_count, summary->session_begin_count,
          summary->chunk_count, summary->session_end_count,
          summary->unknown_record_count);
}

static void iree_profile_print_summary_text_chunks(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(
      file,
      "chunks: devices=%" PRIu64 " queues=%" PRIu64 " executables=%" PRIu64
      " executable_code_objects=%" PRIu64
      " executable_code_object_loads=%" PRIu64 " executable_exports=%" PRIu64
      " command_buffers=%" PRIu64 " command_operations=%" PRIu64
      " clock_correlations=%" PRIu64 " dispatch_events=%" PRIu64
      " queue_events=%" PRIu64 " queue_device_events=%" PRIu64
      " host_execution_events=%" PRIu64 " memory_events=%" PRIu64
      " event_relationships=%" PRIu64 " counter_sets=%" PRIu64
      " counters=%" PRIu64 " counter_samples=%" PRIu64
      " executable_traces=%" PRIu64 " unknown=%" PRIu64 " truncated=%" PRIu64
      " dropped_records=%" PRIu64 "\n",
      summary->device_chunk_count, summary->queue_chunk_count,
      summary->executable_chunk_count,
      summary->executable_code_object_chunk_count,
      summary->executable_code_object_load_chunk_count,
      summary->executable_export_chunk_count,
      summary->command_buffer_chunk_count,
      summary->command_operation_chunk_count,
      summary->clock_correlation_chunk_count,
      summary->dispatch_event_chunk_count, summary->queue_event_chunk_count,
      summary->queue_device_event_chunk_count,
      summary->host_execution_event_chunk_count,
      summary->memory_event_chunk_count,
      summary->event_relationship_chunk_count, summary->counter_set_chunk_count,
      summary->counter_chunk_count, summary->counter_sample_chunk_count,
      summary->executable_trace_chunk_count, summary->unknown_chunk_count,
      summary->truncated_chunk_count, summary->dropped_record_count);
}

static void iree_profile_print_summary_text_metadata_records(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file,
          "metadata_records: executables=%" PRIu64
          " executable_code_objects=%" PRIu64
          " executable_code_object_loads=%" PRIu64
          " executable_exports=%" PRIu64 " command_buffers=%" PRIu64
          " command_operations=%" PRIu64 "\n",
          summary->executable_record_count,
          summary->executable_code_object_record_count,
          summary->executable_code_object_load_record_count,
          summary->executable_export_record_count,
          summary->command_buffer_record_count,
          summary->command_operation_record_count);
}

static void iree_profile_print_summary_text_event_records(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file,
          "event_records: queue_events=%" PRIu64 " queue_device_events=%" PRIu64
          " host_execution_events=%" PRIu64
          " invalid_host_execution_events=%" PRIu64
          " host_execution_duration_ns=%" PRIu64 " memory_events=%" PRIu64
          " event_relationships=%" PRIu64 " counter_samples=%" PRIu64
          " executable_traces=%" PRIu64 "\n",
          summary->queue_event_record_count,
          summary->queue_device_event_record_count,
          summary->host_execution_event_record_count,
          summary->invalid_host_execution_event_record_count,
          summary->total_host_execution_duration_ns,
          summary->memory_event_record_count,
          summary->event_relationship_record_count,
          summary->counter_sample_record_count,
          summary->executable_trace_record_count);
}

static void iree_profile_print_summary_text_trace_data(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file,
          "trace_data_bytes: code_objects=%" PRIu64
          " executable_traces=%" PRIu64 "\n",
          summary->executable_code_object_data_bytes,
          summary->executable_trace_data_bytes);
}

static void iree_profile_print_summary_text_counter_records(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file,
          "counter_records: counter_sets=%" PRIu64 " counters=%" PRIu64 "\n",
          summary->counter_set_record_count, summary->counter_record_count);
}

static void iree_profile_print_summary_text_device_clock(
    const iree_profile_device_summary_t* device,
    const iree_profile_summary_device_timing_t* timing, FILE* file) {
  fprintf(file,
          "    clock_samples=%" PRIu64 " min_uncertainty_ns=%" PRId64
          " max_uncertainty_ns=%" PRId64 "\n",
          device->clock_sample_count,
          iree_profile_summary_minimum_clock_uncertainty_ns(device),
          device->maximum_clock_uncertainty_ns);
  if (timing->has_clock_fit) {
    fprintf(file,
            "    clock_fit: ns_per_tick=%.9f tick_frequency_hz=%.3f"
            " device_delta_ticks=%" PRIu64 " host_delta_ns=%" PRIu64 "\n",
            timing->ns_per_tick, timing->tick_frequency_hz,
            device->last_clock_sample.device_tick -
                device->first_clock_sample.device_tick,
            device->last_clock_sample.host_cpu_timestamp_ns -
                device->first_clock_sample.host_cpu_timestamp_ns);
  } else {
    fprintf(file, "    clock_fit: unavailable\n");
  }
}

static void iree_profile_print_summary_text_device_dispatches(
    const iree_profile_device_summary_t* device,
    const iree_profile_summary_device_timing_t* timing, FILE* file) {
  fprintf(file,
          "    dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64 "\n",
          device->dispatch_event_count, timing->valid_dispatch_count,
          device->invalid_dispatch_event_count);
  if (timing->valid_dispatch_count == 0) {
    return;
  }

  fprintf(file,
          "    dispatch_tick_range: start=%" PRIu64 " end=%" PRIu64
          " covered_by_clock_samples=%s\n",
          device->earliest_dispatch_start_tick,
          device->latest_dispatch_end_tick,
          timing->clock_covers_dispatches ? "true" : "false");
  fprintf(file,
          "    dispatch_ticks: min=%" PRIu64 " avg=%.3f max=%" PRIu64
          " total=%" PRIu64 "\n",
          device->minimum_dispatch_ticks, timing->average_dispatch_ticks,
          device->maximum_dispatch_ticks, device->total_dispatch_ticks);
  if (!timing->has_clock_fit) {
    return;
  }
  if (timing->has_dispatch_ns) {
    fprintf(file,
            "    dispatch_time_ns: min=%" PRId64
            " avg=%.3f"
            " max=%" PRId64 " total=%" PRId64 "\n",
            timing->minimum_dispatch_ns,
            timing->average_dispatch_ticks * timing->ns_per_tick,
            timing->maximum_dispatch_ns, timing->total_dispatch_ns);
  } else {
    fprintf(file, "    dispatch_time_ns: unavailable\n");
  }
}

static void iree_profile_print_summary_text_device(
    const iree_profile_device_summary_t* device, FILE* file) {
  const iree_profile_summary_device_timing_t timing =
      iree_profile_summary_calculate_device_timing(device);
  fprintf(file, "  device[%u]: device_records=%u queues=%u/%u\n",
          device->physical_device_ordinal, device->device_record_count,
          device->queue_record_count, device->queue_count);
  iree_profile_print_summary_text_device_clock(device, &timing, file);
  iree_profile_print_summary_text_device_dispatches(device, &timing, file);
}

static void iree_profile_print_summary_text_devices(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file, "devices:\n");
  for (iree_host_size_t i = 0; i < summary->device_count; ++i) {
    iree_profile_print_summary_text_device(&summary->devices[i], file);
  }
}

static void iree_profile_print_summary_text(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file, "IREE HAL profile summary\n");
  iree_profile_print_summary_text_records(summary, file);
  iree_profile_print_summary_text_chunks(summary, file);
  iree_profile_print_summary_text_metadata_records(summary, file);
  iree_profile_print_summary_text_event_records(summary, file);
  iree_profile_print_summary_text_trace_data(summary, file);
  iree_profile_print_summary_text_counter_records(summary, file);
  iree_profile_print_summary_text_devices(summary, file);
}

static void iree_profile_print_summary_jsonl_record_fields(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file,
          "{\"type\":\"summary\",\"file_records\":%" PRIu64
          ",\"session_begin_records\":%" PRIu64 ",\"chunk_records\":%" PRIu64
          ",\"session_end_records\":%" PRIu64 ",\"unknown_records\":%" PRIu64,
          summary->file_record_count, summary->session_begin_count,
          summary->chunk_count, summary->session_end_count,
          summary->unknown_record_count);
}

static void iree_profile_print_summary_jsonl_metadata_fields(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file,
          ",\"device_chunks\":%" PRIu64 ",\"queue_chunks\":%" PRIu64
          ",\"executable_chunks\":%" PRIu64 ",\"executable_records\":%" PRIu64
          ",\"executable_code_object_chunks\":%" PRIu64
          ",\"executable_code_object_records\":%" PRIu64
          ",\"executable_code_object_data_bytes\":%" PRIu64
          ",\"executable_code_object_load_chunks\":%" PRIu64
          ",\"executable_code_object_load_records\":%" PRIu64
          ",\"executable_export_chunks\":%" PRIu64
          ",\"executable_export_records\":%" PRIu64
          ",\"command_buffer_chunks\":%" PRIu64
          ",\"command_buffer_records\":%" PRIu64
          ",\"command_operation_chunks\":%" PRIu64
          ",\"command_operation_records\":%" PRIu64
          ",\"clock_correlation_chunks\":%" PRIu64,
          summary->device_chunk_count, summary->queue_chunk_count,
          summary->executable_chunk_count, summary->executable_record_count,
          summary->executable_code_object_chunk_count,
          summary->executable_code_object_record_count,
          summary->executable_code_object_data_bytes,
          summary->executable_code_object_load_chunk_count,
          summary->executable_code_object_load_record_count,
          summary->executable_export_chunk_count,
          summary->executable_export_record_count,
          summary->command_buffer_chunk_count,
          summary->command_buffer_record_count,
          summary->command_operation_chunk_count,
          summary->command_operation_record_count,
          summary->clock_correlation_chunk_count);
}

static void iree_profile_print_summary_jsonl_execution_fields(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file,
          ",\"dispatch_event_chunks\":%" PRIu64
          ",\"queue_event_chunks\":%" PRIu64 ",\"queue_event_records\":%" PRIu64
          ",\"queue_device_event_chunks\":%" PRIu64
          ",\"queue_device_event_records\":%" PRIu64
          ",\"host_execution_event_chunks\":%" PRIu64
          ",\"host_execution_event_records\":%" PRIu64
          ",\"invalid_host_execution_event_records\":%" PRIu64
          ",\"total_host_execution_duration_ns\":%" PRIu64
          ",\"memory_event_chunks\":%" PRIu64
          ",\"memory_event_records\":%" PRIu64
          ",\"event_relationship_chunks\":%" PRIu64
          ",\"event_relationship_records\":%" PRIu64,
          summary->dispatch_event_chunk_count, summary->queue_event_chunk_count,
          summary->queue_event_record_count,
          summary->queue_device_event_chunk_count,
          summary->queue_device_event_record_count,
          summary->host_execution_event_chunk_count,
          summary->host_execution_event_record_count,
          summary->invalid_host_execution_event_record_count,
          summary->total_host_execution_duration_ns,
          summary->memory_event_chunk_count, summary->memory_event_record_count,
          summary->event_relationship_chunk_count,
          summary->event_relationship_record_count);
}

static void iree_profile_print_summary_jsonl_counter_trace_fields(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(
      file,
      ",\"counter_set_chunks\":%" PRIu64 ",\"counter_set_records\":%" PRIu64
      ",\"counter_chunks\":%" PRIu64 ",\"counter_records\":%" PRIu64
      ",\"counter_sample_chunks\":%" PRIu64
      ",\"counter_sample_records\":%" PRIu64
      ",\"executable_trace_chunks\":%" PRIu64
      ",\"executable_trace_records\":%" PRIu64
      ",\"executable_trace_data_bytes\":%" PRIu64 ",\"unknown_chunks\":%" PRIu64
      ",\"truncated_chunks\":%" PRIu64 ",\"dropped_records\":%" PRIu64 "}\n",
      summary->counter_set_chunk_count, summary->counter_set_record_count,
      summary->counter_chunk_count, summary->counter_record_count,
      summary->counter_sample_chunk_count, summary->counter_sample_record_count,
      summary->executable_trace_chunk_count,
      summary->executable_trace_record_count,
      summary->executable_trace_data_bytes, summary->unknown_chunk_count,
      summary->truncated_chunk_count, summary->dropped_record_count);
}

static void iree_profile_print_summary_jsonl_summary(
    const iree_profile_summary_t* summary, FILE* file) {
  iree_profile_print_summary_jsonl_record_fields(summary, file);
  iree_profile_print_summary_jsonl_metadata_fields(summary, file);
  iree_profile_print_summary_jsonl_execution_fields(summary, file);
  iree_profile_print_summary_jsonl_counter_trace_fields(summary, file);
}

static void iree_profile_print_summary_jsonl_device(
    const iree_profile_device_summary_t* device, FILE* file) {
  const iree_profile_summary_device_timing_t timing =
      iree_profile_summary_calculate_device_timing(device);
  fprintf(
      file,
      "{\"type\":\"device_summary\",\"physical_device_ordinal\":%u"
      ",\"device_records\":%u,\"queue_records\":%u,\"queues\":%u"
      ",\"clock_samples\":%" PRIu64
      ",\"clock_fit_available\":%s"
      ",\"ns_per_tick\":%.9f,\"tick_frequency_hz\":%.3f"
      ",\"min_clock_uncertainty_ns\":%" PRId64
      ",\"max_clock_uncertainty_ns\":%" PRId64 ",\"dispatches\":%" PRIu64
      ",\"valid_dispatches\":%" PRIu64 ",\"invalid_dispatches\":%" PRIu64
      ",\"min_dispatch_ticks\":%" PRIu64
      ",\"avg_dispatch_ticks\":%.3f"
      ",\"max_dispatch_ticks\":%" PRIu64 ",\"total_dispatch_ticks\":%" PRIu64
      ",\"earliest_dispatch_start_tick\":%" PRIu64
      ",\"latest_dispatch_end_tick\":%" PRIu64
      ",\"dispatch_ticks_covered_by_clock_samples\":%s"
      ",\"dispatch_time_available\":%s"
      ",\"min_dispatch_ns\":%" PRId64
      ",\"avg_dispatch_ns\":%.3f"
      ",\"max_dispatch_ns\":%" PRId64 ",\"total_dispatch_ns\":%" PRId64 "}\n",
      device->physical_device_ordinal, device->device_record_count,
      device->queue_record_count, device->queue_count,
      device->clock_sample_count, timing.has_clock_fit ? "true" : "false",
      timing.ns_per_tick, timing.tick_frequency_hz,
      iree_profile_summary_minimum_clock_uncertainty_ns(device),
      device->maximum_clock_uncertainty_ns, device->dispatch_event_count,
      timing.valid_dispatch_count, device->invalid_dispatch_event_count,
      timing.valid_dispatch_count ? device->minimum_dispatch_ticks : 0,
      timing.average_dispatch_ticks,
      timing.valid_dispatch_count ? device->maximum_dispatch_ticks : 0,
      device->total_dispatch_ticks,
      timing.valid_dispatch_count ? device->earliest_dispatch_start_tick : 0,
      timing.valid_dispatch_count ? device->latest_dispatch_end_tick : 0,
      timing.clock_covers_dispatches ? "true" : "false",
      timing.has_dispatch_ns ? "true" : "false",
      timing.has_dispatch_ns ? timing.minimum_dispatch_ns : 0,
      timing.has_dispatch_ns
          ? timing.average_dispatch_ticks * timing.ns_per_tick
          : 0.0,
      timing.has_dispatch_ns ? timing.maximum_dispatch_ns : 0,
      timing.has_dispatch_ns ? timing.total_dispatch_ns : 0);
}

static void iree_profile_print_summary_jsonl_devices(
    const iree_profile_summary_t* summary, FILE* file) {
  for (iree_host_size_t i = 0; i < summary->device_count; ++i) {
    iree_profile_print_summary_jsonl_device(&summary->devices[i], file);
  }
}

static void iree_profile_print_summary_jsonl(
    const iree_profile_summary_t* summary, FILE* file) {
  iree_profile_print_summary_jsonl_summary(summary, file);
  iree_profile_print_summary_jsonl_devices(summary, file);
}

static iree_status_t iree_profile_summary_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_summary_t* summary = (iree_profile_summary_t*)user_data;
  return iree_profile_summary_process_record(summary, record);
}

iree_status_t iree_profile_summary_file(iree_string_view_t path,
                                        iree_string_view_t format, FILE* file,
                                        iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_profile_file_t profile_file;
  IREE_RETURN_IF_ERROR(
      iree_profile_file_open(path, host_allocator, &profile_file));

  iree_profile_summary_t summary;
  iree_profile_summary_initialize(host_allocator, &summary);
  iree_profile_file_record_callback_t record_callback = {
      .fn = iree_profile_summary_record,
      .user_data = &summary,
  };
  iree_status_t status =
      iree_profile_file_for_each_record(&profile_file, record_callback);

  if (iree_status_is_ok(status)) {
    if (is_text) {
      iree_profile_print_summary_text(&summary, file);
    } else {
      iree_profile_print_summary_jsonl(&summary, file);
    }
  }

  iree_profile_summary_deinitialize(&summary);
  iree_profile_file_close(&profile_file);
  return status;
}
