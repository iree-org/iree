// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/internal.h"

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

static void iree_profile_summary_record_dispatch_event(
    iree_profile_device_summary_t* device,
    const iree_hal_profile_dispatch_event_t* record) {
  ++device->dispatch_event_count;
  if (record->start_tick == 0 || record->end_tick == 0 ||
      record->end_tick < record->start_tick) {
    ++device->invalid_dispatch_event_count;
    return;
  }

  const uint64_t duration_ticks = record->end_tick - record->start_tick;
  device->total_dispatch_ticks += (double)duration_ticks;
  device->earliest_dispatch_start_tick =
      iree_min(device->earliest_dispatch_start_tick, record->start_tick);
  device->latest_dispatch_end_tick =
      iree_max(device->latest_dispatch_end_tick, record->end_tick);
  device->minimum_dispatch_ticks =
      iree_min(device->minimum_dispatch_ticks, duration_ticks);
  device->maximum_dispatch_ticks =
      iree_max(device->maximum_dispatch_ticks, duration_ticks);
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
    iree_profile_summary_record_dispatch_event(device, &dispatch_record);
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

static iree_status_t iree_profile_summary_process_event_relationship_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  return iree_profile_summary_count_typed_records(
      record, sizeof(iree_hal_profile_event_relationship_record_t),
      &summary->event_relationship_record_count);
}

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
  }

  if (iree_string_view_equal(record->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES)) {
    ++summary->device_chunk_count;
    return iree_profile_summary_process_device_records(summary, record);
  } else if (iree_string_view_equal(record->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
    ++summary->queue_chunk_count;
    return iree_profile_summary_process_queue_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
    ++summary->executable_chunk_count;
    return iree_profile_summary_process_executable_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECTS)) {
    ++summary->executable_code_object_chunk_count;
    return iree_profile_summary_process_executable_code_object_records(summary,
                                                                       record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECT_LOADS)) {
    ++summary->executable_code_object_load_chunk_count;
    return iree_profile_summary_process_executable_code_object_load_records(
        summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    ++summary->executable_export_chunk_count;
    return iree_profile_summary_process_executable_export_records(summary,
                                                                  record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS)) {
    ++summary->command_buffer_chunk_count;
    return iree_profile_summary_process_command_buffer_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS)) {
    ++summary->command_operation_chunk_count;
    return iree_profile_summary_process_command_operation_records(summary,
                                                                  record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    ++summary->clock_correlation_chunk_count;
    return iree_profile_summary_process_clock_correlation_records(summary,
                                                                  record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    ++summary->dispatch_event_chunk_count;
    return iree_profile_summary_process_dispatch_event_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
    ++summary->queue_event_chunk_count;
    return iree_profile_summary_process_queue_event_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS)) {
    ++summary->queue_device_event_chunk_count;
    return iree_profile_summary_process_queue_device_event_records(summary,
                                                                   record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
    ++summary->memory_event_chunk_count;
    return iree_profile_summary_process_memory_event_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EVENT_RELATIONSHIPS)) {
    ++summary->event_relationship_chunk_count;
    return iree_profile_summary_process_event_relationship_records(summary,
                                                                   record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS)) {
    ++summary->counter_set_chunk_count;
    return iree_profile_summary_process_counter_set_records(summary, record);
  } else if (iree_string_view_equal(record->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS)) {
    ++summary->counter_chunk_count;
    return iree_profile_summary_process_counter_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES)) {
    ++summary->counter_sample_chunk_count;
    return iree_profile_summary_process_counter_sample_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACES)) {
    ++summary->executable_trace_chunk_count;
    return iree_profile_summary_process_executable_trace_record(summary,
                                                                record);
  }

  ++summary->unknown_chunk_count;
  return iree_ok_status();
}

static bool iree_profile_device_summary_try_fit_clock(
    const iree_profile_device_summary_t* device, double* out_ns_per_tick,
    double* out_tick_frequency_hz) {
  *out_ns_per_tick = 0.0;
  *out_tick_frequency_hz = 0.0;
  if (device->clock_sample_count < 2) return false;

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

static void iree_profile_print_summary_text(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file, "IREE HAL profile summary\n");
  fprintf(file,
          "records: file=%" PRIu64 " session_begin=%" PRIu64 " chunks=%" PRIu64
          " session_end=%" PRIu64 " unknown=%" PRIu64 "\n",
          summary->file_record_count, summary->session_begin_count,
          summary->chunk_count, summary->session_end_count,
          summary->unknown_record_count);
  fprintf(
      file,
      "chunks: devices=%" PRIu64 " queues=%" PRIu64 " executables=%" PRIu64
      " executable_code_objects=%" PRIu64
      " executable_code_object_loads=%" PRIu64 " executable_exports=%" PRIu64
      " command_buffers=%" PRIu64 " command_operations=%" PRIu64
      " clock_correlations=%" PRIu64 " dispatch_events=%" PRIu64
      " queue_events=%" PRIu64 " queue_device_events=%" PRIu64
      " memory_events=%" PRIu64 " event_relationships=%" PRIu64
      " counter_sets=%" PRIu64 " counters=%" PRIu64 " counter_samples=%" PRIu64
      " executable_traces=%" PRIu64 " unknown=%" PRIu64 " truncated=%" PRIu64
      "\n",
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
      summary->memory_event_chunk_count,
      summary->event_relationship_chunk_count, summary->counter_set_chunk_count,
      summary->counter_chunk_count, summary->counter_sample_chunk_count,
      summary->executable_trace_chunk_count, summary->unknown_chunk_count,
      summary->truncated_chunk_count);
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
  fprintf(file,
          "event_records: queue_events=%" PRIu64 " queue_device_events=%" PRIu64
          " memory_events=%" PRIu64 " event_relationships=%" PRIu64
          " counter_samples=%" PRIu64 " executable_traces=%" PRIu64 "\n",
          summary->queue_event_record_count,
          summary->queue_device_event_record_count,
          summary->memory_event_record_count,
          summary->event_relationship_record_count,
          summary->counter_sample_record_count,
          summary->executable_trace_record_count);
  fprintf(file,
          "trace_data_bytes: code_objects=%" PRIu64
          " executable_traces=%" PRIu64 "\n",
          summary->executable_code_object_data_bytes,
          summary->executable_trace_data_bytes);
  fprintf(file,
          "counter_records: counter_sets=%" PRIu64 " counters=%" PRIu64 "\n",
          summary->counter_set_record_count, summary->counter_record_count);
  fprintf(file, "devices:\n");

  for (iree_host_size_t i = 0; i < summary->device_count; ++i) {
    const iree_profile_device_summary_t* device = &summary->devices[i];
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_device_summary_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    const bool clock_covers_dispatches =
        iree_profile_device_summary_clock_covers_dispatches(device);
    const uint64_t valid_dispatch_count =
        device->dispatch_event_count - device->invalid_dispatch_event_count;

    fprintf(file, "  device[%u]: device_records=%u queues=%u/%u\n",
            device->physical_device_ordinal, device->device_record_count,
            device->queue_record_count, device->queue_count);
    fprintf(file,
            "    clock_samples=%" PRIu64 " min_uncertainty_ns=%" PRId64
            " max_uncertainty_ns=%" PRId64 "\n",
            device->clock_sample_count,
            device->minimum_clock_uncertainty_ns == INT64_MAX
                ? 0
                : device->minimum_clock_uncertainty_ns,
            device->maximum_clock_uncertainty_ns);
    if (has_clock_fit) {
      fprintf(file,
              "    clock_fit: ns_per_tick=%.9f tick_frequency_hz=%.3f"
              " device_delta_ticks=%" PRIu64 " host_delta_ns=%" PRIu64 "\n",
              ns_per_tick, tick_frequency_hz,
              device->last_clock_sample.device_tick -
                  device->first_clock_sample.device_tick,
              device->last_clock_sample.host_cpu_timestamp_ns -
                  device->first_clock_sample.host_cpu_timestamp_ns);
    } else {
      fprintf(file, "    clock_fit: unavailable\n");
    }

    fprintf(file,
            "    dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
            "\n",
            device->dispatch_event_count, valid_dispatch_count,
            device->invalid_dispatch_event_count);
    if (valid_dispatch_count != 0) {
      const double average_ticks =
          device->total_dispatch_ticks / (double)valid_dispatch_count;
      fprintf(file,
              "    dispatch_tick_range: start=%" PRIu64 " end=%" PRIu64
              " covered_by_clock_samples=%s\n",
              device->earliest_dispatch_start_tick,
              device->latest_dispatch_end_tick,
              clock_covers_dispatches ? "true" : "false");
      fprintf(file,
              "    dispatch_ticks: min=%" PRIu64 " avg=%.3f max=%" PRIu64
              " total=%.3f\n",
              device->minimum_dispatch_ticks, average_ticks,
              device->maximum_dispatch_ticks, device->total_dispatch_ticks);
      if (has_clock_fit) {
        fprintf(file,
                "    dispatch_time_ns: min=%.3f avg=%.3f max=%.3f"
                " total=%.3f\n",
                (double)device->minimum_dispatch_ticks * ns_per_tick,
                average_ticks * ns_per_tick,
                (double)device->maximum_dispatch_ticks * ns_per_tick,
                device->total_dispatch_ticks * ns_per_tick);
      }
    }
  }
}

static void iree_profile_print_summary_jsonl(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(
      file,
      "{\"type\":\"summary\",\"file_records\":%" PRIu64
      ",\"session_begin_records\":%" PRIu64 ",\"chunk_records\":%" PRIu64
      ",\"session_end_records\":%" PRIu64 ",\"unknown_records\":%" PRIu64
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
      ",\"clock_correlation_chunks\":%" PRIu64
      ",\"dispatch_event_chunks\":%" PRIu64 ",\"queue_event_chunks\":%" PRIu64
      ",\"queue_event_records\":%" PRIu64
      ",\"queue_device_event_chunks\":%" PRIu64
      ",\"queue_device_event_records\":%" PRIu64
      ",\"memory_event_chunks\":%" PRIu64 ",\"memory_event_records\":%" PRIu64
      ",\"event_relationship_chunks\":%" PRIu64
      ",\"event_relationship_records\":%" PRIu64
      ",\"counter_set_chunks\":%" PRIu64 ",\"counter_set_records\":%" PRIu64
      ",\"counter_chunks\":%" PRIu64 ",\"counter_records\":%" PRIu64
      ",\"counter_sample_chunks\":%" PRIu64
      ",\"counter_sample_records\":%" PRIu64
      ",\"executable_trace_chunks\":%" PRIu64
      ",\"executable_trace_records\":%" PRIu64
      ",\"executable_trace_data_bytes\":%" PRIu64 ",\"unknown_chunks\":%" PRIu64
      ",\"truncated_chunks\":%" PRIu64 "}\n",
      summary->file_record_count, summary->session_begin_count,
      summary->chunk_count, summary->session_end_count,
      summary->unknown_record_count, summary->device_chunk_count,
      summary->queue_chunk_count, summary->executable_chunk_count,
      summary->executable_record_count,
      summary->executable_code_object_chunk_count,
      summary->executable_code_object_record_count,
      summary->executable_code_object_data_bytes,
      summary->executable_code_object_load_chunk_count,
      summary->executable_code_object_load_record_count,
      summary->executable_export_chunk_count,
      summary->executable_export_record_count,
      summary->command_buffer_chunk_count, summary->command_buffer_record_count,
      summary->command_operation_chunk_count,
      summary->command_operation_record_count,
      summary->clock_correlation_chunk_count,
      summary->dispatch_event_chunk_count, summary->queue_event_chunk_count,
      summary->queue_event_record_count,
      summary->queue_device_event_chunk_count,
      summary->queue_device_event_record_count,
      summary->memory_event_chunk_count, summary->memory_event_record_count,
      summary->event_relationship_chunk_count,
      summary->event_relationship_record_count,
      summary->counter_set_chunk_count, summary->counter_set_record_count,
      summary->counter_chunk_count, summary->counter_record_count,
      summary->counter_sample_chunk_count, summary->counter_sample_record_count,
      summary->executable_trace_chunk_count,
      summary->executable_trace_record_count,
      summary->executable_trace_data_bytes, summary->unknown_chunk_count,
      summary->truncated_chunk_count);

  for (iree_host_size_t i = 0; i < summary->device_count; ++i) {
    const iree_profile_device_summary_t* device = &summary->devices[i];
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_device_summary_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    const bool clock_covers_dispatches =
        iree_profile_device_summary_clock_covers_dispatches(device);
    const uint64_t valid_dispatch_count =
        device->dispatch_event_count - device->invalid_dispatch_event_count;
    const double average_ticks =
        valid_dispatch_count
            ? device->total_dispatch_ticks / (double)valid_dispatch_count
            : 0.0;
    fprintf(file,
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
            ",\"max_dispatch_ticks\":%" PRIu64
            ",\"total_dispatch_ticks\":%.3f"
            ",\"earliest_dispatch_start_tick\":%" PRIu64
            ",\"latest_dispatch_end_tick\":%" PRIu64
            ",\"dispatch_ticks_covered_by_clock_samples\":%s"
            ",\"min_dispatch_ns\":%.3f,\"avg_dispatch_ns\":%.3f"
            ",\"max_dispatch_ns\":%.3f,\"total_dispatch_ns\":%.3f}\n",
            device->physical_device_ordinal, device->device_record_count,
            device->queue_record_count, device->queue_count,
            device->clock_sample_count, has_clock_fit ? "true" : "false",
            ns_per_tick, tick_frequency_hz,
            device->minimum_clock_uncertainty_ns == INT64_MAX
                ? 0
                : device->minimum_clock_uncertainty_ns,
            device->maximum_clock_uncertainty_ns, device->dispatch_event_count,
            valid_dispatch_count, device->invalid_dispatch_event_count,
            valid_dispatch_count ? device->minimum_dispatch_ticks : 0,
            average_ticks,
            valid_dispatch_count ? device->maximum_dispatch_ticks : 0,
            device->total_dispatch_ticks,
            valid_dispatch_count ? device->earliest_dispatch_start_tick : 0,
            valid_dispatch_count ? device->latest_dispatch_end_tick : 0,
            clock_covers_dispatches ? "true" : "false",
            has_clock_fit && valid_dispatch_count
                ? (double)device->minimum_dispatch_ticks * ns_per_tick
                : 0.0,
            has_clock_fit && valid_dispatch_count ? average_ticks * ns_per_tick
                                                  : 0.0,
            has_clock_fit && valid_dispatch_count
                ? (double)device->maximum_dispatch_ticks * ns_per_tick
                : 0.0,
            has_clock_fit ? device->total_dispatch_ticks * ns_per_tick : 0.0);
  }
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
  iree_status_t status = iree_profile_file_for_each_record(
      &profile_file, iree_profile_summary_record, &summary);

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
