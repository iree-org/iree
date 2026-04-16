// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, format, "text",
          "Output format for `cat`/`summary`: one of `text` or `jsonl`.");

typedef struct iree_profile_device_summary_t {
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Number of device metadata records seen for this ordinal.
  uint32_t device_record_count;
  // Number of queues reported in device metadata.
  uint32_t queue_count;
  // Number of queue metadata records seen for this physical device.
  uint32_t queue_record_count;
  // Number of clock-correlation samples seen for this physical device.
  uint64_t clock_sample_count;
  // First clock-correlation sample seen for this physical device.
  iree_hal_profile_clock_correlation_record_t first_clock_sample;
  // Last clock-correlation sample seen for this physical device.
  iree_hal_profile_clock_correlation_record_t last_clock_sample;
  // Minimum host bracket length observed around KFD clock-counter samples.
  int64_t minimum_clock_uncertainty_ns;
  // Maximum host bracket length observed around KFD clock-counter samples.
  int64_t maximum_clock_uncertainty_ns;
  // Number of dispatch event records seen for this physical device.
  uint64_t dispatch_event_count;
  // Number of dispatch records with unusable or reversed timestamps.
  uint64_t invalid_dispatch_event_count;
  // Sum of valid dispatch durations in raw device ticks.
  double total_dispatch_ticks;
  // Earliest valid dispatch start tick seen for this physical device.
  uint64_t earliest_dispatch_start_tick;
  // Latest valid dispatch end tick seen for this physical device.
  uint64_t latest_dispatch_end_tick;
  // Minimum valid dispatch duration in raw device ticks.
  uint64_t minimum_dispatch_ticks;
  // Maximum valid dispatch duration in raw device ticks.
  uint64_t maximum_dispatch_ticks;
} iree_profile_device_summary_t;

typedef struct iree_profile_summary_t {
  // Host allocator used for dynamic summary arrays.
  iree_allocator_t host_allocator;
  // Dynamic array of per-device summaries.
  iree_profile_device_summary_t* devices;
  // Number of valid entries in |devices|.
  iree_host_size_t device_count;
  // Capacity of |devices| in entries.
  iree_host_size_t device_capacity;
  // Total file records parsed.
  uint64_t file_record_count;
  // Session-begin records parsed.
  uint64_t session_begin_count;
  // Session-end records parsed.
  uint64_t session_end_count;
  // Chunk records parsed.
  uint64_t chunk_count;
  // Records with unknown file record types.
  uint64_t unknown_record_count;
  // Chunks with the truncated flag set.
  uint64_t truncated_chunk_count;
  // Device metadata chunks parsed.
  uint64_t device_chunk_count;
  // Queue metadata chunks parsed.
  uint64_t queue_chunk_count;
  // Executable metadata chunks parsed.
  uint64_t executable_chunk_count;
  // Executable records parsed.
  uint64_t executable_record_count;
  // Executable export metadata chunks parsed.
  uint64_t executable_export_chunk_count;
  // Executable export records parsed.
  uint64_t executable_export_record_count;
  // Command-buffer metadata chunks parsed.
  uint64_t command_buffer_chunk_count;
  // Command-buffer records parsed.
  uint64_t command_buffer_record_count;
  // Clock-correlation chunks parsed.
  uint64_t clock_correlation_chunk_count;
  // Dispatch event chunks parsed.
  uint64_t dispatch_event_chunk_count;
  // Chunk records with unknown content types.
  uint64_t unknown_chunk_count;
} iree_profile_summary_t;

static const char* iree_profile_record_type_name(
    iree_hal_profile_file_record_type_t record_type) {
  switch (record_type) {
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN:
      return "session_begin";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK:
      return "chunk";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END:
      return "session_end";
    default:
      return "unknown";
  }
}

static void iree_profile_fprint_json_string(FILE* file,
                                            iree_string_view_t value) {
  fputc('"', file);
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    uint8_t c = (uint8_t)value.data[i];
    switch (c) {
      case '"':
        fputs("\\\"", file);
        break;
      case '\\':
        fputs("\\\\", file);
        break;
      case '\b':
        fputs("\\b", file);
        break;
      case '\f':
        fputs("\\f", file);
        break;
      case '\n':
        fputs("\\n", file);
        break;
      case '\r':
        fputs("\\r", file);
        break;
      case '\t':
        fputs("\\t", file);
        break;
      default:
        if (c < 0x20) {
          fprintf(file, "\\u%04x", c);
        } else {
          fputc(c, file);
        }
        break;
    }
  }
  fputc('"', file);
}

static void iree_profile_dump_header_text(
    const iree_hal_profile_file_header_t* header, FILE* file) {
  fprintf(file, "IREE HAL profile bundle\n");
  fprintf(file, "version: %u.%u\n", header->version_major,
          header->version_minor);
  fprintf(file, "header_length: %u\n", header->header_length);
  fprintf(file, "flags: 0x%08x\n", header->flags);
  fprintf(file, "records:\n");
}

static void iree_profile_dump_record_text(
    iree_host_size_t record_index, const iree_hal_profile_file_record_t* record,
    FILE* file) {
  const iree_hal_profile_file_record_header_t* header = &record->header;
  fprintf(file,
          "[%" PRIhsz "] %s record_length=%" PRIu64 " payload_length=%" PRIu64
          "\n",
          record_index, iree_profile_record_type_name(header->record_type),
          header->record_length, header->payload_length);
  fprintf(file, "  content_type: %.*s\n", (int)record->content_type.size,
          record->content_type.data);
  fprintf(file, "  name: %.*s\n", (int)record->name.size, record->name.data);
  fprintf(file,
          "  session_id=%" PRIu64 " stream_id=%" PRIu64 " event_id=%" PRIu64
          "\n",
          header->session_id, header->stream_id, header->event_id);
  fprintf(file, "  executable_id=%" PRIu64 " command_buffer_id=%" PRIu64 "\n",
          header->executable_id, header->command_buffer_id);
  fprintf(file, "  physical_device_ordinal=%u queue_ordinal=%u\n",
          header->physical_device_ordinal, header->queue_ordinal);
  fprintf(file, "  chunk_flags=0x%016" PRIx64 " session_status_code=%u\n",
          header->chunk_flags, header->session_status_code);
}

static void iree_profile_dump_header_jsonl(
    const iree_hal_profile_file_header_t* header, FILE* file) {
  fprintf(file,
          "{\"type\":\"file\",\"magic\":\"IRPF\","
          "\"version_major\":%u,\"version_minor\":%u,"
          "\"header_length\":%u,\"flags\":%u}\n",
          header->version_major, header->version_minor, header->header_length,
          header->flags);
}

static void iree_profile_dump_record_jsonl(
    iree_host_size_t record_index, const iree_hal_profile_file_record_t* record,
    FILE* file) {
  const iree_hal_profile_file_record_header_t* header = &record->header;
  fprintf(file, "{\"type\":\"record\",\"index\":%" PRIhsz, record_index);
  fprintf(file, ",\"record_type\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_record_type_name(header->record_type)));
  fprintf(file, ",\"record_type_value\":%u", header->record_type);
  fprintf(file, ",\"record_length\":%" PRIu64, header->record_length);
  fprintf(file, ",\"payload_length\":%" PRIu64, header->payload_length);
  fprintf(file, ",\"content_type\":");
  iree_profile_fprint_json_string(file, record->content_type);
  fprintf(file, ",\"name\":");
  iree_profile_fprint_json_string(file, record->name);
  fprintf(file, ",\"session_id\":%" PRIu64, header->session_id);
  fprintf(file, ",\"stream_id\":%" PRIu64, header->stream_id);
  fprintf(file, ",\"event_id\":%" PRIu64, header->event_id);
  fprintf(file, ",\"executable_id\":%" PRIu64, header->executable_id);
  fprintf(file, ",\"command_buffer_id\":%" PRIu64, header->command_buffer_id);
  fprintf(file, ",\"physical_device_ordinal\":%u",
          header->physical_device_ordinal);
  fprintf(file, ",\"queue_ordinal\":%u", header->queue_ordinal);
  fprintf(file, ",\"chunk_flags\":%" PRIu64, header->chunk_flags);
  fprintf(file, ",\"session_status_code\":%u", header->session_status_code);
  fputs("}\n", file);
}

static void iree_profile_summary_initialize(
    iree_allocator_t host_allocator, iree_profile_summary_t* out_summary) {
  memset(out_summary, 0, sizeof(*out_summary));
  out_summary->host_allocator = host_allocator;
}

static void iree_profile_summary_deinitialize(iree_profile_summary_t* summary) {
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

static iree_status_t iree_profile_payload_record_length(
    iree_string_view_t content_type, iree_const_byte_span_t payload,
    iree_host_size_t payload_offset, iree_host_size_t minimum_record_length,
    iree_host_size_t* out_record_length) {
  *out_record_length = 0;

  const iree_host_size_t remaining_length =
      payload.data_length - payload_offset;
  if (remaining_length < minimum_record_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile chunk '%.*s' has a truncated typed record",
                            (int)content_type.size, content_type.data);
  }

  uint32_t record_length = 0;
  memcpy(&record_length, payload.data + payload_offset, sizeof(record_length));
  if (record_length < minimum_record_length ||
      record_length > remaining_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile chunk '%.*s' has invalid typed record length %u",
        (int)content_type.size, content_type.data, record_length);
  }

  *out_record_length = record_length;
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

static iree_status_t iree_profile_summary_process_device_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_device_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_device_record_t device_record;
      memcpy(&device_record, record->payload.data + payload_offset,
             sizeof(device_record));

      iree_profile_device_summary_t* device = NULL;
      status = iree_profile_summary_get_device(
          summary, device_record.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        ++device->device_record_count;
        device->queue_count = device_record.queue_count;
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_queue_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_queue_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_queue_record_t queue_record;
      memcpy(&queue_record, record->payload.data + payload_offset,
             sizeof(queue_record));

      iree_profile_device_summary_t* device = NULL;
      status = iree_profile_summary_get_device(
          summary, queue_record.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        ++device->queue_record_count;
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_executable_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_executable_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->executable_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_executable_export_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_executable_export_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->executable_export_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_command_buffer_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_command_buffer_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->command_buffer_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_clock_correlation_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_clock_correlation_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_clock_correlation_record_t clock_record;
      memcpy(&clock_record, record->payload.data + payload_offset,
             sizeof(clock_record));

      iree_profile_device_summary_t* device = NULL;
      status = iree_profile_summary_get_device(
          summary, clock_record.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        iree_profile_summary_record_clock_sample(device, &clock_record);
      }
      payload_offset += record_length;
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

  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_dispatch_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_dispatch_event_t dispatch_record;
      memcpy(&dispatch_record, record->payload.data + payload_offset,
             sizeof(dispatch_record));
      iree_profile_summary_record_dispatch_event(device, &dispatch_record);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_record(
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
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    ++summary->clock_correlation_chunk_count;
    return iree_profile_summary_process_clock_correlation_records(summary,
                                                                  record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    ++summary->dispatch_event_chunk_count;
    return iree_profile_summary_process_dispatch_event_records(summary, record);
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
  fprintf(file,
          "chunks: devices=%" PRIu64 " queues=%" PRIu64 " executables=%" PRIu64
          " executable_exports=%" PRIu64 " command_buffers=%" PRIu64
          " clock_correlations=%" PRIu64 " dispatch_events=%" PRIu64
          " unknown=%" PRIu64 " truncated=%" PRIu64 "\n",
          summary->device_chunk_count, summary->queue_chunk_count,
          summary->executable_chunk_count,
          summary->executable_export_chunk_count,
          summary->command_buffer_chunk_count,
          summary->clock_correlation_chunk_count,
          summary->dispatch_event_chunk_count, summary->unknown_chunk_count,
          summary->truncated_chunk_count);
  fprintf(file,
          "metadata_records: executables=%" PRIu64
          " executable_exports=%" PRIu64 " command_buffers=%" PRIu64 "\n",
          summary->executable_record_count,
          summary->executable_export_record_count,
          summary->command_buffer_record_count);
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
      ",\"executable_export_chunks\":%" PRIu64
      ",\"executable_export_records\":%" PRIu64
      ",\"command_buffer_chunks\":%" PRIu64
      ",\"command_buffer_records\":%" PRIu64
      ",\"clock_correlation_chunks\":%" PRIu64
      ",\"dispatch_event_chunks\":%" PRIu64 ",\"unknown_chunks\":%" PRIu64
      ",\"truncated_chunks\":%" PRIu64 "}\n",
      summary->file_record_count, summary->session_begin_count,
      summary->chunk_count, summary->session_end_count,
      summary->unknown_record_count, summary->device_chunk_count,
      summary->queue_chunk_count, summary->executable_chunk_count,
      summary->executable_record_count, summary->executable_export_chunk_count,
      summary->executable_export_record_count,
      summary->command_buffer_chunk_count, summary->command_buffer_record_count,
      summary->clock_correlation_chunk_count,
      summary->dispatch_event_chunk_count, summary->unknown_chunk_count,
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

static iree_status_t iree_profile_summary_file(
    iree_string_view_t path, iree_string_view_t format, FILE* file,
    iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &record_offset);
  }

  iree_profile_summary_t summary;
  iree_profile_summary_initialize(host_allocator, &summary);
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_summary_process_record(&summary, &record);
      record_offset = next_record_offset;
    }
  }

  if (iree_status_is_ok(status)) {
    if (is_text) {
      iree_profile_print_summary_text(&summary, file);
    } else {
      iree_profile_print_summary_jsonl(&summary, file);
    }
  }

  iree_profile_summary_deinitialize(&summary);
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t iree_profile_cat_file(iree_string_view_t path,
                                           iree_string_view_t format,
                                           FILE* file,
                                           iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &record_offset);
  }
  if (iree_status_is_ok(status)) {
    if (is_text) {
      iree_profile_dump_header_text(&header, file);
    } else {
      iree_profile_dump_header_jsonl(&header, file);
    }
  }

  iree_host_size_t record_index = 0;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      if (is_text) {
        iree_profile_dump_record_text(record_index, &record, file);
      } else {
        iree_profile_dump_record_jsonl(record_index, &record, file);
      }
      record_offset = next_record_offset;
      ++record_index;
    }
  }

  iree_io_file_contents_free(file_contents);
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  iree_flags_set_usage(
      "iree-profile",
      "Inspects IREE HAL profile bundles.\n"
      "\n"
      "Usage:\n"
      "  iree-profile cat [--format=text|jsonl] <file.ireeprof>\n"
      "  iree-profile summary [--format=text|jsonl] <file.ireeprof>\n"
      "  iree-profile [--format=text|jsonl] <file.ireeprof>\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_string_view_t command = IREE_SV("cat");
  iree_string_view_t path = iree_string_view_empty();
  if (argc == 2) {
    path = iree_make_cstring_view(argv[1]);
  } else if (argc == 3) {
    command = iree_make_cstring_view(argv[1]);
    path = iree_make_cstring_view(argv[2]);
  }

  iree_status_t status = iree_ok_status();
  if (argc != 2 && argc != 3) {
    fprintf(stderr,
            "Error: expected profile bundle path.\n"
            "Usage: iree-profile cat [--format=text|jsonl] <file.ireeprof>\n"
            "       iree-profile summary [--format=text|jsonl] "
            "<file.ireeprof>\n");
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected profile bundle path");
  } else if (iree_string_view_is_empty(path)) {
    fprintf(stderr,
            "Error: missing profile bundle path.\n"
            "Usage: iree-profile cat [--format=text|jsonl] <file.ireeprof>\n"
            "       iree-profile summary [--format=text|jsonl] "
            "<file.ireeprof>\n");
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing profile bundle path");
  } else if (!iree_string_view_equal(command, IREE_SV("cat")) &&
             !iree_string_view_equal(command, IREE_SV("summary"))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported iree-profile command '%.*s'",
                              (int)command.size, command.data);
  }

  if (iree_status_is_ok(status)) {
    if (iree_string_view_equal(command, IREE_SV("summary"))) {
      status = iree_profile_summary_file(
          path, iree_make_cstring_view(FLAG_format), stdout, host_allocator);
    } else {
      status = iree_profile_cat_file(path, iree_make_cstring_view(FLAG_format),
                                     stdout, host_allocator);
    }
  }

  fflush(stdout);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
