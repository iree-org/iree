// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/statistics.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/hal/api.h"
#include "iree/hal/utils/statistics_sink.h"
#include "iree/tooling/profile/common.h"
#include "iree/tooling/profile/reader.h"

typedef struct iree_profile_statistics_context_t {
  // Aggregate sink receiving CHUNK records from the profile file.
  iree_hal_profile_statistics_sink_t* statistics_sink;
} iree_profile_statistics_context_t;

typedef struct iree_profile_statistics_jsonl_context_t {
  // Aggregate sink owning the row currently being printed.
  const iree_hal_profile_statistics_sink_t* statistics_sink;
  // Output stream receiving JSONL rows.
  FILE* file;
} iree_profile_statistics_jsonl_context_t;

static const char* iree_profile_statistics_row_type_name(
    iree_hal_profile_statistics_row_type_t row_type) {
  switch (row_type) {
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_EXPORT:
      return "dispatch_export";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_COMMAND_BUFFER:
      return "dispatch_command_buffer";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_COMMAND_OPERATION:
      return "dispatch_command_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_DEVICE_OPERATION:
      return "queue_device_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_HOST_OPERATION:
      return "queue_host_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_EXPORT:
      return "host_execution_export";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_COMMAND_BUFFER:
      return "host_execution_command_buffer";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_COMMAND_OPERATION:
      return "host_execution_command_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_QUEUE_OPERATION:
      return "host_execution_queue_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_MEMORY_LIFECYCLE:
      return "memory_lifecycle";
    default:
      return "unknown";
  }
}

static const char* iree_profile_statistics_time_domain_name(
    iree_hal_profile_statistics_time_domain_t time_domain) {
  switch (time_domain) {
    case IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK:
      return "device_tick";
    case IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_IREE_HOST_TIME_NS:
      return "iree_host_time_ns";
    default:
      return "none";
  }
}

static const char* iree_profile_statistics_duration_unit_name(
    iree_hal_profile_statistics_time_domain_t time_domain) {
  switch (time_domain) {
    case IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK:
      return "device_tick";
    case IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_IREE_HOST_TIME_NS:
      return "ns";
    default:
      return "none";
  }
}

static const char* iree_profile_statistics_queue_event_type_name(
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

static const char* iree_profile_statistics_memory_event_type_name(
    iree_hal_profile_memory_event_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
      return "slab_acquire";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      return "slab_release";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      return "pool_reserve";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
      return "pool_materialize";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
      return "pool_release";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      return "pool_wait";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      return "queue_alloca";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      return "queue_dealloca";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
      return "buffer_allocate";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      return "buffer_free";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT:
      return "buffer_import";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT:
      return "buffer_unimport";
    default:
      return "unknown";
  }
}

static const char* iree_profile_statistics_event_type_name(
    const iree_hal_profile_statistics_row_t* row) {
  switch (row->row_type) {
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_DEVICE_OPERATION:
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_HOST_OPERATION:
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_QUEUE_OPERATION:
      return iree_profile_statistics_queue_event_type_name(
          (iree_hal_profile_queue_event_type_t)row->event_type);
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_MEMORY_LIFECYCLE:
      return iree_profile_statistics_memory_event_type_name(
          (iree_hal_profile_memory_event_type_t)row->event_type);
    default:
      return "";
  }
}

static iree_hal_profile_chunk_metadata_t
iree_profile_statistics_metadata_from_record(
    const iree_hal_profile_file_record_t* record) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = record->content_type;
  metadata.name = record->name;
  metadata.session_id = record->header.session_id;
  metadata.stream_id = record->header.stream_id;
  metadata.event_id = record->header.event_id;
  metadata.executable_id = record->header.executable_id;
  metadata.command_buffer_id = record->header.command_buffer_id;
  metadata.physical_device_ordinal = record->header.physical_device_ordinal;
  metadata.queue_ordinal = record->header.queue_ordinal;
  metadata.flags = record->header.chunk_flags;
  metadata.dropped_record_count = record->header.dropped_record_count;
  return metadata;
}

static iree_status_t iree_profile_statistics_process_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  // Aggregate the full bundle. The statistics sink resets on begin_session,
  // so replay only chunk records instead of forwarding session markers.
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }

  iree_profile_statistics_context_t* context =
      (iree_profile_statistics_context_t*)user_data;
  const iree_hal_profile_chunk_metadata_t metadata =
      iree_profile_statistics_metadata_from_record(record);
  const iree_const_byte_span_t* iovecs =
      record->payload.data_length != 0 ? &record->payload : NULL;
  return iree_hal_profile_sink_write(
      iree_hal_profile_statistics_sink_base(context->statistics_sink),
      &metadata, iovecs ? 1 : 0, iovecs);
}

static iree_status_t iree_profile_statistics_load_file(
    iree_string_view_t path,
    iree_hal_profile_statistics_sink_t* statistics_sink,
    iree_allocator_t host_allocator) {
  iree_profile_file_t profile_file;
  IREE_RETURN_IF_ERROR(
      iree_profile_file_open(path, host_allocator, &profile_file));
  iree_profile_statistics_context_t context = {
      .statistics_sink = statistics_sink,
  };
  const iree_profile_file_record_callback_t callback = {
      .fn = iree_profile_statistics_process_record,
      .user_data = &context,
  };
  iree_status_t status =
      iree_profile_file_for_each_record(&profile_file, callback);
  iree_profile_file_close(&profile_file);
  return status;
}

static void iree_profile_statistics_fprint_duration_ns_or_null(
    FILE* file, const iree_hal_profile_statistics_sink_t* statistics_sink,
    const iree_hal_profile_statistics_row_t* row, const char* field_name,
    uint64_t duration) {
  fprintf(file, ",\"%s\":", field_name);
  if (!iree_all_bits_set(row->flags,
                         IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING)) {
    fprintf(file, "null");
    return;
  }
  uint64_t duration_ns = 0;
  if (iree_hal_profile_statistics_sink_scale_duration_to_ns(
          statistics_sink, row, duration, &duration_ns)) {
    fprintf(file, "%" PRIu64, duration_ns);
  } else {
    fprintf(file, "null");
  }
}

static iree_status_t iree_profile_statistics_print_jsonl_row(
    void* user_data, const iree_hal_profile_statistics_row_t* row) {
  const iree_profile_statistics_jsonl_context_t* context =
      (const iree_profile_statistics_jsonl_context_t*)user_data;
  FILE* file = context->file;
  const bool has_timing = iree_all_bits_set(
      row->flags, IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING);
  const bool has_payload_bytes = iree_all_bits_set(
      row->flags, IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_PAYLOAD_BYTES);
  const bool has_operation_count = iree_all_bits_set(
      row->flags, IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_OPERATION_COUNT);
  const bool has_tile_totals = iree_all_bits_set(
      row->flags, IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TILE_TOTALS);

  fprintf(file, "{\"type\":\"statistics_row\",\"row_type\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_statistics_row_type_name(row->row_type)));
  fprintf(file, ",\"row_type_value\":%u", row->row_type);
  fprintf(file, ",\"time_domain\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_statistics_time_domain_name(row->time_domain)));
  fprintf(file, ",\"time_domain_value\":%u", row->time_domain);
  fprintf(file, ",\"duration_unit\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_statistics_duration_unit_name(row->time_domain)));
  fprintf(file,
          ",\"flags\":%u,\"has_timing\":%s,\"has_payload_bytes\":%s"
          ",\"has_operation_count\":%s,\"has_tile_totals\":%s",
          row->flags, has_timing ? "true" : "false",
          has_payload_bytes ? "true" : "false",
          has_operation_count ? "true" : "false",
          has_tile_totals ? "true" : "false");
  fprintf(file,
          ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
          ",\"event_type\":%u,\"event_name\":",
          row->physical_device_ordinal, row->queue_ordinal, row->event_type);
  const char* event_type_name = iree_profile_statistics_event_type_name(row);
  if (event_type_name[0] != '\0') {
    iree_profile_fprint_json_string(file,
                                    iree_make_cstring_view(event_type_name));
  } else {
    fprintf(file, "null");
  }
  fprintf(file,
          ",\"executable_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
          ",\"export_ordinal\":%u,\"command_index\":%u",
          row->executable_id, row->command_buffer_id, row->export_ordinal,
          row->command_index);

  iree_string_view_t export_name = iree_string_view_empty();
  const bool has_export_name =
      iree_hal_profile_statistics_sink_find_export_name(
          context->statistics_sink, row->executable_id, row->export_ordinal,
          &export_name);
  fprintf(file, ",\"export_name\":");
  if (has_export_name) {
    iree_profile_fprint_json_string(file, export_name);
  } else {
    fprintf(file, "null");
  }

  fprintf(file,
          ",\"sample_count\":%" PRIu64 ",\"invalid_sample_count\":%" PRIu64
          ",\"operation_count\":%" PRIu64 ",\"payload_bytes\":%" PRIu64
          ",\"tile_count\":%" PRIu64 ",\"tile_duration_sum_ns\":%" PRIu64,
          row->sample_count, row->invalid_sample_count, row->operation_count,
          row->payload_bytes, row->tile_count, row->tile_duration_sum_ns);
  fprintf(file,
          ",\"first_start_time\":%" PRIu64 ",\"last_end_time\":%" PRIu64
          ",\"total_duration\":%" PRIu64 ",\"minimum_duration\":%" PRIu64
          ",\"maximum_duration\":%" PRIu64,
          has_timing ? row->first_start_time : 0,
          has_timing ? row->last_end_time : 0, row->total_duration,
          has_timing ? row->minimum_duration : 0,
          has_timing ? row->maximum_duration : 0);
  iree_profile_statistics_fprint_duration_ns_or_null(
      file, context->statistics_sink, row, "total_duration_ns",
      row->total_duration);
  iree_profile_statistics_fprint_duration_ns_or_null(
      file, context->statistics_sink, row, "minimum_duration_ns",
      row->minimum_duration);
  iree_profile_statistics_fprint_duration_ns_or_null(
      file, context->statistics_sink, row, "maximum_duration_ns",
      row->maximum_duration);
  fprintf(file, "}\n");
  return iree_ok_status();
}

static iree_status_t iree_profile_statistics_print_jsonl(
    FILE* file, const iree_hal_profile_statistics_sink_t* statistics_sink) {
  fprintf(
      file,
      "{\"type\":\"statistics_summary\",\"row_count\":%" PRIhsz
      ",\"dropped_record_count\":%" PRIu64 "}\n",
      iree_hal_profile_statistics_sink_row_count(statistics_sink),
      iree_hal_profile_statistics_sink_dropped_record_count(statistics_sink));
  const iree_profile_statistics_jsonl_context_t context = {
      .statistics_sink = statistics_sink,
      .file = file,
  };
  const iree_hal_profile_statistics_row_callback_t callback = {
      .fn = iree_profile_statistics_print_jsonl_row,
      .user_data = (void*)&context,
  };
  return iree_hal_profile_statistics_sink_for_each_row(statistics_sink,
                                                       callback);
}

iree_status_t iree_profile_statistics_file(iree_string_view_t path,
                                           iree_string_view_t format,
                                           FILE* file,
                                           iree_allocator_t host_allocator) {
  iree_hal_profile_statistics_sink_t* statistics_sink = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_profile_statistics_sink_create(
      host_allocator, &statistics_sink));

  iree_status_t status =
      iree_profile_statistics_load_file(path, statistics_sink, host_allocator);
  if (iree_status_is_ok(status)) {
    if (iree_string_view_equal(format, IREE_SV("text"))) {
      status = iree_hal_profile_statistics_sink_fprint(file, statistics_sink);
    } else if (iree_string_view_equal(format, IREE_SV("jsonl"))) {
      status = iree_profile_statistics_print_jsonl(file, statistics_sink);
    } else {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported statistics format '%.*s'",
                                (int)format.size, format.data);
    }
  }

  iree_hal_profile_statistics_sink_release(statistics_sink);
  return status;
}
