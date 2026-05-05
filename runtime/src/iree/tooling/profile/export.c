// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/export.h"

#include <errno.h>
#include <string.h>

#include "iree/hal/api.h"
#include "iree/tooling/profile/memory.h"
#include "iree/tooling/profile/model.h"
#include "iree/tooling/profile/reader.h"

#define IREE_PROFILE_EXPORT_SCHEMA_VERSION 15

static void iree_profile_export_print_prefix(FILE* file,
                                             const char* record_type,
                                             iree_host_size_t record_index) {
  fprintf(file,
          "{\"schema_version\":%d,\"record_type\":\"%s\""
          ",\"source_record_index\":%" PRIhsz,
          IREE_PROFILE_EXPORT_SCHEMA_VERSION, record_type, record_index);
}

static void iree_profile_export_fprint_hex_bytes(FILE* file,
                                                 const uint8_t* data,
                                                 iree_host_size_t length) {
  static const char kHexDigits[] = "0123456789abcdef";
  for (iree_host_size_t i = 0; i < length; ++i) {
    fputc(kHexDigits[data[i] >> 4], file);
    fputc(kHexDigits[data[i] & 0x0F], file);
  }
}

static void iree_profile_export_fprint_nullable_hash(FILE* file, bool has_hash,
                                                     const uint64_t hash[2]) {
  if (has_hash) {
    fputc('"', file);
    iree_profile_fprint_hash_hex(file, hash);
    fputc('"', file);
  } else {
    fprintf(file, "null");
  }
}

static const char* iree_profile_export_device_class_name(
    iree_hal_profile_device_class_t device_class) {
  switch (device_class) {
    case IREE_HAL_PROFILE_DEVICE_CLASS_NONE:
      return "none";
    case IREE_HAL_PROFILE_DEVICE_CLASS_CPU:
      return "cpu";
    case IREE_HAL_PROFILE_DEVICE_CLASS_GPU:
      return "gpu";
    case IREE_HAL_PROFILE_DEVICE_CLASS_NPU:
      return "npu";
    case IREE_HAL_PROFILE_DEVICE_CLASS_OTHER:
      return "other";
    default:
      return "unknown";
  }
}

static const char* iree_profile_export_counter_sample_scope_name(
    iree_hal_profile_counter_sample_scope_t scope) {
  switch (scope) {
    case IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_NONE:
      return "none";
    case IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DISPATCH:
      return "dispatch";
    case IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_COMMAND_OPERATION:
      return "command_operation";
    case IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DEVICE_TIME_RANGE:
      return "device_time_range";
    default:
      return "unknown";
  }
}

static void iree_profile_export_print_device_metric_value(
    FILE* file, iree_hal_profile_metric_value_kind_t value_kind,
    uint64_t value_bits) {
  switch (value_kind) {
    case IREE_HAL_PROFILE_METRIC_VALUE_KIND_I64:
      fprintf(file, "%" PRId64, (int64_t)value_bits);
      break;
    case IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64:
      fprintf(file, "%" PRIu64, value_bits);
      break;
    case IREE_HAL_PROFILE_METRIC_VALUE_KIND_F64: {
      double value = 0.0;
      memcpy(&value, &value_bits, sizeof(value));
      fprintf(file, "%.17g", value);
      break;
    }
    default:
      fprintf(file, "null");
      break;
  }
}

static bool iree_profile_export_try_get_driver_host_cpu_clock_fit(
    const iree_profile_model_t* model, uint32_t physical_device_ordinal,
    iree_profile_model_clock_fit_t* out_fit) {
  const iree_profile_model_device_t* device =
      iree_profile_model_find_device(model, physical_device_ordinal);
  return iree_profile_model_device_try_fit_clock_exact(
      device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS,
      out_fit);
}

typedef struct iree_profile_export_device_time_t {
  // True when the source tick pair is non-zero and ordered.
  bool is_valid;
  // Raw device tick delta when the source tick pair is valid.
  uint64_t duration_ticks;
  // True when the source ticks were mapped into the driver host CPU time
  // domain.
  bool has_derived_time;
  // First clock correlation sample used by the fitted mapping.
  uint64_t clock_fit_first_sample_id;
  // Last clock correlation sample used by the fitted mapping.
  uint64_t clock_fit_last_sample_id;
  // Source start tick mapped to driver host CPU timestamp nanoseconds.
  int64_t start_driver_host_cpu_time_ns;
  // Source end tick mapped to driver host CPU timestamp nanoseconds.
  int64_t end_driver_host_cpu_time_ns;
  // Derived duration in nanoseconds when the clock mapping is available.
  int64_t duration_ns;
} iree_profile_export_device_time_t;

static iree_profile_export_device_time_t
iree_profile_export_calculate_device_time(const iree_profile_model_t* model,
                                          uint32_t physical_device_ordinal,
                                          uint64_t start_tick,
                                          uint64_t end_tick) {
  iree_profile_export_device_time_t device_time;
  memset(&device_time, 0, sizeof(device_time));
  device_time.is_valid =
      start_tick != 0 && end_tick != 0 && end_tick >= start_tick;
  if (!device_time.is_valid) return device_time;

  device_time.duration_ticks = end_tick - start_tick;
  iree_profile_model_clock_fit_t clock_fit;
  if (!iree_profile_export_try_get_driver_host_cpu_clock_fit(
          model, physical_device_ordinal, &clock_fit)) {
    return device_time;
  }
  int64_t start_driver_host_cpu_time_ns = 0;
  int64_t end_driver_host_cpu_time_ns = 0;
  if (!iree_profile_model_clock_fit_map_tick(&clock_fit, start_tick,
                                             &start_driver_host_cpu_time_ns) ||
      !iree_profile_model_clock_fit_map_tick(&clock_fit, end_tick,
                                             &end_driver_host_cpu_time_ns) ||
      end_driver_host_cpu_time_ns < start_driver_host_cpu_time_ns) {
    return device_time;
  }

  device_time.has_derived_time = true;
  device_time.clock_fit_first_sample_id = clock_fit.first_sample_id;
  device_time.clock_fit_last_sample_id = clock_fit.last_sample_id;
  device_time.start_driver_host_cpu_time_ns = start_driver_host_cpu_time_ns;
  device_time.end_driver_host_cpu_time_ns = end_driver_host_cpu_time_ns;
  device_time.duration_ns =
      end_driver_host_cpu_time_ns - start_driver_host_cpu_time_ns;
  return device_time;
}

static void iree_profile_export_print_session_record(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  const char* event_name =
      record->header.record_type ==
              IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN
          ? "begin"
          : "end";
  iree_profile_export_print_prefix(file, "session", record_index);
  fprintf(file,
          ",\"event\":\"%s\",\"session_id\":%" PRIu64 ",\"stream_id\":%" PRIu64
          ",\"event_id\":%" PRIu64 ",\"session_status_code\":%u",
          event_name, record->header.session_id, record->header.stream_id,
          record->header.event_id, record->header.session_status_code);
  fprintf(file, ",\"session_status\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(iree_profile_status_code_name(
                record->header.session_status_code)));
  fputs("}\n", file);
}

static void iree_profile_export_print_diagnostic(
    iree_host_size_t record_index, const char* severity, const char* category,
    iree_string_view_t content_type, const char* message, FILE* file) {
  iree_profile_export_print_prefix(file, "diagnostic", record_index);
  fprintf(file, ",\"severity\":");
  iree_profile_fprint_json_string(file, iree_make_cstring_view(severity));
  fprintf(file, ",\"category\":");
  iree_profile_fprint_json_string(file, iree_make_cstring_view(category));
  fprintf(file, ",\"content_type\":");
  iree_profile_fprint_json_string(file, content_type);
  fprintf(file, ",\"message\":");
  iree_profile_fprint_json_string(file, iree_make_cstring_view(message));
  fputs("}\n", file);
}

static void iree_profile_export_print_truncated_chunk_diagnostic(
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_profile_export_print_prefix(file, "diagnostic", record_index);
  fprintf(file, ",\"severity\":\"error\",\"category\":\"truncated_chunk\"");
  fprintf(file, ",\"content_type\":");
  iree_profile_fprint_json_string(file, record->content_type);
  fprintf(file,
          ",\"message\":\"chunk was marked truncated by the producer\""
          ",\"dropped_records\":%" PRIu64 "}\n",
          record->header.dropped_record_count);
}

static iree_status_t iree_profile_export_process_device_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
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
    const bool has_uuid = iree_all_bits_set(
        device_record.flags, IREE_HAL_PROFILE_DEVICE_FLAG_PHYSICAL_DEVICE_UUID);
    iree_profile_export_print_prefix(file, "device", record_index);
    fprintf(file,
            ",\"physical_device_ordinal\":%u,\"flags\":%u"
            ",\"queue_count\":%u,\"physical_device_uuid_present\":%s"
            ",\"physical_device_uuid\":",
            device_record.physical_device_ordinal, device_record.flags,
            device_record.queue_count, has_uuid ? "true" : "false");
    if (has_uuid) {
      fputc('"', file);
      iree_profile_export_fprint_hex_bytes(
          file, device_record.physical_device_uuid,
          sizeof(device_record.physical_device_uuid));
      fputc('"', file);
    } else {
      fprintf(file, "null");
    }
    fputs("}\n", file);
  }
  return status;
}

static iree_status_t iree_profile_export_process_queue_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
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
    iree_profile_export_print_prefix(file, "queue", record_index);
    fprintf(file,
            ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
            ",\"stream_id\":%" PRIu64 "}\n",
            queue_record.physical_device_ordinal, queue_record.queue_ordinal,
            queue_record.stream_id);
  }
  return status;
}

static iree_status_t iree_profile_export_process_executable_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
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

    iree_hal_profile_executable_record_t executable_record;
    memcpy(&executable_record, typed_record.contents.data,
           sizeof(executable_record));
    const bool has_code_object_hash =
        iree_all_bits_set(executable_record.flags,
                          IREE_HAL_PROFILE_EXECUTABLE_FLAG_CODE_OBJECT_HASH);
    iree_profile_export_print_prefix(file, "executable", record_index);
    fprintf(file,
            ",\"executable_id\":%" PRIu64
            ",\"flags\":%u"
            ",\"export_count\":%u,\"code_object_hash_present\":%s"
            ",\"code_object_hash\":",
            executable_record.executable_id, executable_record.flags,
            executable_record.export_count,
            has_code_object_hash ? "true" : "false");
    iree_profile_export_fprint_nullable_hash(
        file, has_code_object_hash, executable_record.code_object_hash);
    fputs("}\n", file);
  }
  return status;
}

static iree_status_t iree_profile_export_process_executable_export_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
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
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "executable export name length is inconsistent");
    }
    if (iree_status_is_ok(status)) {
      const bool has_pipeline_hash = iree_all_bits_set(
          export_record.flags,
          IREE_HAL_PROFILE_EXECUTABLE_EXPORT_FLAG_PIPELINE_HASH);
      iree_string_view_t name =
          iree_make_string_view((const char*)typed_record.inline_payload.data,
                                typed_record.inline_payload.data_length);
      iree_profile_export_print_prefix(file, "executable_export", record_index);
      fprintf(file,
              ",\"executable_id\":%" PRIu64
              ",\"export_ordinal\":%u"
              ",\"flags\":%u,\"name\":",
              export_record.executable_id, export_record.export_ordinal,
              export_record.flags);
      iree_profile_fprint_json_string(file, name);
      fprintf(file,
              ",\"constant_count\":%u,\"binding_count\":%u"
              ",\"parameter_count\":%u,\"workgroup_size\":[%u,%u,%u]"
              ",\"pipeline_hash_present\":%s,\"pipeline_hash\":",
              export_record.constant_count, export_record.binding_count,
              export_record.parameter_count, export_record.workgroup_size[0],
              export_record.workgroup_size[1], export_record.workgroup_size[2],
              has_pipeline_hash ? "true" : "false");
      iree_profile_export_fprint_nullable_hash(file, has_pipeline_hash,
                                               export_record.pipeline_hash);
      fputs("}\n", file);
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_command_buffer_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
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

    iree_hal_profile_command_buffer_record_t command_buffer_record;
    memcpy(&command_buffer_record, typed_record.contents.data,
           sizeof(command_buffer_record));
    iree_profile_export_print_prefix(file, "command_buffer", record_index);
    fprintf(
        file,
        ",\"command_buffer_id\":%" PRIu64
        ",\"flags\":%u"
        ",\"physical_device_ordinal\":%u,\"mode\":%" PRIu64
        ",\"command_categories\":%" PRIu64 ",\"queue_affinity\":%" PRIu64 "}\n",
        command_buffer_record.command_buffer_id, command_buffer_record.flags,
        command_buffer_record.physical_device_ordinal,
        command_buffer_record.mode, command_buffer_record.command_categories,
        command_buffer_record.queue_affinity);
  }
  return status;
}

static iree_status_t iree_profile_export_process_command_operation_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
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

    iree_hal_profile_command_operation_record_t operation_record;
    memcpy(&operation_record, typed_record.contents.data,
           sizeof(operation_record));
    const char* operation_name =
        iree_profile_command_operation_type_name(operation_record.type);
    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_model_resolve_command_operation_key(
        model, &operation_record, numeric_buffer, sizeof(numeric_buffer), &key);
    if (iree_status_is_ok(status)) {
      const bool has_block_structure =
          iree_hal_profile_command_operation_has_block_structure(
              &operation_record);
      iree_profile_export_print_prefix(file, "command_operation", record_index);
      fprintf(file,
              ",\"command_buffer_id\":%" PRIu64
              ",\"command_index\":%u"
              ",\"op\":",
              operation_record.command_buffer_id,
              operation_record.command_index);
      iree_profile_fprint_json_string(file,
                                      iree_make_cstring_view(operation_name));
      fprintf(file, ",\"key\":");
      iree_profile_fprint_json_string(file, key);
      fprintf(file,
              ",\"flags\":%u,\"block_structure\":%s"
              ",\"block_ordinal\":%u"
              ",\"block_command_ordinal\":%u"
              ",\"executable_id\":%" PRIu64
              ",\"export_ordinal\":%u"
              ",\"binding_count\":%u,\"workgroup_count\":[%u,%u,%u]"
              ",\"workgroup_size\":[%u,%u,%u]"
              ",\"source_ordinal\":%u,\"target_ordinal\":%u"
              ",\"source_offset\":%" PRIu64 ",\"target_offset\":%" PRIu64
              ",\"length\":%" PRIu64
              ",\"target_block_ordinal\":%u"
              ",\"alternate_block_ordinal\":%u}\n",
              operation_record.flags, has_block_structure ? "true" : "false",
              operation_record.block_ordinal,
              operation_record.block_command_ordinal,
              operation_record.executable_id, operation_record.export_ordinal,
              operation_record.binding_count,
              operation_record.workgroup_count[0],
              operation_record.workgroup_count[1],
              operation_record.workgroup_count[2],
              operation_record.workgroup_size[0],
              operation_record.workgroup_size[1],
              operation_record.workgroup_size[2],
              operation_record.source_ordinal, operation_record.target_ordinal,
              operation_record.source_offset, operation_record.target_offset,
              operation_record.length, operation_record.target_block_ordinal,
              operation_record.alternate_block_ordinal);
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_clock_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
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
    iree_profile_export_print_prefix(file, "clock_correlation", record_index);
    fprintf(
        file,
        ",\"sample_id\":%" PRIu64
        ",\"flags\":%u"
        ",\"physical_device_ordinal\":%u"
        ",\"device_tick_domain\":\"device_tick\""
        ",\"device_tick\":%" PRIu64
        ",\"host_cpu_timestamp_domain\":\"driver_host_cpu_timestamp_ns\""
        ",\"host_cpu_timestamp_ns\":%" PRIu64
        ",\"host_system_timestamp_domain\":\"driver_host_system_timestamp\""
        ",\"host_system_timestamp\":%" PRIu64
        ",\"host_system_frequency_hz\":%" PRIu64
        ",\"host_time_domain\":\"iree_host_time_ns\""
        ",\"host_time_begin_ns\":%" PRId64 ",\"host_time_end_ns\":%" PRId64
        ",\"host_time_uncertainty_ns\":%" PRId64 "}\n",
        clock_record.sample_id, clock_record.flags,
        clock_record.physical_device_ordinal, clock_record.device_tick,
        clock_record.host_cpu_timestamp_ns, clock_record.host_system_timestamp,
        clock_record.host_system_frequency_hz, clock_record.host_time_begin_ns,
        clock_record.host_time_end_ns,
        clock_record.host_time_end_ns >= clock_record.host_time_begin_ns
            ? clock_record.host_time_end_ns - clock_record.host_time_begin_ns
            : 0);
  }
  return status;
}

static iree_status_t iree_profile_export_process_dispatch_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
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
    const iree_profile_export_device_time_t device_time =
        iree_profile_export_calculate_device_time(
            model, record->header.physical_device_ordinal,
            dispatch_record.start_tick, dispatch_record.end_tick);
    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_model_resolve_dispatch_key(
        model, record->header.physical_device_ordinal,
        dispatch_record.executable_id, dispatch_record.export_ordinal,
        numeric_buffer, sizeof(numeric_buffer), &key);
    if (iree_status_is_ok(status)) {
      iree_profile_export_print_prefix(file, "dispatch_event", record_index);
      fprintf(file,
              ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
              ",\"stream_id\":%" PRIu64 ",\"event_id\":%" PRIu64
              ",\"submission_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
              ",\"command_index\":%u,\"executable_id\":%" PRIu64
              ",\"export_ordinal\":%u,\"key\":",
              record->header.physical_device_ordinal,
              record->header.queue_ordinal, record->header.stream_id,
              dispatch_record.event_id, dispatch_record.submission_id,
              dispatch_record.command_buffer_id, dispatch_record.command_index,
              dispatch_record.executable_id, dispatch_record.export_ordinal);
      iree_profile_fprint_json_string(file, key);
      fprintf(
          file,
          ",\"flags\":%u,\"workgroup_count\":[%u,%u,%u]"
          ",\"workgroup_size\":[%u,%u,%u]"
          ",\"device_tick_domain\":\"device_tick\""
          ",\"start_tick\":%" PRIu64 ",\"end_tick\":%" PRIu64
          ",\"duration_ticks\":%" PRIu64
          ",\"valid\":%s"
          ",\"derived_time_available\":%s"
          ",\"derived_time_domain\":\"driver_host_cpu_timestamp_ns\""
          ",\"derived_time_basis\":\"first_last_clock_correlation\""
          ",\"clock_fit_first_sample_id\":%" PRIu64
          ",\"clock_fit_last_sample_id\":%" PRIu64
          ",\"start_driver_host_cpu_time_ns\":%" PRId64
          ",\"end_driver_host_cpu_time_ns\":%" PRId64
          ",\"duration_time_domain\":\"device_tick_duration_ns\""
          ",\"duration_ns\":%" PRId64 "}\n",
          dispatch_record.flags, dispatch_record.workgroup_count[0],
          dispatch_record.workgroup_count[1],
          dispatch_record.workgroup_count[2], dispatch_record.workgroup_size[0],
          dispatch_record.workgroup_size[1], dispatch_record.workgroup_size[2],
          dispatch_record.start_tick, dispatch_record.end_tick,
          device_time.duration_ticks, device_time.is_valid ? "true" : "false",
          device_time.has_derived_time ? "true" : "false",
          device_time.clock_fit_first_sample_id,
          device_time.clock_fit_last_sample_id,
          device_time.start_driver_host_cpu_time_ns,
          device_time.end_driver_host_cpu_time_ns, device_time.duration_ns);
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_queue_device_event_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
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

    iree_hal_profile_queue_device_event_t queue_device_event;
    memcpy(&queue_device_event, typed_record.contents.data,
           sizeof(queue_device_event));
    const iree_profile_export_device_time_t device_time =
        iree_profile_export_calculate_device_time(
            model, queue_device_event.physical_device_ordinal,
            queue_device_event.start_tick, queue_device_event.end_tick);
    iree_profile_export_print_prefix(file, "queue_device_event", record_index);
    fprintf(file,
            ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
            ",\"stream_id\":%" PRIu64 ",\"event_id\":%" PRIu64
            ",\"submission_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
            ",\"allocation_id\":%" PRIu64 ",\"op\":",
            queue_device_event.physical_device_ordinal,
            queue_device_event.queue_ordinal, queue_device_event.stream_id,
            queue_device_event.event_id, queue_device_event.submission_id,
            queue_device_event.command_buffer_id,
            queue_device_event.allocation_id);
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(
                  iree_profile_queue_event_type_name(queue_device_event.type)));
    fprintf(file,
            ",\"type_value\":%u,\"flags\":%u"
            ",\"payload_length\":%" PRIu64
            ",\"operation_count\":%u"
            ",\"device_tick_domain\":\"device_tick\""
            ",\"start_tick\":%" PRIu64 ",\"end_tick\":%" PRIu64
            ",\"duration_ticks\":%" PRIu64
            ",\"valid\":%s"
            ",\"derived_time_available\":%s"
            ",\"derived_time_domain\":\"driver_host_cpu_timestamp_ns\""
            ",\"derived_time_basis\":\"first_last_clock_correlation\""
            ",\"clock_fit_first_sample_id\":%" PRIu64
            ",\"clock_fit_last_sample_id\":%" PRIu64
            ",\"start_driver_host_cpu_time_ns\":%" PRId64
            ",\"end_driver_host_cpu_time_ns\":%" PRId64
            ",\"duration_time_domain\":\"device_tick_duration_ns\""
            ",\"duration_ns\":%" PRId64 "}\n",
            queue_device_event.type, queue_device_event.flags,
            queue_device_event.payload_length,
            queue_device_event.operation_count, queue_device_event.start_tick,
            queue_device_event.end_tick, device_time.duration_ticks,
            device_time.is_valid ? "true" : "false",
            device_time.has_derived_time ? "true" : "false",
            device_time.clock_fit_first_sample_id,
            device_time.clock_fit_last_sample_id,
            device_time.start_driver_host_cpu_time_ns,
            device_time.end_driver_host_cpu_time_ns, device_time.duration_ns);
  }
  return status;
}

static iree_status_t iree_profile_export_process_host_execution_event_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
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

    iree_hal_profile_host_execution_event_t host_event;
    memcpy(&host_event, typed_record.contents.data, sizeof(host_event));
    const bool is_valid =
        host_event.start_host_time_ns >= 0 &&
        host_event.end_host_time_ns >= host_event.start_host_time_ns;
    const int64_t duration_ns =
        is_valid ? host_event.end_host_time_ns - host_event.start_host_time_ns
                 : 0;
    iree_string_view_t key = iree_string_view_empty();
    char numeric_buffer[128];
    const bool has_dispatch_key = host_event.executable_id != 0 &&
                                  host_event.export_ordinal != UINT32_MAX;
    if (has_dispatch_key) {
      status = iree_profile_model_resolve_dispatch_key(
          model, host_event.physical_device_ordinal, host_event.executable_id,
          host_event.export_ordinal, numeric_buffer, sizeof(numeric_buffer),
          &key);
    }
    if (iree_status_is_ok(status)) {
      iree_profile_export_print_prefix(file, "host_execution_event",
                                       record_index);
      fprintf(file,
              ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
              ",\"stream_id\":%" PRIu64 ",\"event_id\":%" PRIu64
              ",\"submission_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
              ",\"command_index\":%u,\"executable_id\":%" PRIu64
              ",\"export_ordinal\":%u,\"allocation_id\":%" PRIu64 ",\"op\":",
              host_event.physical_device_ordinal, host_event.queue_ordinal,
              host_event.stream_id, host_event.event_id,
              host_event.submission_id, host_event.command_buffer_id,
              host_event.command_index, host_event.executable_id,
              host_event.export_ordinal, host_event.allocation_id);
      iree_profile_fprint_json_string(
          file, iree_make_cstring_view(
                    iree_profile_queue_event_type_name(host_event.type)));
      fprintf(file, ",\"key\":");
      if (has_dispatch_key) {
        iree_profile_fprint_json_string(file, key);
      } else {
        fprintf(file, "null");
      }
      fprintf(file,
              ",\"type_value\":%u,\"flags\":%u,\"status_code\":%u"
              ",\"workgroup_count\":[%u,%u,%u]"
              ",\"workgroup_size\":[%u,%u,%u]"
              ",\"host_time_domain\":\"iree_host_time_ns\""
              ",\"start_host_time_ns\":%" PRId64
              ",\"end_host_time_ns\":%" PRId64 ",\"duration_ns\":%" PRId64
              ",\"valid\":%s"
              ",\"payload_length\":%" PRIu64 ",\"tile_count\":%" PRIu64
              ",\"tile_duration_sum_ns\":%" PRId64 ",\"operation_count\":%u}\n",
              host_event.type, host_event.flags, host_event.status_code,
              host_event.workgroup_count[0], host_event.workgroup_count[1],
              host_event.workgroup_count[2], host_event.workgroup_size[0],
              host_event.workgroup_size[1], host_event.workgroup_size[2],
              host_event.start_host_time_ns, host_event.end_host_time_ns,
              duration_ns, is_valid ? "true" : "false",
              host_event.payload_length, host_event.tile_count,
              host_event.tile_duration_sum_ns, host_event.operation_count);
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_command_region_event_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_command_region_event_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_command_region_event_t region_event;
    memcpy(&region_event, typed_record.contents.data, sizeof(region_event));
    const bool is_valid = region_event.command_region.start_host_time_ns >= 0 &&
                          region_event.command_region.end_host_time_ns >=
                              region_event.command_region.start_host_time_ns;
    const int64_t duration_ns =
        is_valid ? region_event.command_region.end_host_time_ns -
                       region_event.command_region.start_host_time_ns
                 : 0;
    iree_profile_export_print_prefix(file, "command_region_event",
                                     record_index);
    fprintf(
        file,
        ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
        ",\"stream_id\":%" PRIu64 ",\"event_id\":%" PRIu64
        ",\"submission_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
        ",\"block_sequence\":%u,\"region_epoch\":%u"
        ",\"region_index\":%d,\"next_region_index\":%d"
        ",\"flags\":%u,\"host_time_domain\":\"iree_host_time_ns\""
        ",\"start_host_time_ns\":%" PRId64 ",\"end_host_time_ns\":%" PRId64
        ",\"duration_ns\":%" PRId64
        ",\"valid\":%s"
        ",\"region_dispatch_count\":%u,\"region_tile_count\":%u"
        ",\"region_width_bucket\":%u"
        ",\"region_lookahead_width_bucket\":%u"
        ",\"next_region_tile_count\":%u"
        ",\"next_region_width_bucket\":%u"
        ",\"next_region_lookahead_width_bucket\":%u"
        ",\"worker_count\":%u,\"old_wake_budget\":%d"
        ",\"new_wake_budget\":%d,\"wake_delta\":%d"
        ",\"useful_drain_count\":%u"
        ",\"no_work_drain_count\":%u"
        ",\"first_useful_drain_start_host_time_ns\":%" PRId64
        ",\"last_useful_drain_end_host_time_ns\":%" PRId64
        ",\"tail_no_work\":{\"count\":%u"
        ",\"remaining_tiles\":{\"min\":%u,\"max\":%u"
        ",\"bucket_counts\":[%u,%u,%u,%u,%u,%u,%u,%u]}"
        ",\"first_start_host_time_ns\":%" PRId64
        ",\"last_end_host_time_ns\":%" PRId64
        ",\"time_sums\":{\"start_offset_ns\":%" PRId64
        ",\"drain_duration_ns\":%" PRId64
        "}"
        "}"
        ",\"retention_keep_active_count\":%u"
        ",\"retention_publish_keep_active_count\":%u"
        ",\"retention_keep_warm_count\":%u}\n",
        region_event.queue.physical_device_ordinal,
        region_event.queue.queue_ordinal, region_event.stream_id,
        region_event.event_id, region_event.submission_id,
        region_event.command_buffer_id,
        region_event.command_region.block_sequence,
        region_event.command_region.epoch, region_event.command_region.index,
        region_event.next_command_region.index, region_event.flags,
        region_event.command_region.start_host_time_ns,
        region_event.command_region.end_host_time_ns, duration_ns,
        is_valid ? "true" : "false", region_event.command_region.dispatch_count,
        region_event.command_region.tile_count,
        region_event.command_region.width_bucket,
        region_event.command_region.lookahead_width_bucket,
        region_event.next_command_region.tile_count,
        region_event.next_command_region.width_bucket,
        region_event.next_command_region.lookahead_width_bucket,
        region_event.scheduler.worker_count,
        region_event.scheduler.old_wake_budget,
        region_event.scheduler.new_wake_budget,
        region_event.scheduler.wake_delta,
        region_event.command_region.useful_drain_count,
        region_event.command_region.no_work_drain_count,
        region_event.command_region.first_useful_drain_start_host_time_ns,
        region_event.command_region.last_useful_drain_end_host_time_ns,
        region_event.command_region.tail_no_work.count,
        region_event.command_region.tail_no_work.remaining_tiles.min,
        region_event.command_region.tail_no_work.remaining_tiles.max,
        region_event.command_region.tail_no_work.remaining_tiles
            .bucket_counts[0],
        region_event.command_region.tail_no_work.remaining_tiles
            .bucket_counts[1],
        region_event.command_region.tail_no_work.remaining_tiles
            .bucket_counts[2],
        region_event.command_region.tail_no_work.remaining_tiles
            .bucket_counts[3],
        region_event.command_region.tail_no_work.remaining_tiles
            .bucket_counts[4],
        region_event.command_region.tail_no_work.remaining_tiles
            .bucket_counts[5],
        region_event.command_region.tail_no_work.remaining_tiles
            .bucket_counts[6],
        region_event.command_region.tail_no_work.remaining_tiles
            .bucket_counts[7],
        region_event.command_region.tail_no_work.first_start_host_time_ns,
        region_event.command_region.tail_no_work.last_end_host_time_ns,
        region_event.command_region.tail_no_work.time_sums.start_offset_ns,
        region_event.command_region.tail_no_work.time_sums.drain_duration_ns,
        region_event.retention.keep_active_count,
        region_event.retention.publish_keep_active_count,
        region_event.retention.keep_warm_count);
  }
  return status;
}

static iree_status_t iree_profile_export_process_queue_event_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
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

    iree_hal_profile_queue_event_t queue_event;
    memcpy(&queue_event, typed_record.contents.data, sizeof(queue_event));
    iree_profile_export_print_prefix(file, "queue_event", record_index);
    fprintf(file, ",\"event_id\":%" PRIu64 ",\"op\":", queue_event.event_id);
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(
                  iree_profile_queue_event_type_name(queue_event.type)));
    fprintf(file, ",\"type_value\":%u,\"flags\":%u,\"dependency_strategy\":",
            queue_event.type, queue_event.flags);
    iree_profile_fprint_json_string(
        file,
        iree_make_cstring_view(iree_profile_queue_dependency_strategy_name(
            queue_event.dependency_strategy)));
    fprintf(file,
            ",\"submission_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
            ",\"allocation_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
            ",\"stream_id\":%" PRIu64 ",\"host_time_ns\":%" PRId64
            ",\"ready_host_time_ns\":%" PRId64
            ",\"host_time_domain\":\"iree_host_time_ns\""
            ",\"wait_count\":%u,\"signal_count\":%u"
            ",\"barrier_count\":%u,\"operation_count\":%u"
            ",\"payload_length\":%" PRIu64 "}\n",
            queue_event.submission_id, queue_event.command_buffer_id,
            queue_event.allocation_id, queue_event.physical_device_ordinal,
            queue_event.queue_ordinal, queue_event.stream_id,
            queue_event.host_time_ns, queue_event.ready_host_time_ns,
            queue_event.wait_count, queue_event.signal_count,
            queue_event.barrier_count, queue_event.operation_count,
            queue_event.payload_length);
  }
  return status;
}

static iree_status_t iree_profile_export_process_memory_event_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_memory_event_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_memory_event_t memory_event;
    memcpy(&memory_event, typed_record.contents.data, sizeof(memory_event));
    iree_profile_export_print_prefix(file, "memory_event", record_index);
    fprintf(file,
            ",\"event_id\":%" PRIu64 ",\"event_type\":", memory_event.event_id);
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(
                  iree_profile_memory_event_type_name(memory_event.type)));
    fprintf(
        file,
        ",\"event_type_value\":%u,\"flags\":%u,\"result\":%u"
        ",\"host_time_ns\":%" PRId64 ",\"allocation_id\":%" PRIu64
        ",\"host_time_domain\":\"iree_host_time_ns\""
        ",\"pool_id\":%" PRIu64 ",\"backing_id\":%" PRIu64
        ",\"submission_id\":%" PRIu64
        ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
        ",\"frontier_entry_count\":%u,\"memory_type\":%" PRIu64
        ",\"buffer_usage\":%" PRIu64 ",\"offset\":%" PRIu64
        ",\"length\":%" PRIu64 ",\"alignment\":%" PRIu64
        ",\"externally_owned\":%s,\"pool_stats_available\":%s"
        ",\"pool_bytes_reserved\":%" PRIu64 ",\"pool_bytes_free\":%" PRIu64
        ",\"pool_bytes_committed\":%" PRIu64 ",\"pool_budget_limit\":%" PRIu64
        ",\"pool_reservation_count\":%u"
        ",\"pool_slab_count\":%u}\n",
        memory_event.type, memory_event.flags, memory_event.result,
        memory_event.host_time_ns, memory_event.allocation_id,
        memory_event.pool_id, memory_event.backing_id,
        memory_event.submission_id, memory_event.physical_device_ordinal,
        memory_event.queue_ordinal, memory_event.frontier_entry_count,
        memory_event.memory_type, memory_event.buffer_usage,
        memory_event.offset, memory_event.length, memory_event.alignment,
        iree_all_bits_set(memory_event.flags,
                          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_EXTERNALLY_OWNED)
            ? "true"
            : "false",
        iree_all_bits_set(memory_event.flags,
                          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_STATS)
            ? "true"
            : "false",
        memory_event.pool_bytes_reserved, memory_event.pool_bytes_free,
        memory_event.pool_bytes_committed, memory_event.pool_budget_limit,
        memory_event.pool_reservation_count, memory_event.pool_slab_count);
  }
  return status;
}

static iree_status_t iree_profile_export_process_event_relationship_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_event_relationship_record_t), &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_event_relationship_record_t relationship_record;
    memcpy(&relationship_record, typed_record.contents.data,
           sizeof(relationship_record));
    iree_profile_export_print_prefix(file, "event_relationship", record_index);
    fprintf(file, ",\"relationship_id\":%" PRIu64 ",\"kind\":",
            relationship_record.relationship_id);
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(iree_profile_event_relationship_type_name(
                  relationship_record.type)));
    fprintf(file,
            ",\"kind_value\":%u,\"source_type\":", relationship_record.type);
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(iree_profile_event_endpoint_type_name(
                  relationship_record.source_type)));
    fprintf(file, ",\"source_type_value\":%u,\"target_type\":",
            relationship_record.source_type);
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(iree_profile_event_endpoint_type_name(
                  relationship_record.target_type)));
    fprintf(
        file,
        ",\"target_type_value\":%u"
        ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
        ",\"stream_id\":%" PRIu64 ",\"source_id\":%" PRIu64
        ",\"source_secondary_id\":%" PRIu64 ",\"target_id\":%" PRIu64
        ",\"target_secondary_id\":%" PRIu64 "}\n",
        relationship_record.target_type,
        relationship_record.physical_device_ordinal,
        relationship_record.queue_ordinal, relationship_record.stream_id,
        relationship_record.source_id, relationship_record.source_secondary_id,
        relationship_record.target_id, relationship_record.target_secondary_id);
  }
  return status;
}

static iree_status_t iree_profile_export_process_counter_set_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
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
      iree_string_view_t name =
          iree_make_string_view((const char*)typed_record.inline_payload.data,
                                typed_record.inline_payload.data_length);
      iree_profile_export_print_prefix(file, "counter_set", record_index);
      fprintf(file,
              ",\"counter_set_id\":%" PRIu64
              ",\"physical_device_ordinal\":%u"
              ",\"flags\":%u,\"counter_count\":%u"
              ",\"sample_value_count\":%u,\"name\":",
              counter_set_record.counter_set_id,
              counter_set_record.physical_device_ordinal,
              counter_set_record.flags, counter_set_record.counter_count,
              counter_set_record.sample_value_count);
      iree_profile_fprint_json_string(file, name);
      fputs("}\n", file);
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_counter_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
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
      const char* string_base = (const char*)typed_record.inline_payload.data;
      iree_string_view_t block_name =
          iree_make_string_view(string_base, counter_record.block_name_length);
      iree_string_view_t name =
          iree_make_string_view(string_base + counter_record.block_name_length,
                                counter_record.name_length);
      iree_string_view_t description =
          iree_make_string_view(string_base + counter_record.block_name_length +
                                    counter_record.name_length,
                                counter_record.description_length);
      iree_profile_export_print_prefix(file, "counter", record_index);
      fprintf(file,
              ",\"counter_set_id\":%" PRIu64
              ",\"counter_ordinal\":%u"
              ",\"physical_device_ordinal\":%u,\"flags\":%u"
              ",\"unit\":%u,\"sample_value_offset\":%u"
              ",\"sample_value_count\":%u,\"block\":",
              counter_record.counter_set_id, counter_record.counter_ordinal,
              counter_record.physical_device_ordinal, counter_record.flags,
              counter_record.unit, counter_record.sample_value_offset,
              counter_record.sample_value_count);
      iree_profile_fprint_json_string(file, block_name);
      fprintf(file, ",\"name\":");
      iree_profile_fprint_json_string(file, name);
      fprintf(file, ",\"description\":");
      iree_profile_fprint_json_string(file, description);
      fputs("}\n", file);
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_counter_sample_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
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
      const uint64_t* values =
          (const uint64_t*)typed_record.inline_payload.data;
      const bool has_device_tick_range = iree_all_bits_set(
          sample_record.flags,
          IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE);
      iree_profile_export_device_time_t device_time = {0};
      if (has_device_tick_range) {
        device_time = iree_profile_export_calculate_device_time(
            model, sample_record.physical_device_ordinal,
            sample_record.start_tick, sample_record.end_tick);
      }
      iree_profile_export_print_prefix(file, "counter_sample", record_index);
      fprintf(file,
              ",\"sample_id\":%" PRIu64 ",\"counter_set_id\":%" PRIu64
              ",\"scope\":",
              sample_record.sample_id, sample_record.counter_set_id);
      iree_profile_fprint_json_string(
          file,
          iree_make_cstring_view(iree_profile_export_counter_sample_scope_name(
              sample_record.scope)));
      fprintf(
          file,
          ",\"scope_value\":%u"
          ",\"dispatch_event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64
          ",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u,\"executable_id\":%" PRIu64
          ",\"export_ordinal\":%u"
          ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
          ",\"stream_id\":%" PRIu64
          ",\"flags\":%u"
          ",\"device_tick_range_present\":%s"
          ",\"device_tick_domain\":\"device_tick\""
          ",\"start_tick\":%" PRIu64 ",\"end_tick\":%" PRIu64
          ",\"duration_ticks\":%" PRIu64
          ",\"device_tick_range_valid\":%s"
          ",\"derived_time_available\":%s"
          ",\"derived_time_domain\":\"driver_host_cpu_timestamp_ns\""
          ",\"derived_time_basis\":\"first_last_clock_correlation\""
          ",\"clock_fit_first_sample_id\":%" PRIu64
          ",\"clock_fit_last_sample_id\":%" PRIu64
          ",\"start_driver_host_cpu_time_ns\":%" PRId64
          ",\"end_driver_host_cpu_time_ns\":%" PRId64
          ",\"duration_time_domain\":\"device_tick_duration_ns\""
          ",\"duration_ns\":%" PRId64 ",\"values\":[",
          sample_record.scope, sample_record.dispatch_event_id,
          sample_record.submission_id, sample_record.command_buffer_id,
          sample_record.command_index, sample_record.executable_id,
          sample_record.export_ordinal, sample_record.physical_device_ordinal,
          sample_record.queue_ordinal, sample_record.stream_id,
          sample_record.flags, has_device_tick_range ? "true" : "false",
          sample_record.start_tick, sample_record.end_tick,
          device_time.duration_ticks, device_time.is_valid ? "true" : "false",
          device_time.has_derived_time ? "true" : "false",
          device_time.clock_fit_first_sample_id,
          device_time.clock_fit_last_sample_id,
          device_time.start_driver_host_cpu_time_ns,
          device_time.end_driver_host_cpu_time_ns, device_time.duration_ns);
      for (uint32_t i = 0; i < sample_record.sample_value_count; ++i) {
        if (i != 0) fputc(',', file);
        fprintf(file, "%" PRIu64, values[i]);
      }
      fputs("]}\n", file);
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_device_metric_source_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_device_metric_source_record_t),
      &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_device_metric_source_record_t source_record;
    memcpy(&source_record, typed_record.contents.data, sizeof(source_record));
    if ((iree_host_size_t)source_record.name_length !=
        typed_record.inline_payload.data_length) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "device metric source name length is "
                                "inconsistent with record length");
    }
    if (iree_status_is_ok(status)) {
      iree_string_view_t name =
          iree_make_string_view((const char*)typed_record.inline_payload.data,
                                typed_record.inline_payload.data_length);
      iree_profile_export_print_prefix(file, "device_metric_source",
                                       record_index);
      fprintf(file,
              ",\"source_id\":%" PRIu64
              ",\"physical_device_ordinal\":%u"
              ",\"device_class\":",
              source_record.source_id, source_record.physical_device_ordinal);
      iree_profile_fprint_json_string(
          file, iree_make_cstring_view(iree_profile_export_device_class_name(
                    source_record.device_class)));
      fprintf(file,
              ",\"device_class_value\":%u,\"flags\":%u"
              ",\"source_kind\":%u,\"source_revision\":%u"
              ",\"metric_count\":%u,\"name\":",
              source_record.device_class, source_record.flags,
              source_record.source_kind, source_record.source_revision,
              source_record.metric_count);
      iree_profile_fprint_json_string(file, name);
      fputs("}\n", file);
    }
  }
  return status;
}

static iree_status_t
iree_profile_export_process_device_metric_descriptor_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_device_metric_descriptor_record_t),
      &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_device_metric_descriptor_record_t descriptor_record;
    memcpy(&descriptor_record, typed_record.contents.data,
           sizeof(descriptor_record));
    iree_host_size_t trailing_length = 0;
    if (!iree_host_size_checked_add(descriptor_record.name_length,
                                    descriptor_record.description_length,
                                    &trailing_length) ||
        trailing_length != typed_record.inline_payload.data_length) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "device metric descriptor string lengths are inconsistent with "
          "record length");
    }
    if (iree_status_is_ok(status)) {
      const char* string_base = (const char*)typed_record.inline_payload.data;
      iree_string_view_t name =
          iree_make_string_view(string_base, descriptor_record.name_length);
      iree_string_view_t description =
          iree_make_string_view(string_base + descriptor_record.name_length,
                                descriptor_record.description_length);
      iree_profile_export_print_prefix(file, "device_metric_descriptor",
                                       record_index);
      fprintf(file,
              ",\"source_id\":%" PRIu64 ",\"metric_id\":%" PRIu64
              ",\"builtin\":%s,\"producer_specific\":%s"
              ",\"flags\":%u,\"unit\":",
              descriptor_record.source_id, descriptor_record.metric_id,
              iree_hal_profile_metric_id_is_builtin(descriptor_record.metric_id)
                  ? "true"
                  : "false",
              iree_hal_profile_metric_id_is_producer_specific(
                  descriptor_record.metric_id)
                  ? "true"
                  : "false",
              descriptor_record.flags);
      iree_profile_fprint_json_string(
          file, iree_hal_profile_metric_unit_string(descriptor_record.unit));
      fprintf(file,
              ",\"unit_value\":%u,\"value_kind\":", descriptor_record.unit);
      iree_profile_fprint_json_string(file,
                                      iree_hal_profile_metric_value_kind_string(
                                          descriptor_record.value_kind));
      fprintf(file, ",\"value_kind_value\":%u,\"semantic\":",
              descriptor_record.value_kind);
      iree_profile_fprint_json_string(
          file,
          iree_hal_profile_metric_semantic_string(descriptor_record.semantic));
      fprintf(file, ",\"semantic_value\":%u,\"plot_hint\":",
              descriptor_record.semantic);
      iree_profile_fprint_json_string(file,
                                      iree_hal_profile_metric_plot_hint_string(
                                          descriptor_record.plot_hint));
      fprintf(file,
              ",\"plot_hint_value\":%u,\"name\":", descriptor_record.plot_hint);
      iree_profile_fprint_json_string(file, name);
      fprintf(file, ",\"description\":");
      iree_profile_fprint_json_string(file, description);
      fputs("}\n", file);
    }
  }
  return status;
}

static iree_status_t iree_profile_export_process_device_metric_sample_records(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_device_metric_sample_record_t),
      &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_device_metric_sample_record_t sample_record;
    memcpy(&sample_record, typed_record.contents.data, sizeof(sample_record));
    iree_host_size_t values_length = 0;
    if (!iree_host_size_checked_mul(
            sample_record.value_count,
            sizeof(iree_hal_profile_device_metric_value_t), &values_length) ||
        values_length != typed_record.inline_payload.data_length) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "device metric sample value count is "
                                "inconsistent with record length");
    }
    if (iree_status_is_ok(status)) {
      const iree_hal_profile_device_metric_value_t* values =
          (const iree_hal_profile_device_metric_value_t*)
              typed_record.inline_payload.data;
      for (uint32_t i = 0;
           i < sample_record.value_count && iree_status_is_ok(status); ++i) {
        const iree_profile_model_metric_descriptor_t* descriptor = NULL;
        status = iree_profile_model_resolve_metric_descriptor(
            model, sample_record.source_id, values[i].metric_id, &descriptor);
      }
    }
    if (iree_status_is_ok(status)) {
      const iree_hal_profile_device_metric_value_t* values =
          (const iree_hal_profile_device_metric_value_t*)
              typed_record.inline_payload.data;
      const int64_t host_time_uncertainty_ns =
          sample_record.host_time_end_ns >= sample_record.host_time_begin_ns
              ? sample_record.host_time_end_ns -
                    sample_record.host_time_begin_ns
              : 0;
      const bool has_source_timestamp = iree_all_bits_set(
          sample_record.flags,
          IREE_HAL_PROFILE_DEVICE_METRIC_SAMPLE_FLAG_SOURCE_TIMESTAMP);
      iree_profile_export_print_prefix(file, "device_metric_sample",
                                       record_index);
      fprintf(file,
              ",\"sample_id\":%" PRIu64 ",\"source_id\":%" PRIu64
              ",\"flags\":%u"
              ",\"physical_device_ordinal\":%u"
              ",\"host_time_domain\":\"iree_host_time_ns\""
              ",\"host_time_begin_ns\":%" PRId64
              ",\"host_time_end_ns\":%" PRId64
              ",\"host_time_uncertainty_ns\":%" PRId64
              ",\"source_timestamp_present\":%s"
              ",\"source_timestamp\":%" PRIu64
              ",\"source_timestamp_frequency_hz\":%" PRIu64 ",\"values\":[",
              sample_record.sample_id, sample_record.source_id,
              sample_record.flags, sample_record.physical_device_ordinal,
              sample_record.host_time_begin_ns, sample_record.host_time_end_ns,
              host_time_uncertainty_ns, has_source_timestamp ? "true" : "false",
              sample_record.source_timestamp,
              sample_record.source_timestamp_frequency_hz);
      for (uint32_t i = 0; i < sample_record.value_count; ++i) {
        if (i != 0) fputc(',', file);
        const iree_profile_model_metric_descriptor_t* descriptor =
            iree_profile_model_find_metric_descriptor(
                model, sample_record.source_id, values[i].metric_id);
        fprintf(file,
                "{\"metric_id\":%" PRIu64 ",\"name\":", values[i].metric_id);
        iree_profile_fprint_json_string(file, descriptor->name);
        fprintf(file, ",\"unit\":");
        iree_profile_fprint_json_string(
            file, iree_hal_profile_metric_unit_string(descriptor->record.unit));
        fprintf(file, ",\"value_kind\":");
        iree_profile_fprint_json_string(
            file, iree_hal_profile_metric_value_kind_string(
                      descriptor->record.value_kind));
        fprintf(file, ",\"semantic\":");
        iree_profile_fprint_json_string(file,
                                        iree_hal_profile_metric_semantic_string(
                                            descriptor->record.semantic));
        fprintf(file, ",\"plot_hint\":");
        iree_profile_fprint_json_string(
            file, iree_hal_profile_metric_plot_hint_string(
                      descriptor->record.plot_hint));
        fprintf(file,
                ",\"flags\":%u"
                ",\"value\":",
                values[i].flags);
        iree_profile_export_print_device_metric_value(
            file, descriptor->record.value_kind, values[i].value_bits);
        fprintf(file, ",\"value_bits\":%" PRIu64 "}", values[i].value_bits);
      }
      fputs("]}\n", file);
    }
  }
  return status;
}

static const char* iree_profile_executable_trace_format_name(
    iree_hal_profile_executable_trace_format_t format) {
  switch (format) {
    case IREE_HAL_PROFILE_EXECUTABLE_TRACE_FORMAT_NONE:
      return "none";
    case IREE_HAL_PROFILE_EXECUTABLE_TRACE_FORMAT_AMDGPU_ATT:
      return "amdgpu_att";
    default:
      return "unknown";
  }
}

static iree_status_t iree_profile_export_process_executable_trace_record(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  (void)model;
  iree_hal_profile_executable_trace_record_t trace_record;
  IREE_RETURN_IF_ERROR(iree_profile_executable_trace_record_parse(
      record, &trace_record, /*out_trace_data=*/NULL));

  iree_profile_export_print_prefix(file, "executable_trace", record_index);
  fprintf(file, ",\"trace_id\":%" PRIu64 ",\"format\":", trace_record.trace_id);
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(iree_profile_executable_trace_format_name(
                trace_record.format)));
  fprintf(file,
          ",\"format_value\":%u,\"flags\":%u,\"shader_engine\":%u"
          ",\"dispatch_event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64
          ",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u"
          ",\"executable_id\":%" PRIu64
          ",\"export_ordinal\":%u"
          ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
          ",\"stream_id\":%" PRIu64
          ",\"record_length\":%u"
          ",\"data_length\":%" PRIu64 "}\n",
          trace_record.format, trace_record.flags, trace_record.shader_engine,
          trace_record.dispatch_event_id, trace_record.submission_id,
          trace_record.command_buffer_id, trace_record.command_index,
          trace_record.executable_id, trace_record.export_ordinal,
          trace_record.physical_device_ordinal, trace_record.queue_ordinal,
          trace_record.stream_id, trace_record.record_length,
          trace_record.data_length);
  return iree_ok_status();
}

typedef iree_status_t (*iree_profile_export_chunk_processor_fn_t)(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file);

typedef struct iree_profile_export_chunk_route_t {
  // Profile chunk content type handled by this route.
  iree_string_view_t content_type;
  // Processor used to emit zero or more interchange rows for the chunk.
  iree_profile_export_chunk_processor_fn_t process;
} iree_profile_export_chunk_route_t;

static iree_status_t iree_profile_export_process_chunk_record(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  const iree_profile_export_chunk_route_t routes[] = {
      {IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES,
       iree_profile_export_process_device_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES,
       iree_profile_export_process_queue_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES,
       iree_profile_export_process_executable_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS,
       iree_profile_export_process_executable_export_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS,
       iree_profile_export_process_command_buffer_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS,
       iree_profile_export_process_command_operation_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS,
       iree_profile_export_process_clock_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS,
       iree_profile_export_process_dispatch_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS,
       iree_profile_export_process_queue_event_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS,
       iree_profile_export_process_queue_device_event_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS,
       iree_profile_export_process_host_execution_event_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_REGION_EVENTS,
       iree_profile_export_process_command_region_event_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS,
       iree_profile_export_process_memory_event_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_EVENT_RELATIONSHIPS,
       iree_profile_export_process_event_relationship_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS,
       iree_profile_export_process_counter_set_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS,
       iree_profile_export_process_counter_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES,
       iree_profile_export_process_counter_sample_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SOURCES,
       iree_profile_export_process_device_metric_source_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_DESCRIPTORS,
       iree_profile_export_process_device_metric_descriptor_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SAMPLES,
       iree_profile_export_process_device_metric_sample_records},
      {IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACES,
       iree_profile_export_process_executable_trace_record},
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(routes); ++i) {
    if (iree_string_view_equal(record->content_type, routes[i].content_type)) {
      return routes[i].process(model, record, record_index, file);
    }
  }

  iree_profile_export_print_diagnostic(
      record_index, "warning", "unknown_chunk", record->content_type,
      "unknown profile chunk content type", file);
  return iree_ok_status();
}

static iree_status_t iree_profile_export_process_decoded_record(
    const iree_profile_model_t* model,
    const iree_hal_profile_file_record_t* record, iree_host_size_t record_index,
    FILE* file) {
  if (record->header.record_type ==
          IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN ||
      record->header.record_type ==
          IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END) {
    iree_profile_export_print_session_record(record, record_index, file);
    return iree_ok_status();
  }
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    iree_profile_export_print_diagnostic(
        record_index, "warning", "unknown_record", record->content_type,
        "unknown profile file record type", file);
    return iree_ok_status();
  }

  if (iree_any_bit_set(record->header.chunk_flags,
                       IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED)) {
    iree_profile_export_print_truncated_chunk_diagnostic(record, record_index,
                                                         file);
  }

  return iree_profile_export_process_chunk_record(model, record, record_index,
                                                  file);
}

typedef struct iree_profile_export_parse_context_t {
  // Shared profile metadata used to resolve executable export keys.
  iree_profile_model_t* model;
  // Output stream receiving decoded JSONL records.
  FILE* file;
} iree_profile_export_parse_context_t;

static iree_status_t iree_profile_export_metadata_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_export_parse_context_t* context =
      (iree_profile_export_parse_context_t*)user_data;
  return iree_profile_model_process_metadata_record(context->model, record);
}

static iree_status_t iree_profile_export_decoded_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  iree_profile_export_parse_context_t* context =
      (iree_profile_export_parse_context_t*)user_data;
  return iree_profile_export_process_decoded_record(
      context->model, record, record_index, context->file);
}

static iree_status_t iree_profile_export_ireeperf_jsonl_file(
    iree_string_view_t path, FILE* file, iree_allocator_t host_allocator) {
  iree_profile_file_t profile_file;
  IREE_RETURN_IF_ERROR(
      iree_profile_file_open(path, host_allocator, &profile_file));

  iree_profile_model_t model;
  iree_profile_model_initialize(host_allocator, &model);
  iree_profile_export_parse_context_t parse_context = {
      .model = &model,
      .file = file,
  };
  iree_profile_file_record_callback_t record_callback = {
      .fn = iree_profile_export_metadata_record,
      .user_data = &parse_context,
  };
  iree_status_t status =
      iree_profile_file_for_each_record(&profile_file, record_callback);

  if (iree_status_is_ok(status)) {
    fprintf(file,
            "{\"schema_version\":%d,\"record_type\":\"schema\""
            ",\"format\":\"ireeperf-jsonl\""
            ",\"contract\":\"schema_versioned_interchange\""
            ",\"row_key\":\"record_type\""
            ",\"source_format\":\"ireeprof\""
            ",\"source_version_major\":%u,\"source_version_minor\":%u"
            ",\"time_domains\":{"
            "\"iree_host_time_ns\":\"iree_time_now monotonic nanoseconds\","
            "\"device_tick\":\"raw physical-device timestamp ticks\","
            "\"driver_host_cpu_timestamp_ns\":"
            "\"driver-sampled CPU timestamp nanoseconds\","
            "\"driver_host_system_timestamp\":"
            "\"driver-sampled system timestamp units scaled by "
            "host_system_frequency_hz\","
            "\"device_tick_duration_ns\":"
            "\"duration nanoseconds derived from device_tick clock fit\"},"
            "\"clock_fit\":{\"basis\":\"first_last_clock_correlation\","
            "\"rounding\":\"nearest_integer_nanosecond\"}}\n",
            IREE_PROFILE_EXPORT_SCHEMA_VERSION,
            profile_file.header.version_major,
            profile_file.header.version_minor);
  }

  if (iree_status_is_ok(status)) {
    record_callback.fn = iree_profile_export_decoded_record;
    status = iree_profile_file_for_each_record(&profile_file, record_callback);
  }

  iree_profile_model_deinitialize(&model);
  iree_profile_file_close(&profile_file);
  return status;
}

iree_status_t iree_profile_export_file(iree_string_view_t path,
                                       iree_string_view_t format,
                                       iree_string_view_t output_path,
                                       iree_allocator_t host_allocator) {
  if (!iree_string_view_equal(format, IREE_SV("ireeperf-jsonl"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile export format '%.*s'",
                            (int)format.size, format.data);
  }

  FILE* file = stdout;
  bool should_close_file = false;
  if (!iree_string_view_equal(output_path, IREE_SV("-"))) {
    char* c_path = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        host_allocator, output_path.size + 1, (void**)&c_path));
    iree_string_view_to_cstring(output_path, c_path, output_path.size + 1);
    file = fopen(c_path, "wb");
    const int open_errno = errno;
    iree_allocator_free(host_allocator, c_path);
    if (!file) {
      return iree_make_status(iree_status_code_from_errno(open_errno),
                              "failed to open export output file: %s",
                              strerror(open_errno));
    }
    should_close_file = true;
  }

  iree_status_t status =
      iree_profile_export_ireeperf_jsonl_file(path, file, host_allocator);
  if (should_close_file) {
    if (fclose(file) != 0 && iree_status_is_ok(status)) {
      const int close_errno = errno;
      status = iree_make_status(iree_status_code_from_errno(close_errno),
                                "failed to close export output file: %s",
                                strerror(close_errno));
    }
  }
  return status;
}
