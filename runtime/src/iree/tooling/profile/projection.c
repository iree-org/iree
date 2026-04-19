// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/projection.h"

#include "iree/tooling/profile/reader.h"

static bool iree_profile_command_operation_filter_matches(
    iree_string_view_t operation_name, iree_string_view_t key,
    iree_string_view_t filter) {
  return iree_profile_key_matches(operation_name, filter) ||
         iree_profile_key_matches(key, filter);
}

static bool iree_profile_projection_try_fit_driver_host_cpu_clock(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal,
    iree_profile_model_clock_fit_t* out_clock_fit) {
  const iree_profile_model_device_t* device =
      iree_profile_model_find_device(&context->model, physical_device_ordinal);
  return iree_profile_model_device_try_fit_clock_exact(
      device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS,
      out_clock_fit);
}

typedef struct iree_profile_projection_dispatch_timing_t {
  // True when the aggregate's device has a host CPU clock fit.
  bool has_clock_fit;
  // True when min/max/total tick values were converted to nanoseconds.
  bool has_time_ns;
  // Average dispatch duration in device ticks.
  double average_ticks;
  // Nanoseconds per device tick when |has_clock_fit| is true.
  double ns_per_tick;
  // Minimum dispatch duration in nanoseconds when |has_time_ns| is true.
  int64_t minimum_ns;
  // Maximum dispatch duration in nanoseconds when |has_time_ns| is true.
  int64_t maximum_ns;
  // Total dispatch duration in nanoseconds when |has_time_ns| is true.
  int64_t total_ns;
} iree_profile_projection_dispatch_timing_t;

static iree_profile_projection_dispatch_timing_t
iree_profile_projection_calculate_dispatch_timing(
    const iree_profile_dispatch_context_t* context,
    const iree_profile_dispatch_aggregate_t* aggregate) {
  iree_profile_projection_dispatch_timing_t timing = {0};
  timing.average_ticks =
      aggregate->valid_count
          ? aggregate->total_ticks / (double)aggregate->valid_count
          : 0.0;

  iree_profile_model_clock_fit_t clock_fit;
  timing.has_clock_fit = iree_profile_projection_try_fit_driver_host_cpu_clock(
      context, aggregate->physical_device_ordinal, &clock_fit);
  timing.ns_per_tick =
      timing.has_clock_fit
          ? iree_profile_model_clock_fit_ns_per_tick(&clock_fit)
          : 0.0;
  timing.has_time_ns =
      timing.has_clock_fit && aggregate->valid_count != 0 &&
      iree_profile_model_clock_fit_scale_ticks_to_ns(
          &clock_fit, aggregate->minimum_ticks, &timing.minimum_ns) &&
      iree_profile_model_clock_fit_scale_ticks_to_ns(
          &clock_fit, aggregate->maximum_ticks, &timing.maximum_ns) &&
      iree_profile_model_clock_fit_scale_ticks_to_ns(
          &clock_fit, aggregate->total_ticks, &timing.total_ns);
  return timing;
}

typedef struct iree_profile_projection_submission_timing_t {
  // True when the aggregate's device has a host CPU clock fit.
  bool has_clock_fit;
  // True when |total_dispatch_ticks| was converted to nanoseconds.
  bool has_total_dispatch_ns;
  // Span from earliest dispatch start to latest dispatch end in device ticks.
  double span_ticks;
  // Nanoseconds per device tick when |has_clock_fit| is true.
  double ns_per_tick;
  // Total dispatch duration in nanoseconds when |has_total_dispatch_ns| is
  // true.
  int64_t total_dispatch_ns;
} iree_profile_projection_submission_timing_t;

static iree_profile_projection_submission_timing_t
iree_profile_projection_calculate_submission_timing(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal, uint64_t earliest_start_tick,
    uint64_t latest_end_tick, uint64_t valid_count,
    uint64_t total_dispatch_ticks) {
  iree_profile_projection_submission_timing_t timing = {0};
  timing.span_ticks =
      iree_profile_model_span_ticks(earliest_start_tick, latest_end_tick);

  iree_profile_model_clock_fit_t clock_fit;
  timing.has_clock_fit = iree_profile_projection_try_fit_driver_host_cpu_clock(
      context, physical_device_ordinal, &clock_fit);
  timing.ns_per_tick =
      timing.has_clock_fit
          ? iree_profile_model_clock_fit_ns_per_tick(&clock_fit)
          : 0.0;
  timing.has_total_dispatch_ns =
      timing.has_clock_fit && valid_count != 0 &&
      iree_profile_model_clock_fit_scale_ticks_to_ns(
          &clock_fit, total_dispatch_ticks, &timing.total_dispatch_ns);
  return timing;
}

static double iree_profile_projection_dispatch_stddev_ticks(
    const iree_profile_dispatch_aggregate_t* aggregate) {
  const double variance_ticks =
      aggregate->valid_count > 1
          ? aggregate->m2_ticks / (double)(aggregate->valid_count - 1)
          : 0.0;
  return iree_profile_sqrt_f64(variance_ticks);
}

static void iree_profile_dispatch_print_event_jsonl(
    const iree_profile_dispatch_event_row_t* row, FILE* file) {
  const iree_hal_profile_file_record_t* file_record = row->file_record;
  const iree_hal_profile_dispatch_event_t* event = row->event;
  const bool is_valid = event->start_tick != 0 && event->end_tick != 0 &&
                        event->end_tick >= event->start_tick;
  const uint64_t duration_ticks =
      is_valid ? event->end_tick - event->start_tick : 0;
  int64_t duration_ns = 0;
  const bool has_duration_ns =
      row->has_clock_fit && is_valid &&
      iree_profile_model_clock_fit_scale_ticks_to_ns(
          row->clock_fit, duration_ticks, &duration_ns);

  fprintf(file,
          "{\"type\":\"dispatch_event\",\"physical_device_ordinal\":%u"
          ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64
          ",\"event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64,
          file_record->header.physical_device_ordinal,
          file_record->header.queue_ordinal, file_record->header.stream_id,
          event->event_id, event->submission_id);
  fprintf(file,
          ",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u"
          ",\"executable_id\":%" PRIu64 ",\"export_ordinal\":%u",
          event->command_buffer_id, event->command_index, event->executable_id,
          event->export_ordinal);
  fprintf(file, ",\"key\":");
  iree_profile_fprint_json_string(file, row->key);
  fprintf(file,
          ",\"flags\":%u,\"workgroup_count\":[%u,%u,%u]"
          ",\"workgroup_size\":[%u,%u,%u]",
          event->flags, event->workgroup_count[0], event->workgroup_count[1],
          event->workgroup_count[2], event->workgroup_size[0],
          event->workgroup_size[1], event->workgroup_size[2]);
  fprintf(file,
          ",\"start_tick\":%" PRIu64 ",\"end_tick\":%" PRIu64
          ",\"duration_ticks\":%" PRIu64 ",\"valid\":%s",
          event->start_tick, event->end_tick, duration_ticks,
          is_valid ? "true" : "false");
  fprintf(file, ",\"clock_fit_available\":%s",
          row->has_clock_fit ? "true" : "false");
  fprintf(file,
          ",\"device_tick_domain\":\"device_tick\""
          ",\"duration_time_domain\":\"device_tick_duration_ns\""
          ",\"duration_ns\":%" PRId64,
          has_duration_ns ? duration_ns : 0);
  fputs("}\n", file);
}

static iree_status_t iree_profile_projection_emit_dispatch_event(
    void* user_data, const iree_profile_dispatch_event_row_t* row) {
  FILE* file = (FILE*)user_data;
  iree_profile_dispatch_print_event_jsonl(row, file);
  return iree_ok_status();
}

static iree_status_t iree_profile_command_count_matching_operations(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, iree_host_size_t* out_operation_count) {
  *out_operation_count = 0;
  iree_host_size_t operation_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->model.command_operation_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_profile_command_operation_record_t* operation =
        &context->model.command_operations[i].record;
    if (id_filter >= 0 && operation->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    const char* operation_name =
        iree_profile_command_operation_type_name(operation->type);
    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_model_resolve_command_operation_key(
        &context->model, operation, numeric_buffer, sizeof(numeric_buffer),
        &key);
    if (iree_status_is_ok(status) &&
        iree_profile_command_operation_filter_matches(
            iree_make_cstring_view(operation_name), key, filter)) {
      ++operation_count;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_operation_count = operation_count;
  }
  return status;
}

static iree_status_t iree_profile_command_print_operation_text(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* operation,
    iree_string_view_t filter, FILE* file) {
  const char* operation_name =
      iree_profile_command_operation_type_name(operation->type);
  char numeric_buffer[128];
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_profile_model_resolve_command_operation_key(
      &context->model, operation, numeric_buffer, sizeof(numeric_buffer),
      &key));
  if (!iree_profile_command_operation_filter_matches(
          iree_make_cstring_view(operation_name), key, filter)) {
    return iree_ok_status();
  }

  fprintf(file, "    command[%u]: op=%s flags=0x%x", operation->command_index,
          operation_name, operation->flags);
  if (iree_hal_profile_command_operation_has_block_structure(operation)) {
    fprintf(file, " block=%u local=%u", operation->block_ordinal,
            operation->block_command_ordinal);
  }
  if (operation->type == IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH) {
    fprintf(file,
            " executable=%" PRIu64
            " export=%u key=%.*s bindings=%u"
            " workgroups=[%u,%u,%u] workgroup_size=[%u,%u,%u]",
            operation->executable_id, operation->export_ordinal, (int)key.size,
            key.data, operation->binding_count, operation->workgroup_count[0],
            operation->workgroup_count[1], operation->workgroup_count[2],
            operation->workgroup_size[0], operation->workgroup_size[1],
            operation->workgroup_size[2]);
  } else if (operation->type == IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL ||
             operation->type ==
                 IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_UPDATE) {
    fprintf(file, " target=%u target_offset=%" PRIu64 " length=%" PRIu64,
            operation->target_ordinal, operation->target_offset,
            operation->length);
  } else if (operation->type == IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COPY) {
    fprintf(file,
            " source=%u source_offset=%" PRIu64
            " target=%u target_offset=%" PRIu64 " length=%" PRIu64,
            operation->source_ordinal, operation->source_offset,
            operation->target_ordinal, operation->target_offset,
            operation->length);
  } else if (operation->type ==
                 IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BRANCH ||
             operation->type ==
                 IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COND_BRANCH) {
    fprintf(file, " target_block=%u alternate_block=%u",
            operation->target_block_ordinal,
            operation->alternate_block_ordinal);
  }
  fputc('\n', file);
  return iree_ok_status();
}

static iree_status_t iree_profile_command_print_operation_jsonl(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* operation,
    iree_string_view_t filter, FILE* file) {
  const char* operation_name =
      iree_profile_command_operation_type_name(operation->type);
  char numeric_buffer[128];
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_profile_model_resolve_command_operation_key(
      &context->model, operation, numeric_buffer, sizeof(numeric_buffer),
      &key));
  if (!iree_profile_command_operation_filter_matches(
          iree_make_cstring_view(operation_name), key, filter)) {
    return iree_ok_status();
  }

  fprintf(file,
          "{\"type\":\"command_operation\",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u,\"op\":\"%s\",\"flags\":%u"
          ",\"block_structure\":%s"
          ",\"block_ordinal\":%u,\"block_command_ordinal\":%u",
          operation->command_buffer_id, operation->command_index,
          operation_name, operation->flags,
          iree_hal_profile_command_operation_has_block_structure(operation)
              ? "true"
              : "false",
          operation->block_ordinal, operation->block_command_ordinal);
  fprintf(file, ",\"key\":");
  iree_profile_fprint_json_string(file, key);
  fprintf(file,
          ",\"executable_id\":%" PRIu64
          ",\"export_ordinal\":%u"
          ",\"binding_count\":%u,\"workgroup_count\":[%u,%u,%u]"
          ",\"workgroup_size\":[%u,%u,%u]"
          ",\"source_ordinal\":%u,\"target_ordinal\":%u"
          ",\"source_offset\":%" PRIu64 ",\"target_offset\":%" PRIu64
          ",\"length\":%" PRIu64
          ",\"target_block_ordinal\":%u,\"alternate_block_ordinal\":%u}\n",
          operation->executable_id, operation->export_ordinal,
          operation->binding_count, operation->workgroup_count[0],
          operation->workgroup_count[1], operation->workgroup_count[2],
          operation->workgroup_size[0], operation->workgroup_size[1],
          operation->workgroup_size[2], operation->source_ordinal,
          operation->target_ordinal, operation->source_offset,
          operation->target_offset, operation->length,
          operation->target_block_ordinal, operation->alternate_block_ordinal);
  return iree_ok_status();
}

static void iree_profile_dispatch_print_text_header(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    FILE* file) {
  fprintf(file, "IREE HAL profile dispatch summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "dispatches: total=%" PRIu64 " matched=%" PRIu64 " valid=%" PRIu64
          " invalid=%" PRIu64 " groups=%" PRIhsz "\n",
          context->total_dispatch_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count,
          context->aggregate_count);
  fprintf(file, "groups:\n");
}

static void iree_profile_dispatch_print_text_group(
    const iree_profile_dispatch_context_t* context,
    const iree_profile_dispatch_aggregate_t* aggregate, iree_string_view_t key,
    FILE* file) {
  const iree_profile_projection_dispatch_timing_t timing =
      iree_profile_projection_calculate_dispatch_timing(context, aggregate);
  const double stddev_ticks =
      iree_profile_projection_dispatch_stddev_ticks(aggregate);
  fprintf(file, "  %.*s\n", (int)key.size, key.data);
  fprintf(file,
          "    device=%u executable=%" PRIu64 " export=%u count=%" PRIu64
          " valid=%" PRIu64 " invalid=%" PRIu64 "\n",
          aggregate->physical_device_ordinal, aggregate->executable_id,
          aggregate->export_ordinal, aggregate->dispatch_count,
          aggregate->valid_count, aggregate->invalid_count);
  if (aggregate->valid_count != 0) {
    fprintf(file,
            "    ticks: min=%" PRIu64 " avg=%.3f stddev=%.3f max=%" PRIu64
            " total=%" PRIu64 "\n",
            aggregate->minimum_ticks, aggregate->mean_ticks, stddev_ticks,
            aggregate->maximum_ticks, aggregate->total_ticks);
    if (timing.has_time_ns) {
      fprintf(file,
              "    time_ns: min=%" PRId64
              " avg=%.3f stddev=%.3f"
              " max=%" PRId64 " total=%" PRId64 "\n",
              timing.minimum_ns, aggregate->mean_ticks * timing.ns_per_tick,
              stddev_ticks * timing.ns_per_tick, timing.maximum_ns,
              timing.total_ns);
    } else {
      fprintf(file, "    time_ns: unavailable\n");
    }
  }
  fprintf(file,
          "    last_geometry: workgroup_count=%ux%ux%u"
          " workgroup_size=%ux%ux%u\n",
          aggregate->last_workgroup_count[0],
          aggregate->last_workgroup_count[1],
          aggregate->last_workgroup_count[2], aggregate->last_workgroup_size[0],
          aggregate->last_workgroup_size[1], aggregate->last_workgroup_size[2]);
}

static iree_status_t iree_profile_dispatch_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    FILE* file) {
  iree_profile_dispatch_print_text_header(context, filter, file);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_dispatch_aggregate_t* aggregate =
        &context->aggregates[i];
    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_model_resolve_dispatch_key(
        &context->model, aggregate->physical_device_ordinal,
        aggregate->executable_id, aggregate->export_ordinal, numeric_buffer,
        sizeof(numeric_buffer), &key);
    if (iree_status_is_ok(status)) {
      iree_profile_dispatch_print_text_group(context, aggregate, key, file);
    }
  }
  return status;
}

static void iree_profile_dispatch_print_jsonl_summary(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    bool emit_events, FILE* file) {
  fprintf(file, "{\"type\":\"dispatch_summary\",\"mode\":\"%s\",\"filter\":",
          emit_events ? "events" : "aggregate");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"total_dispatches\":%" PRIu64 ",\"matched_dispatches\":%" PRIu64
          ",\"valid_dispatches\":%" PRIu64 ",\"invalid_dispatches\":%" PRIu64
          ",\"aggregate_groups\":%" PRIhsz "}\n",
          context->total_dispatch_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count,
          context->aggregate_count);
}

static void iree_profile_dispatch_print_jsonl_group(
    const iree_profile_dispatch_context_t* context,
    const iree_profile_dispatch_aggregate_t* aggregate, iree_string_view_t key,
    FILE* file) {
  const iree_profile_projection_dispatch_timing_t timing =
      iree_profile_projection_calculate_dispatch_timing(context, aggregate);
  const double stddev_ticks =
      iree_profile_projection_dispatch_stddev_ticks(aggregate);
  fprintf(file,
          "{\"type\":\"dispatch_group\",\"physical_device_ordinal\":%u"
          ",\"executable_id\":%" PRIu64 ",\"export_ordinal\":%u,\"key\":",
          aggregate->physical_device_ordinal, aggregate->executable_id,
          aggregate->export_ordinal);
  iree_profile_fprint_json_string(file, key);
  fprintf(file,
          ",\"count\":%" PRIu64 ",\"valid\":%" PRIu64 ",\"invalid\":%" PRIu64
          ",\"min_ticks\":%" PRIu64
          ",\"avg_ticks\":%.3f,\"stddev_ticks\":%.3f"
          ",\"max_ticks\":%" PRIu64 ",\"total_ticks\":%" PRIu64,
          aggregate->dispatch_count, aggregate->valid_count,
          aggregate->invalid_count,
          aggregate->valid_count ? aggregate->minimum_ticks : 0,
          aggregate->valid_count ? aggregate->mean_ticks : 0.0, stddev_ticks,
          aggregate->valid_count ? aggregate->maximum_ticks : 0,
          aggregate->total_ticks);
  fprintf(file, ",\"clock_fit_available\":%s",
          timing.has_clock_fit ? "true" : "false");
  fprintf(file,
          ",\"time_ns_available\":%s"
          ",\"min_ns\":%" PRId64
          ",\"avg_ns\":%.3f"
          ",\"stddev_ns\":%.3f,\"max_ns\":%" PRId64 ",\"total_ns\":%" PRId64,
          timing.has_time_ns ? "true" : "false",
          timing.has_time_ns ? timing.minimum_ns : 0,
          timing.has_time_ns ? aggregate->mean_ticks * timing.ns_per_tick : 0.0,
          timing.has_time_ns ? stddev_ticks * timing.ns_per_tick : 0.0,
          timing.has_time_ns ? timing.maximum_ns : 0,
          timing.has_time_ns ? timing.total_ns : 0);
  fprintf(file,
          ",\"last_workgroup_count\":[%u,%u,%u]"
          ",\"last_workgroup_size\":[%u,%u,%u]}\n",
          aggregate->last_workgroup_count[0],
          aggregate->last_workgroup_count[1],
          aggregate->last_workgroup_count[2], aggregate->last_workgroup_size[0],
          aggregate->last_workgroup_size[1], aggregate->last_workgroup_size[2]);
}

static iree_status_t iree_profile_dispatch_print_jsonl_aggregates(
    const iree_profile_dispatch_context_t* context, FILE* file) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_dispatch_aggregate_t* aggregate =
        &context->aggregates[i];
    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_model_resolve_dispatch_key(
        &context->model, aggregate->physical_device_ordinal,
        aggregate->executable_id, aggregate->export_ordinal, numeric_buffer,
        sizeof(numeric_buffer), &key);
    if (iree_status_is_ok(status)) {
      iree_profile_dispatch_print_jsonl_group(context, aggregate, key, file);
    }
  }
  return status;
}

static bool iree_profile_executable_matches_id(
    const iree_hal_profile_executable_record_t* executable, int64_t id_filter) {
  return id_filter < 0 || executable->executable_id == (uint64_t)id_filter;
}

static bool iree_profile_executable_export_matches_executable(
    const iree_profile_model_export_t* export_info,
    const iree_hal_profile_executable_record_t* executable) {
  return export_info->executable_id == executable->executable_id;
}

static bool iree_profile_executable_dispatch_group_matches_export(
    const iree_profile_dispatch_aggregate_t* aggregate,
    const iree_profile_model_export_t* export_info) {
  return aggregate->executable_id == export_info->executable_id &&
         aggregate->export_ordinal == export_info->export_ordinal;
}

static void iree_profile_executable_print_text_header(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    FILE* file) {
  fprintf(file, "IREE HAL profile executable summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "executables=%" PRIhsz " exports=%" PRIhsz
          " matched_dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
          "\n",
          context->model.executable_count, context->model.export_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);
}

static void iree_profile_executable_print_text_executable(
    const iree_hal_profile_executable_record_t* executable, FILE* file) {
  const bool has_code_object_hash = iree_all_bits_set(
      executable->flags, IREE_HAL_PROFILE_EXECUTABLE_FLAG_CODE_OBJECT_HASH);
  fprintf(file, "executable %" PRIu64 ": exports=%u flags=%u ",
          executable->executable_id, executable->export_count,
          executable->flags);
  fprintf(file, "code_object_hash=");
  if (has_code_object_hash) {
    iree_profile_fprint_hash_hex(file, executable->code_object_hash);
  } else {
    fprintf(file, "unavailable");
  }
  fputc('\n', file);
}

static void iree_profile_executable_print_text_export(
    const iree_profile_model_export_t* export_info, iree_string_view_t key,
    FILE* file) {
  const bool has_pipeline_hash =
      iree_all_bits_set(export_info->flags,
                        IREE_HAL_PROFILE_EXECUTABLE_EXPORT_FLAG_PIPELINE_HASH);
  fprintf(file,
          "  export %u: %.*s flags=%u constants=%u bindings=%u "
          "parameters=%u workgroup_size=%ux%ux%u pipeline_hash=",
          export_info->export_ordinal, (int)key.size, key.data,
          export_info->flags, export_info->constant_count,
          export_info->binding_count, export_info->parameter_count,
          export_info->workgroup_size[0], export_info->workgroup_size[1],
          export_info->workgroup_size[2]);
  if (has_pipeline_hash) {
    iree_profile_fprint_hash_hex(file, export_info->pipeline_hash);
  } else {
    fprintf(file, "unavailable");
  }
  fputc('\n', file);
}

static void iree_profile_executable_print_text_dispatch_group(
    const iree_profile_dispatch_context_t* context,
    const iree_profile_dispatch_aggregate_t* aggregate, FILE* file) {
  const iree_profile_projection_dispatch_timing_t timing =
      iree_profile_projection_calculate_dispatch_timing(context, aggregate);
  fprintf(file,
          "    device=%u dispatches=%" PRIu64 " valid=%" PRIu64
          " invalid=%" PRIu64 " ticks[min/avg/max/total]=%" PRIu64
          "/%.3f/%" PRIu64 "/%" PRIu64,
          aggregate->physical_device_ordinal, aggregate->dispatch_count,
          aggregate->valid_count, aggregate->invalid_count,
          aggregate->valid_count ? aggregate->minimum_ticks : 0,
          timing.average_ticks,
          aggregate->valid_count ? aggregate->maximum_ticks : 0,
          aggregate->total_ticks);
  if (timing.has_time_ns) {
    fprintf(file,
            " ns[min/avg/max/total]=%" PRId64 "/%.3f/%" PRId64 "/%" PRId64,
            timing.minimum_ns, timing.average_ticks * timing.ns_per_tick,
            timing.maximum_ns, timing.total_ns);
  }
  fputc('\n', file);
}

static bool iree_profile_executable_print_text_dispatch_groups(
    const iree_profile_dispatch_context_t* context,
    const iree_profile_model_export_t* export_info, FILE* file) {
  bool has_aggregate = false;
  for (iree_host_size_t i = 0; i < context->aggregate_count; ++i) {
    const iree_profile_dispatch_aggregate_t* aggregate =
        &context->aggregates[i];
    if (!iree_profile_executable_dispatch_group_matches_export(aggregate,
                                                               export_info)) {
      continue;
    }
    has_aggregate = true;
    iree_profile_executable_print_text_dispatch_group(context, aggregate, file);
  }
  return has_aggregate;
}

static void iree_profile_executable_print_text_exports(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_executable_record_t* executable,
    iree_string_view_t filter, FILE* file) {
  for (iree_host_size_t i = 0; i < context->model.export_count; ++i) {
    const iree_profile_model_export_t* export_info = &context->model.exports[i];
    if (!iree_profile_executable_export_matches_executable(export_info,
                                                           executable)) {
      continue;
    }

    char numeric_buffer[128];
    iree_string_view_t key = iree_profile_model_format_export_key(
        export_info, UINT32_MAX, numeric_buffer, sizeof(numeric_buffer));
    if (!iree_profile_key_matches(key, filter)) {
      continue;
    }

    iree_profile_executable_print_text_export(export_info, key, file);
    if (!iree_profile_executable_print_text_dispatch_groups(
            context, export_info, file)) {
      fprintf(file, "    dispatches=0\n");
    }
  }
}

static iree_status_t iree_profile_executable_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_profile_executable_print_text_header(context, filter, file);

  for (iree_host_size_t i = 0; i < context->model.executable_count; ++i) {
    const iree_hal_profile_executable_record_t* executable =
        &context->model.executables[i].record;
    if (!iree_profile_executable_matches_id(executable, id_filter)) {
      continue;
    }
    iree_profile_executable_print_text_executable(executable, file);
    iree_profile_executable_print_text_exports(context, executable, filter,
                                               file);
  }
  return iree_ok_status();
}

static void iree_profile_executable_print_jsonl_summary(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    FILE* file) {
  fprintf(file, "{\"type\":\"executable_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"executables\":%" PRIhsz ",\"exports\":%" PRIhsz
          ",\"matched_dispatches\":%" PRIu64 ",\"valid_dispatches\":%" PRIu64
          ",\"invalid_dispatches\":%" PRIu64 "}\n",
          context->model.executable_count, context->model.export_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);
}

static void iree_profile_executable_print_jsonl_hash(const uint64_t hash[2],
                                                     bool has_hash,
                                                     FILE* file) {
  if (has_hash) {
    fputc('"', file);
    iree_profile_fprint_hash_hex(file, hash);
    fputc('"', file);
  } else {
    fprintf(file, "null");
  }
}

static void iree_profile_executable_print_jsonl_executable(
    const iree_hal_profile_executable_record_t* executable, FILE* file) {
  const bool has_code_object_hash = iree_all_bits_set(
      executable->flags, IREE_HAL_PROFILE_EXECUTABLE_FLAG_CODE_OBJECT_HASH);
  fprintf(file,
          "{\"type\":\"executable\",\"executable_id\":%" PRIu64
          ",\"flags\":%u,\"export_count\":%u"
          ",\"code_object_hash_present\":%s,\"code_object_hash\":",
          executable->executable_id, executable->flags,
          executable->export_count, has_code_object_hash ? "true" : "false");
  iree_profile_executable_print_jsonl_hash(executable->code_object_hash,
                                           has_code_object_hash, file);
  fputs("}\n", file);
}

static void iree_profile_executable_print_jsonl_export(
    const iree_profile_model_export_t* export_info, iree_string_view_t key,
    FILE* file) {
  const bool has_pipeline_hash =
      iree_all_bits_set(export_info->flags,
                        IREE_HAL_PROFILE_EXECUTABLE_EXPORT_FLAG_PIPELINE_HASH);
  fprintf(file,
          "{\"type\":\"executable_export\",\"executable_id\":%" PRIu64
          ",\"export_ordinal\":%u,\"flags\":%u,\"key\":",
          export_info->executable_id, export_info->export_ordinal,
          export_info->flags);
  iree_profile_fprint_json_string(file, key);
  fprintf(file,
          ",\"constant_count\":%u,\"binding_count\":%u"
          ",\"parameter_count\":%u,\"workgroup_size\":[%u,%u,%u]"
          ",\"pipeline_hash_present\":%s,\"pipeline_hash\":",
          export_info->constant_count, export_info->binding_count,
          export_info->parameter_count, export_info->workgroup_size[0],
          export_info->workgroup_size[1], export_info->workgroup_size[2],
          has_pipeline_hash ? "true" : "false");
  iree_profile_executable_print_jsonl_hash(export_info->pipeline_hash,
                                           has_pipeline_hash, file);
  fputs("}\n", file);
}

static void iree_profile_executable_print_jsonl_dispatch_group(
    const iree_profile_dispatch_context_t* context,
    const iree_profile_dispatch_aggregate_t* aggregate, iree_string_view_t key,
    FILE* file) {
  const iree_profile_projection_dispatch_timing_t timing =
      iree_profile_projection_calculate_dispatch_timing(context, aggregate);
  fprintf(file,
          "{\"type\":\"executable_export_dispatch_group\""
          ",\"physical_device_ordinal\":%u"
          ",\"executable_id\":%" PRIu64
          ",\"export_ordinal\":%u"
          ",\"key\":",
          aggregate->physical_device_ordinal, aggregate->executable_id,
          aggregate->export_ordinal);
  iree_profile_fprint_json_string(file, key);
  fprintf(file,
          ",\"dispatches\":%" PRIu64 ",\"valid\":%" PRIu64
          ",\"invalid\":%" PRIu64 ",\"min_ticks\":%" PRIu64
          ",\"avg_ticks\":%.3f,\"max_ticks\":%" PRIu64
          ",\"total_ticks\":%" PRIu64
          ",\"clock_fit_available\":%s"
          ",\"time_ns_available\":%s"
          ",\"min_ns\":%" PRId64
          ",\"avg_ns\":%.3f"
          ",\"max_ns\":%" PRId64 ",\"total_ns\":%" PRId64 "}\n",
          aggregate->dispatch_count, aggregate->valid_count,
          aggregate->invalid_count,
          aggregate->valid_count ? aggregate->minimum_ticks : 0,
          timing.average_ticks,
          aggregate->valid_count ? aggregate->maximum_ticks : 0,
          aggregate->total_ticks, timing.has_clock_fit ? "true" : "false",
          timing.has_time_ns ? "true" : "false",
          timing.has_time_ns ? timing.minimum_ns : 0,
          timing.has_time_ns ? timing.average_ticks * timing.ns_per_tick : 0.0,
          timing.has_time_ns ? timing.maximum_ns : 0,
          timing.has_time_ns ? timing.total_ns : 0);
}

static void iree_profile_executable_print_jsonl_dispatch_groups(
    const iree_profile_dispatch_context_t* context,
    const iree_profile_model_export_t* export_info, iree_string_view_t key,
    FILE* file) {
  for (iree_host_size_t i = 0; i < context->aggregate_count; ++i) {
    const iree_profile_dispatch_aggregate_t* aggregate =
        &context->aggregates[i];
    if (!iree_profile_executable_dispatch_group_matches_export(aggregate,
                                                               export_info)) {
      continue;
    }
    iree_profile_executable_print_jsonl_dispatch_group(context, aggregate, key,
                                                       file);
  }
}

static void iree_profile_executable_print_jsonl_exports(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_executable_record_t* executable,
    iree_string_view_t filter, FILE* file) {
  for (iree_host_size_t i = 0; i < context->model.export_count; ++i) {
    const iree_profile_model_export_t* export_info = &context->model.exports[i];
    if (!iree_profile_executable_export_matches_executable(export_info,
                                                           executable)) {
      continue;
    }

    char numeric_buffer[128];
    iree_string_view_t key = iree_profile_model_format_export_key(
        export_info, UINT32_MAX, numeric_buffer, sizeof(numeric_buffer));
    if (!iree_profile_key_matches(key, filter)) {
      continue;
    }

    iree_profile_executable_print_jsonl_export(export_info, key, file);
    iree_profile_executable_print_jsonl_dispatch_groups(context, export_info,
                                                        key, file);
  }
}

static iree_status_t iree_profile_executable_print_jsonl(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_profile_executable_print_jsonl_summary(context, filter, file);
  for (iree_host_size_t i = 0; i < context->model.executable_count; ++i) {
    const iree_hal_profile_executable_record_t* executable =
        &context->model.executables[i].record;
    if (!iree_profile_executable_matches_id(executable, id_filter)) {
      continue;
    }
    iree_profile_executable_print_jsonl_executable(executable, file);
    iree_profile_executable_print_jsonl_exports(context, executable, filter,
                                                file);
  }
  return iree_ok_status();
}

static bool iree_profile_command_buffer_matches_id(
    const iree_hal_profile_command_buffer_record_t* command_buffer,
    int64_t id_filter) {
  return id_filter < 0 ||
         command_buffer->command_buffer_id == (uint64_t)id_filter;
}

static bool iree_profile_command_operation_matches_command_buffer(
    const iree_hal_profile_command_operation_record_t* operation,
    const iree_hal_profile_command_buffer_record_t* command_buffer) {
  return operation->command_buffer_id == command_buffer->command_buffer_id;
}

static bool iree_profile_command_execution_matches_command_buffer(
    const iree_profile_dispatch_command_aggregate_t* aggregate,
    const iree_hal_profile_command_buffer_record_t* command_buffer) {
  return aggregate->command_buffer_id == command_buffer->command_buffer_id;
}

static void iree_profile_command_print_text_header(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    iree_host_size_t matched_command_operation_count, FILE* file) {
  fprintf(file, "IREE HAL profile command-buffer summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "command_buffers=%" PRIhsz " executions=%" PRIhsz
          " command_operations=%" PRIhsz " matched_command_operations=%" PRIhsz
          " matched_dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
          "\n",
          context->model.command_buffer_count, context->command_aggregate_count,
          context->model.command_operation_count,
          matched_command_operation_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count);
}

static void iree_profile_command_print_text_command_buffer(
    const iree_hal_profile_command_buffer_record_t* command_buffer,
    FILE* file) {
  fprintf(file,
          "command_buffer %" PRIu64 ": device=%u mode=%" PRIu64
          " categories=%" PRIu64 " queue_affinity=%" PRIu64 "\n",
          command_buffer->command_buffer_id,
          command_buffer->physical_device_ordinal, command_buffer->mode,
          command_buffer->command_categories, command_buffer->queue_affinity);
}

static iree_status_t iree_profile_command_print_text_operations(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_buffer_record_t* command_buffer,
    iree_string_view_t filter, FILE* file) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->model.command_operation_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_profile_command_operation_record_t* operation =
        &context->model.command_operations[i].record;
    if (!iree_profile_command_operation_matches_command_buffer(
            operation, command_buffer)) {
      continue;
    }
    status = iree_profile_command_print_operation_text(context, operation,
                                                       filter, file);
  }
  return status;
}

static void iree_profile_command_print_text_execution(
    const iree_profile_dispatch_context_t* context,
    const iree_profile_dispatch_command_aggregate_t* aggregate, FILE* file) {
  const iree_profile_projection_submission_timing_t timing =
      iree_profile_projection_calculate_submission_timing(
          context, aggregate->physical_device_ordinal,
          aggregate->earliest_start_tick, aggregate->latest_end_tick,
          aggregate->valid_count, aggregate->total_ticks);
  fprintf(file,
          "  submission=%" PRIu64 " queue=%u stream=%" PRIu64
          " dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
          " span_ticks=%.3f total_dispatch_ticks=%" PRIu64,
          aggregate->submission_id, aggregate->queue_ordinal,
          aggregate->stream_id, aggregate->dispatch_count,
          aggregate->valid_count, aggregate->invalid_count, timing.span_ticks,
          aggregate->total_ticks);
  if (timing.has_total_dispatch_ns) {
    fprintf(file, " span_ns=%.3f total_dispatch_ns=%" PRId64,
            timing.span_ticks * timing.ns_per_tick, timing.total_dispatch_ns);
  }
  fputc('\n', file);
}

static void iree_profile_command_print_text_executions(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_buffer_record_t* command_buffer,
    FILE* file) {
  for (iree_host_size_t i = 0; i < context->command_aggregate_count; ++i) {
    const iree_profile_dispatch_command_aggregate_t* aggregate =
        &context->command_aggregates[i];
    if (!iree_profile_command_execution_matches_command_buffer(
            aggregate, command_buffer)) {
      continue;
    }
    iree_profile_command_print_text_execution(context, aggregate, file);
  }
}

static iree_status_t iree_profile_command_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_host_size_t matched_command_operation_count = 0;
  IREE_RETURN_IF_ERROR(iree_profile_command_count_matching_operations(
      context, filter, id_filter, &matched_command_operation_count));

  iree_profile_command_print_text_header(context, filter,
                                         matched_command_operation_count, file);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->model.command_buffer_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_profile_command_buffer_record_t* command_buffer =
        &context->model.command_buffers[i].record;
    if (!iree_profile_command_buffer_matches_id(command_buffer, id_filter)) {
      continue;
    }
    iree_profile_command_print_text_command_buffer(command_buffer, file);
    status = iree_profile_command_print_text_operations(context, command_buffer,
                                                        filter, file);
    if (iree_status_is_ok(status)) {
      iree_profile_command_print_text_executions(context, command_buffer, file);
    }
  }
  return status;
}

static void iree_profile_command_print_jsonl_summary(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    iree_host_size_t matched_command_operation_count, FILE* file) {
  fprintf(file, "{\"type\":\"command_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"command_buffers\":%" PRIhsz ",\"executions\":%" PRIhsz
          ",\"command_operations\":%" PRIhsz
          ",\"matched_command_operations\":%" PRIhsz
          ",\"matched_dispatches\":%" PRIu64 ",\"valid_dispatches\":%" PRIu64
          ",\"invalid_dispatches\":%" PRIu64 "}\n",
          context->model.command_buffer_count, context->command_aggregate_count,
          context->model.command_operation_count,
          matched_command_operation_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count);
}

static void iree_profile_command_print_jsonl_command_buffer(
    const iree_hal_profile_command_buffer_record_t* command_buffer,
    FILE* file) {
  fprintf(file,
          "{\"type\":\"command_buffer\",\"command_buffer_id\":%" PRIu64
          ",\"physical_device_ordinal\":%u,\"mode\":%" PRIu64
          ",\"command_categories\":%" PRIu64 ",\"queue_affinity\":%" PRIu64
          "}\n",
          command_buffer->command_buffer_id,
          command_buffer->physical_device_ordinal, command_buffer->mode,
          command_buffer->command_categories, command_buffer->queue_affinity);
}

static void iree_profile_command_print_jsonl_command_buffers(
    const iree_profile_dispatch_context_t* context, int64_t id_filter,
    FILE* file) {
  for (iree_host_size_t i = 0; i < context->model.command_buffer_count; ++i) {
    const iree_hal_profile_command_buffer_record_t* command_buffer =
        &context->model.command_buffers[i].record;
    if (!iree_profile_command_buffer_matches_id(command_buffer, id_filter)) {
      continue;
    }
    iree_profile_command_print_jsonl_command_buffer(command_buffer, file);
  }
}

static iree_status_t iree_profile_command_print_jsonl_operations(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->model.command_operation_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_profile_command_operation_record_t* operation =
        &context->model.command_operations[i].record;
    if (id_filter >= 0 && operation->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    status = iree_profile_command_print_operation_jsonl(context, operation,
                                                        filter, file);
  }
  return status;
}

static void iree_profile_command_print_jsonl_execution(
    const iree_profile_dispatch_context_t* context,
    const iree_profile_dispatch_command_aggregate_t* aggregate, FILE* file) {
  const iree_profile_projection_submission_timing_t timing =
      iree_profile_projection_calculate_submission_timing(
          context, aggregate->physical_device_ordinal,
          aggregate->earliest_start_tick, aggregate->latest_end_tick,
          aggregate->valid_count, aggregate->total_ticks);
  fprintf(file,
          "{\"type\":\"command_execution\",\"command_buffer_id\":%" PRIu64
          ",\"submission_id\":%" PRIu64
          ",\"physical_device_ordinal\":%u"
          ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64
          ",\"dispatches\":%" PRIu64 ",\"valid\":%" PRIu64
          ",\"invalid\":%" PRIu64
          ",\"span_ticks\":%.3f"
          ",\"total_dispatch_ticks\":%" PRIu64
          ",\"clock_fit_available\":%s"
          ",\"total_dispatch_time_ns_available\":%s"
          ",\"span_ns\":%.3f,\"total_dispatch_ns\":%" PRId64 "}\n",
          aggregate->command_buffer_id, aggregate->submission_id,
          aggregate->physical_device_ordinal, aggregate->queue_ordinal,
          aggregate->stream_id, aggregate->dispatch_count,
          aggregate->valid_count, aggregate->invalid_count, timing.span_ticks,
          aggregate->total_ticks, timing.has_clock_fit ? "true" : "false",
          timing.has_total_dispatch_ns ? "true" : "false",
          timing.has_clock_fit ? timing.span_ticks * timing.ns_per_tick : 0.0,
          timing.has_total_dispatch_ns ? timing.total_dispatch_ns : 0);
}

static void iree_profile_command_print_jsonl_executions(
    const iree_profile_dispatch_context_t* context, int64_t id_filter,
    FILE* file) {
  for (iree_host_size_t i = 0; i < context->command_aggregate_count; ++i) {
    const iree_profile_dispatch_command_aggregate_t* aggregate =
        &context->command_aggregates[i];
    if (id_filter >= 0 && aggregate->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    iree_profile_command_print_jsonl_execution(context, aggregate, file);
  }
}

static iree_status_t iree_profile_command_print_jsonl(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_host_size_t matched_command_operation_count = 0;
  IREE_RETURN_IF_ERROR(iree_profile_command_count_matching_operations(
      context, filter, id_filter, &matched_command_operation_count));

  iree_profile_command_print_jsonl_summary(
      context, filter, matched_command_operation_count, file);
  iree_profile_command_print_jsonl_command_buffers(context, id_filter, file);

  iree_status_t status = iree_profile_command_print_jsonl_operations(
      context, filter, id_filter, file);
  if (iree_status_is_ok(status)) {
    iree_profile_command_print_jsonl_executions(context, id_filter, file);
  }
  return status;
}

static bool iree_profile_queue_identity_matches(
    const iree_hal_profile_queue_record_t* queue,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    uint64_t stream_id) {
  return physical_device_ordinal == queue->physical_device_ordinal &&
         queue_ordinal == queue->queue_ordinal && stream_id == queue->stream_id;
}

static void iree_profile_queue_print_text_header(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    FILE* file) {
  fprintf(file, "IREE HAL profile queue summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "queues=%" PRIhsz " queue_events=%" PRIhsz
          " queue_device_events=%" PRIhsz " host_execution_events=%" PRIhsz
          " truncated_chunks=%" PRIu64 " dropped_records=%" PRIu64
          " submissions=%" PRIhsz " matched_dispatches=%" PRIu64
          " valid=%" PRIu64 " invalid=%" PRIu64 "\n",
          context->model.queue_count, context->queue_query.queue_event_count,
          context->queue_query.queue_device_event_count,
          context->queue_query.host_execution_event_count,
          context->queue_query.truncated_chunk_count,
          context->queue_query.dropped_record_count,
          context->queue_aggregate_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count);
}

static void iree_profile_queue_print_text_submission(
    const iree_profile_dispatch_context_t* context,
    const iree_profile_dispatch_queue_aggregate_t* aggregate, FILE* file) {
  const iree_profile_projection_submission_timing_t timing =
      iree_profile_projection_calculate_submission_timing(
          context, aggregate->physical_device_ordinal,
          aggregate->earliest_start_tick, aggregate->latest_end_tick,
          aggregate->valid_count, aggregate->total_ticks);
  fprintf(file,
          "  submission=%" PRIu64 " dispatches=%" PRIu64 " valid=%" PRIu64
          " invalid=%" PRIu64 " span_ticks=%.3f total_dispatch_ticks=%" PRIu64,
          aggregate->submission_id, aggregate->dispatch_count,
          aggregate->valid_count, aggregate->invalid_count, timing.span_ticks,
          aggregate->total_ticks);
  if (timing.has_total_dispatch_ns) {
    fprintf(file, " span_ns=%.3f total_dispatch_ns=%" PRId64,
            timing.span_ticks * timing.ns_per_tick, timing.total_dispatch_ns);
  }
  fputc('\n', file);
}

static void iree_profile_queue_print_text_device_event(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_device_event_t* event, FILE* file) {
  const bool is_valid = event->start_tick != 0 && event->end_tick != 0 &&
                        event->end_tick >= event->start_tick;
  const uint64_t duration_ticks =
      is_valid ? event->end_tick - event->start_tick : 0;
  iree_profile_model_clock_fit_t clock_fit;
  const bool has_clock_fit =
      iree_profile_projection_try_fit_driver_host_cpu_clock(
          context, event->physical_device_ordinal, &clock_fit);
  int64_t duration_ns = 0;
  const bool has_duration_ns = is_valid && has_clock_fit &&
                               iree_profile_model_clock_fit_scale_ticks_to_ns(
                                   &clock_fit, duration_ticks, &duration_ns);
  fprintf(file,
          "  device_event=%" PRIu64 " type=%s submission=%" PRIu64
          " command_buffer=%" PRIu64 " allocation=%" PRIu64
          " ops=%u ticks=%" PRIu64 " valid=%s",
          event->event_id, iree_profile_queue_event_type_name(event->type),
          event->submission_id, event->command_buffer_id, event->allocation_id,
          event->operation_count, duration_ticks, is_valid ? "true" : "false");
  if (has_duration_ns) {
    fprintf(file, " ns=%" PRId64, duration_ns);
  }
  fputc('\n', file);
}

static void iree_profile_queue_print_text_host_execution_event(
    const iree_hal_profile_host_execution_event_t* event, FILE* file) {
  const bool is_valid = event->start_host_time_ns >= 0 &&
                        event->end_host_time_ns >= event->start_host_time_ns;
  const int64_t duration_ns =
      is_valid ? event->end_host_time_ns - event->start_host_time_ns : 0;
  fprintf(file,
          "  host_execution_event=%" PRIu64 " type=%s submission=%" PRIu64
          " command_buffer=%" PRIu64 " allocation=%" PRIu64
          " ops=%u ns=%" PRId64 " valid=%s\n",
          event->event_id, iree_profile_queue_event_type_name(event->type),
          event->submission_id, event->command_buffer_id, event->allocation_id,
          event->operation_count, duration_ns, is_valid ? "true" : "false");
}

static void iree_profile_queue_print_text_event(
    const iree_hal_profile_queue_event_t* event, FILE* file) {
  fprintf(
      file,
      "  event=%" PRIu64 " type=%s submission=%" PRIu64
      " strategy=%s submit=%" PRId64 " ready=%" PRId64
      " waits=%u signals=%u barriers=%u ops=%u bytes=%" PRIu64 "\n",
      event->event_id, iree_profile_queue_event_type_name(event->type),
      event->submission_id,
      iree_profile_queue_dependency_strategy_name(event->dependency_strategy),
      event->host_time_ns, event->ready_host_time_ns, event->wait_count,
      event->signal_count, event->barrier_count, event->operation_count,
      event->payload_length);
}

static void iree_profile_queue_print_text_submissions_for_queue(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_record_t* queue, int64_t id_filter,
    FILE* file) {
  for (iree_host_size_t i = 0; i < context->queue_aggregate_count; ++i) {
    const iree_profile_dispatch_queue_aggregate_t* aggregate =
        &context->queue_aggregates[i];
    if (!iree_profile_queue_identity_matches(
            queue, aggregate->physical_device_ordinal, aggregate->queue_ordinal,
            aggregate->stream_id)) {
      continue;
    }
    if (id_filter >= 0 && aggregate->submission_id != (uint64_t)id_filter) {
      continue;
    }
    iree_profile_queue_print_text_submission(context, aggregate, file);
  }
}

static void iree_profile_queue_print_text_device_events_for_queue(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_record_t* queue, int64_t id_filter,
    FILE* file) {
  for (iree_host_size_t i = 0;
       i < context->queue_query.queue_device_event_count; ++i) {
    const iree_hal_profile_queue_device_event_t* event =
        &context->queue_query.queue_device_events[i].record;
    if (!iree_profile_queue_identity_matches(
            queue, event->physical_device_ordinal, event->queue_ordinal,
            event->stream_id)) {
      continue;
    }
    if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
      continue;
    }
    iree_profile_queue_print_text_device_event(context, event, file);
  }
}

static void iree_profile_queue_print_text_host_execution_events_for_queue(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_record_t* queue, int64_t id_filter,
    FILE* file) {
  for (iree_host_size_t i = 0;
       i < context->queue_query.host_execution_event_count; ++i) {
    const iree_hal_profile_host_execution_event_t* event =
        &context->queue_query.host_execution_events[i].record;
    if (!iree_profile_queue_identity_matches(
            queue, event->physical_device_ordinal, event->queue_ordinal,
            event->stream_id)) {
      continue;
    }
    if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
      continue;
    }
    iree_profile_queue_print_text_host_execution_event(event, file);
  }
}

static void iree_profile_queue_print_text_events_for_queue(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_record_t* queue, int64_t id_filter,
    FILE* file) {
  for (iree_host_size_t i = 0; i < context->queue_query.queue_event_count;
       ++i) {
    const iree_hal_profile_queue_event_t* event =
        &context->queue_query.queue_events[i].record;
    if (!iree_profile_queue_identity_matches(
            queue, event->physical_device_ordinal, event->queue_ordinal,
            event->stream_id)) {
      continue;
    }
    if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
      continue;
    }
    iree_profile_queue_print_text_event(event, file);
  }
}

static void iree_profile_queue_print_text_queue(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_record_t* queue, int64_t id_filter,
    FILE* file) {
  fprintf(file, "queue device=%u ordinal=%u stream=%" PRIu64 "\n",
          queue->physical_device_ordinal, queue->queue_ordinal,
          queue->stream_id);

  iree_profile_queue_print_text_submissions_for_queue(context, queue, id_filter,
                                                      file);
  iree_profile_queue_print_text_device_events_for_queue(context, queue,
                                                        id_filter, file);
  iree_profile_queue_print_text_host_execution_events_for_queue(
      context, queue, id_filter, file);
  iree_profile_queue_print_text_events_for_queue(context, queue, id_filter,
                                                 file);
}

static iree_status_t iree_profile_queue_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_profile_queue_print_text_header(context, filter, file);
  for (iree_host_size_t i = 0; i < context->model.queue_count; ++i) {
    const iree_hal_profile_queue_record_t* queue =
        &context->model.queues[i].record;
    iree_profile_queue_print_text_queue(context, queue, id_filter, file);
  }
  return iree_ok_status();
}

static void iree_profile_queue_print_jsonl_summary(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    FILE* file) {
  fprintf(file, "{\"type\":\"queue_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"queues\":%" PRIhsz ",\"queue_events\":%" PRIhsz
          ",\"queue_device_events\":%" PRIhsz
          ",\"host_execution_events\":%" PRIhsz ",\"truncated_chunks\":%" PRIu64
          ",\"dropped_records\":%" PRIu64 ",\"submissions\":%" PRIhsz
          ",\"matched_dispatches\":%" PRIu64 ",\"valid_dispatches\":%" PRIu64
          ",\"invalid_dispatches\":%" PRIu64 "}\n",
          context->model.queue_count, context->queue_query.queue_event_count,
          context->queue_query.queue_device_event_count,
          context->queue_query.host_execution_event_count,
          context->queue_query.truncated_chunk_count,
          context->queue_query.dropped_record_count,
          context->queue_aggregate_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count);
}

static void iree_profile_queue_print_jsonl_queues(
    const iree_profile_dispatch_context_t* context, FILE* file) {
  for (iree_host_size_t i = 0; i < context->model.queue_count; ++i) {
    const iree_hal_profile_queue_record_t* queue =
        &context->model.queues[i].record;
    fprintf(file,
            "{\"type\":\"queue\",\"physical_device_ordinal\":%u"
            ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64 "}\n",
            queue->physical_device_ordinal, queue->queue_ordinal,
            queue->stream_id);
  }
}

static void iree_profile_queue_print_jsonl_submissions(
    const iree_profile_dispatch_context_t* context, int64_t id_filter,
    FILE* file) {
  for (iree_host_size_t i = 0; i < context->queue_aggregate_count; ++i) {
    const iree_profile_dispatch_queue_aggregate_t* aggregate =
        &context->queue_aggregates[i];
    if (id_filter >= 0 && aggregate->submission_id != (uint64_t)id_filter) {
      continue;
    }
    const iree_profile_projection_submission_timing_t timing =
        iree_profile_projection_calculate_submission_timing(
            context, aggregate->physical_device_ordinal,
            aggregate->earliest_start_tick, aggregate->latest_end_tick,
            aggregate->valid_count, aggregate->total_ticks);
    fprintf(file,
            "{\"type\":\"queue_submission\",\"submission_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
            ",\"stream_id\":%" PRIu64 ",\"dispatches\":%" PRIu64
            ",\"valid\":%" PRIu64 ",\"invalid\":%" PRIu64
            ",\"span_ticks\":%.3f,\"total_dispatch_ticks\":%" PRIu64
            ",\"clock_fit_available\":%s"
            ",\"total_dispatch_time_ns_available\":%s"
            ",\"span_ns\":%.3f,\"total_dispatch_ns\":%" PRId64 "}\n",
            aggregate->submission_id, aggregate->physical_device_ordinal,
            aggregate->queue_ordinal, aggregate->stream_id,
            aggregate->dispatch_count, aggregate->valid_count,
            aggregate->invalid_count, timing.span_ticks, aggregate->total_ticks,
            timing.has_clock_fit ? "true" : "false",
            timing.has_total_dispatch_ns ? "true" : "false",
            timing.has_clock_fit ? timing.span_ticks * timing.ns_per_tick : 0.0,
            timing.has_total_dispatch_ns ? timing.total_dispatch_ns : 0);
  }
}

static void iree_profile_queue_print_jsonl_device_events(
    const iree_profile_dispatch_context_t* context, int64_t id_filter,
    FILE* file) {
  for (iree_host_size_t i = 0;
       i < context->queue_query.queue_device_event_count; ++i) {
    const iree_hal_profile_queue_device_event_t* event =
        &context->queue_query.queue_device_events[i].record;
    if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
      continue;
    }
    const bool is_valid = event->start_tick != 0 && event->end_tick != 0 &&
                          event->end_tick >= event->start_tick;
    const uint64_t duration_ticks =
        is_valid ? event->end_tick - event->start_tick : 0;
    iree_profile_model_clock_fit_t clock_fit;
    const bool has_clock_fit =
        iree_profile_projection_try_fit_driver_host_cpu_clock(
            context, event->physical_device_ordinal, &clock_fit);
    int64_t duration_ns = 0;
    const bool has_duration_ns = is_valid && has_clock_fit &&
                                 iree_profile_model_clock_fit_scale_ticks_to_ns(
                                     &clock_fit, duration_ticks, &duration_ns);
    fprintf(file,
            "{\"type\":\"queue_device_event\",\"event_id\":%" PRIu64
            ",\"op\":\"%s\",\"flags\":%u,\"submission_id\":%" PRIu64
            ",\"command_buffer_id\":%" PRIu64 ",\"allocation_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
            ",\"stream_id\":%" PRIu64
            ",\"operation_count\":%u"
            ",\"payload_length\":%" PRIu64 ",\"start_tick\":%" PRIu64
            ",\"end_tick\":%" PRIu64 ",\"duration_ticks\":%" PRIu64
            ",\"valid\":%s"
            ",\"clock_fit_available\":%s,\"duration_ns\":%" PRId64 "}\n",
            event->event_id, iree_profile_queue_event_type_name(event->type),
            event->flags, event->submission_id, event->command_buffer_id,
            event->allocation_id, event->physical_device_ordinal,
            event->queue_ordinal, event->stream_id, event->operation_count,
            event->payload_length, event->start_tick, event->end_tick,
            duration_ticks, is_valid ? "true" : "false",
            has_clock_fit ? "true" : "false",
            has_duration_ns ? duration_ns : 0);
  }
}

static void iree_profile_queue_print_jsonl_host_execution_events(
    const iree_profile_dispatch_context_t* context, int64_t id_filter,
    FILE* file) {
  for (iree_host_size_t i = 0;
       i < context->queue_query.host_execution_event_count; ++i) {
    const iree_hal_profile_host_execution_event_t* event =
        &context->queue_query.host_execution_events[i].record;
    if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
      continue;
    }
    const bool is_valid = event->start_host_time_ns >= 0 &&
                          event->end_host_time_ns >= event->start_host_time_ns;
    const int64_t duration_ns =
        is_valid ? event->end_host_time_ns - event->start_host_time_ns : 0;
    fprintf(file,
            "{\"type\":\"host_execution_event\",\"event_id\":%" PRIu64
            ",\"op\":\"%s\",\"flags\":%u,\"status_code\":%u"
            ",\"submission_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
            ",\"command_index\":%u,\"executable_id\":%" PRIu64
            ",\"export_ordinal\":%u,\"allocation_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
            ",\"stream_id\":%" PRIu64
            ",\"operation_count\":%u"
            ",\"payload_length\":%" PRIu64
            ",\"host_time_domain\":\"iree_host_time_ns\""
            ",\"start_host_time_ns\":%" PRId64 ",\"end_host_time_ns\":%" PRId64
            ",\"duration_ns\":%" PRId64 ",\"valid\":%s}\n",
            event->event_id, iree_profile_queue_event_type_name(event->type),
            event->flags, event->status_code, event->submission_id,
            event->command_buffer_id, event->command_index,
            event->executable_id, event->export_ordinal, event->allocation_id,
            event->physical_device_ordinal, event->queue_ordinal,
            event->stream_id, event->operation_count, event->payload_length,
            event->start_host_time_ns, event->end_host_time_ns, duration_ns,
            is_valid ? "true" : "false");
  }
}

static void iree_profile_queue_print_jsonl_events(
    const iree_profile_dispatch_context_t* context, int64_t id_filter,
    FILE* file) {
  for (iree_host_size_t i = 0; i < context->queue_query.queue_event_count;
       ++i) {
    const iree_hal_profile_queue_event_t* event =
        &context->queue_query.queue_events[i].record;
    if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
      continue;
    }
    fprintf(
        file,
        "{\"type\":\"queue_event\",\"event_id\":%" PRIu64
        ",\"op\":\"%s\",\"flags\":%u,\"dependency_strategy\":\"%s\""
        ",\"submission_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
        ",\"allocation_id\":%" PRIu64
        ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
        ",\"stream_id\":%" PRIu64 ",\"host_time_ns\":%" PRId64
        ",\"ready_host_time_ns\":%" PRId64
        ",\"host_time_domain\":\"iree_host_time_ns\""
        ",\"wait_count\":%u,\"signal_count\":%u,\"barrier_count\":%u"
        ",\"operation_count\":%u,\"payload_length\":%" PRIu64 "}\n",
        event->event_id, iree_profile_queue_event_type_name(event->type),
        event->flags,
        iree_profile_queue_dependency_strategy_name(event->dependency_strategy),
        event->submission_id, event->command_buffer_id, event->allocation_id,
        event->physical_device_ordinal, event->queue_ordinal, event->stream_id,
        event->host_time_ns, event->ready_host_time_ns, event->wait_count,
        event->signal_count, event->barrier_count, event->operation_count,
        event->payload_length);
  }
}

static iree_status_t iree_profile_queue_print_jsonl(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_profile_queue_print_jsonl_summary(context, filter, file);
  iree_profile_queue_print_jsonl_queues(context, file);
  iree_profile_queue_print_jsonl_submissions(context, id_filter, file);
  iree_profile_queue_print_jsonl_device_events(context, id_filter, file);
  iree_profile_queue_print_jsonl_host_execution_events(context, id_filter,
                                                       file);
  iree_profile_queue_print_jsonl_events(context, id_filter, file);
  return iree_ok_status();
}

typedef struct iree_profile_projection_parse_context_t {
  // Dispatch aggregation and shared model state for this projection.
  iree_profile_dispatch_context_t* dispatch_context;
  // Optional glob filter applied to projected keys.
  iree_string_view_t filter;
  // Projection selected by the command entry point.
  iree_profile_projection_mode_t projection_mode;
  // Optional entity identifier filter, or -1 when disabled.
  int64_t id_filter;
  // True when matched dispatches should update aggregate arrays.
  bool aggregate_events;
  // Optional callback receiving matched raw dispatch events.
  iree_profile_dispatch_event_callback_t event_callback;
} iree_profile_projection_parse_context_t;

static iree_status_t iree_profile_projection_metadata_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_projection_parse_context_t* context =
      (iree_profile_projection_parse_context_t*)user_data;
  return iree_profile_model_process_metadata_record(
      &context->dispatch_context->model, record);
}

static iree_status_t iree_profile_projection_event_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_projection_parse_context_t* context =
      (iree_profile_projection_parse_context_t*)user_data;
  return iree_profile_dispatch_process_events_record(
      context->dispatch_context, record, context->filter,
      context->projection_mode, context->id_filter, context->aggregate_events,
      context->event_callback);
}

static iree_status_t iree_profile_projection_print(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool emit_events, bool is_text, FILE* file) {
  switch (projection_mode) {
    case IREE_PROFILE_PROJECTION_MODE_DISPATCH:
      if (is_text) {
        return iree_profile_dispatch_print_text(context, filter, file);
      }
      iree_profile_dispatch_print_jsonl_summary(context, filter, emit_events,
                                                file);
      return emit_events
                 ? iree_ok_status()
                 : iree_profile_dispatch_print_jsonl_aggregates(context, file);
    case IREE_PROFILE_PROJECTION_MODE_EXECUTABLE:
      if (emit_events) {
        iree_profile_dispatch_print_jsonl_summary(context, filter, emit_events,
                                                  file);
        return iree_ok_status();
      }
      return is_text ? iree_profile_executable_print_text(context, filter,
                                                          id_filter, file)
                     : iree_profile_executable_print_jsonl(context, filter,
                                                           id_filter, file);
    case IREE_PROFILE_PROJECTION_MODE_COMMAND:
      if (emit_events) {
        iree_profile_dispatch_print_jsonl_summary(context, filter, emit_events,
                                                  file);
        return iree_ok_status();
      }
      return is_text ? iree_profile_command_print_text(context, filter,
                                                       id_filter, file)
                     : iree_profile_command_print_jsonl(context, filter,
                                                        id_filter, file);
    case IREE_PROFILE_PROJECTION_MODE_QUEUE:
      if (emit_events) {
        iree_profile_dispatch_print_jsonl_summary(context, filter, emit_events,
                                                  file);
        return iree_ok_status();
      }
      return is_text ? iree_profile_queue_print_text(context, filter, id_filter,
                                                     file)
                     : iree_profile_queue_print_jsonl(context, filter,
                                                      id_filter, file);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported profile projection mode %d",
                              (int)projection_mode);
  }
}

iree_status_t iree_profile_projection_file(
    iree_string_view_t path, iree_string_view_t format,
    iree_string_view_t filter, iree_profile_projection_mode_t projection_mode,
    int64_t id_filter, bool emit_events, FILE* file,
    iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }
  if (emit_events && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--dispatch_events requires --format=jsonl");
  }

  iree_profile_file_t profile_file;
  IREE_RETURN_IF_ERROR(
      iree_profile_file_open(path, host_allocator, &profile_file));

  iree_profile_dispatch_context_t context;
  iree_profile_dispatch_context_initialize(host_allocator, &context);
  const iree_profile_dispatch_event_callback_t event_callback = {
      .fn = emit_events ? iree_profile_projection_emit_dispatch_event : NULL,
      .user_data = file,
  };
  iree_profile_projection_parse_context_t parse_context = {
      .dispatch_context = &context,
      .filter = filter,
      .projection_mode = projection_mode,
      .id_filter = id_filter,
      .aggregate_events = !emit_events,
      .event_callback = event_callback,
  };
  iree_profile_file_record_callback_t record_callback = {
      .fn = iree_profile_projection_metadata_record,
      .user_data = &parse_context,
  };
  iree_status_t status =
      iree_profile_file_for_each_record(&profile_file, record_callback);
  if (iree_status_is_ok(status)) {
    record_callback.fn = iree_profile_projection_event_record;
    status = iree_profile_file_for_each_record(&profile_file, record_callback);
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_profile_projection_print(&context, filter, projection_mode,
                                      id_filter, emit_events, is_text, file);
  }

  iree_profile_dispatch_context_deinitialize(&context);
  iree_profile_file_close(&profile_file);
  return status;
}
