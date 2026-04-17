// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/internal.h"

static bool iree_profile_command_operation_filter_matches(
    iree_string_view_t operation_name, iree_string_view_t key,
    iree_string_view_t filter) {
  return iree_profile_dispatch_key_matches(operation_name, filter) ||
         iree_profile_dispatch_key_matches(key, filter);
}

static iree_status_t iree_profile_command_count_matching_operations(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, iree_host_size_t* out_operation_count) {
  *out_operation_count = 0;
  iree_host_size_t operation_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->command_operation_count && iree_status_is_ok(status); ++i) {
    const iree_hal_profile_command_operation_record_t* operation =
        &context->command_operations[i].record;
    if (id_filter >= 0 && operation->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    const char* operation_name =
        iree_profile_command_operation_type_name(operation->type);
    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_command_operation_resolve_key(
        context, operation, numeric_buffer, sizeof(numeric_buffer), &key);
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
  IREE_RETURN_IF_ERROR(iree_profile_command_operation_resolve_key(
      context, operation, numeric_buffer, sizeof(numeric_buffer), &key));
  if (!iree_profile_command_operation_filter_matches(
          iree_make_cstring_view(operation_name), key, filter)) {
    return iree_ok_status();
  }

  fprintf(file, "    command[%u]: op=%s block=%u local=%u flags=0x%x",
          operation->command_index, operation_name, operation->block_ordinal,
          operation->block_command_ordinal, operation->flags);
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
  IREE_RETURN_IF_ERROR(iree_profile_command_operation_resolve_key(
      context, operation, numeric_buffer, sizeof(numeric_buffer), &key));
  if (!iree_profile_command_operation_filter_matches(
          iree_make_cstring_view(operation_name), key, filter)) {
    return iree_ok_status();
  }

  fprintf(file,
          "{\"type\":\"command_operation\",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u,\"op\":\"%s\",\"flags\":%u"
          ",\"block_ordinal\":%u,\"block_command_ordinal\":%u",
          operation->command_buffer_id, operation->command_index,
          operation_name, operation->flags, operation->block_ordinal,
          operation->block_command_ordinal);
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

static iree_status_t iree_profile_dispatch_print_text(
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

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_dispatch_aggregate_t* aggregate =
        &context->aggregates[i];
    const iree_profile_dispatch_device_t* device = NULL;
    for (iree_host_size_t j = 0; j < context->device_count; ++j) {
      if (context->devices[j].physical_device_ordinal ==
          aggregate->physical_device_ordinal) {
        device = &context->devices[j];
        break;
      }
    }
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_dispatch_resolve_key(
        context, aggregate->physical_device_ordinal, aggregate->executable_id,
        aggregate->export_ordinal, numeric_buffer, sizeof(numeric_buffer),
        &key);
    if (iree_status_is_ok(status)) {
      const double variance_ticks =
          aggregate->valid_count > 1
              ? aggregate->m2_ticks / (double)(aggregate->valid_count - 1)
              : 0.0;
      const double stddev_ticks = iree_profile_sqrt_f64(variance_ticks);
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
                " total=%.3f\n",
                aggregate->minimum_ticks, aggregate->mean_ticks, stddev_ticks,
                aggregate->maximum_ticks, aggregate->total_ticks);
        if (has_clock_fit) {
          fprintf(file,
                  "    time_ns: min=%.3f avg=%.3f stddev=%.3f max=%.3f"
                  " total=%.3f\n",
                  (double)aggregate->minimum_ticks * ns_per_tick,
                  aggregate->mean_ticks * ns_per_tick,
                  stddev_ticks * ns_per_tick,
                  (double)aggregate->maximum_ticks * ns_per_tick,
                  aggregate->total_ticks * ns_per_tick);
        } else {
          fprintf(file, "    time_ns: unavailable\n");
        }
      }
      fprintf(
          file,
          "    last_geometry: workgroup_count=%ux%ux%u"
          " workgroup_size=%ux%ux%u\n",
          aggregate->last_workgroup_count[0],
          aggregate->last_workgroup_count[1],
          aggregate->last_workgroup_count[2], aggregate->last_workgroup_size[0],
          aggregate->last_workgroup_size[1], aggregate->last_workgroup_size[2]);
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

static iree_status_t iree_profile_dispatch_print_jsonl_aggregates(
    const iree_profile_dispatch_context_t* context, FILE* file) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_dispatch_aggregate_t* aggregate =
        &context->aggregates[i];
    const iree_profile_dispatch_device_t* device = NULL;
    for (iree_host_size_t j = 0; j < context->device_count; ++j) {
      if (context->devices[j].physical_device_ordinal ==
          aggregate->physical_device_ordinal) {
        device = &context->devices[j];
        break;
      }
    }
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    (void)tick_frequency_hz;

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_dispatch_resolve_key(
        context, aggregate->physical_device_ordinal, aggregate->executable_id,
        aggregate->export_ordinal, numeric_buffer, sizeof(numeric_buffer),
        &key);
    if (iree_status_is_ok(status)) {
      const double variance_ticks =
          aggregate->valid_count > 1
              ? aggregate->m2_ticks / (double)(aggregate->valid_count - 1)
              : 0.0;
      const double stddev_ticks = iree_profile_sqrt_f64(variance_ticks);
      fprintf(file,
              "{\"type\":\"dispatch_group\",\"physical_device_ordinal\":%u"
              ",\"executable_id\":%" PRIu64 ",\"export_ordinal\":%u,\"key\":",
              aggregate->physical_device_ordinal, aggregate->executable_id,
              aggregate->export_ordinal);
      iree_profile_fprint_json_string(file, key);
      fprintf(file,
              ",\"count\":%" PRIu64 ",\"valid\":%" PRIu64
              ",\"invalid\":%" PRIu64 ",\"min_ticks\":%" PRIu64
              ",\"avg_ticks\":%.3f,\"stddev_ticks\":%.3f"
              ",\"max_ticks\":%" PRIu64 ",\"total_ticks\":%.3f",
              aggregate->dispatch_count, aggregate->valid_count,
              aggregate->invalid_count,
              aggregate->valid_count ? aggregate->minimum_ticks : 0,
              aggregate->valid_count ? aggregate->mean_ticks : 0.0,
              stddev_ticks,
              aggregate->valid_count ? aggregate->maximum_ticks : 0,
              aggregate->total_ticks);
      fprintf(file, ",\"clock_fit_available\":%s",
              has_clock_fit ? "true" : "false");
      fprintf(file,
              ",\"min_ns\":%.3f,\"avg_ns\":%.3f,\"stddev_ns\":%.3f"
              ",\"max_ns\":%.3f,\"total_ns\":%.3f",
              has_clock_fit && aggregate->valid_count
                  ? (double)aggregate->minimum_ticks * ns_per_tick
                  : 0.0,
              has_clock_fit && aggregate->valid_count
                  ? aggregate->mean_ticks * ns_per_tick
                  : 0.0,
              has_clock_fit ? stddev_ticks * ns_per_tick : 0.0,
              has_clock_fit && aggregate->valid_count
                  ? (double)aggregate->maximum_ticks * ns_per_tick
                  : 0.0,
              has_clock_fit ? aggregate->total_ticks * ns_per_tick : 0.0);
      fprintf(
          file,
          ",\"last_workgroup_count\":[%u,%u,%u]"
          ",\"last_workgroup_size\":[%u,%u,%u]}\n",
          aggregate->last_workgroup_count[0],
          aggregate->last_workgroup_count[1],
          aggregate->last_workgroup_count[2], aggregate->last_workgroup_size[0],
          aggregate->last_workgroup_size[1], aggregate->last_workgroup_size[2]);
    }
  }
  return status;
}

static iree_status_t iree_profile_executable_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "IREE HAL profile executable summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "executables=%" PRIhsz " exports=%" PRIhsz
          " matched_dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
          "\n",
          context->executable_count, context->export_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  for (iree_host_size_t i = 0; i < context->executable_count; ++i) {
    const iree_hal_profile_executable_record_t* executable =
        &context->executables[i].record;
    if (id_filter >= 0 && executable->executable_id != (uint64_t)id_filter) {
      continue;
    }
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
    for (iree_host_size_t j = 0; j < context->export_count; ++j) {
      const iree_profile_dispatch_export_t* export_info = &context->exports[j];
      if (export_info->executable_id != executable->executable_id) continue;

      char numeric_buffer[128];
      iree_string_view_t key = iree_profile_dispatch_format_export_key(
          export_info, UINT32_MAX, numeric_buffer, sizeof(numeric_buffer));
      if (!iree_profile_dispatch_key_matches(key, filter)) continue;

      const bool has_pipeline_hash = iree_all_bits_set(
          export_info->flags,
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
      bool has_aggregate = false;
      for (iree_host_size_t k = 0; k < context->aggregate_count; ++k) {
        const iree_profile_dispatch_aggregate_t* aggregate =
            &context->aggregates[k];
        if (aggregate->executable_id != export_info->executable_id ||
            aggregate->export_ordinal != export_info->export_ordinal) {
          continue;
        }
        has_aggregate = true;
        const iree_profile_dispatch_device_t* device =
            iree_profile_dispatch_find_device(
                context, aggregate->physical_device_ordinal);
        double ns_per_tick = 0.0;
        double tick_frequency_hz = 0.0;
        const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
            device, &ns_per_tick, &tick_frequency_hz);
        const double average_ticks =
            aggregate->valid_count
                ? aggregate->total_ticks / (double)aggregate->valid_count
                : 0.0;
        fprintf(file,
                "    device=%u dispatches=%" PRIu64 " valid=%" PRIu64
                " invalid=%" PRIu64 " ticks[min/avg/max/total]=%" PRIu64
                "/%.3f/%" PRIu64 "/%.3f",
                aggregate->physical_device_ordinal, aggregate->dispatch_count,
                aggregate->valid_count, aggregate->invalid_count,
                aggregate->valid_count ? aggregate->minimum_ticks : 0,
                average_ticks,
                aggregate->valid_count ? aggregate->maximum_ticks : 0,
                aggregate->total_ticks);
        if (has_clock_fit) {
          fprintf(file, " ns[min/avg/max/total]=%.3f/%.3f/%.3f/%.3f",
                  aggregate->valid_count
                      ? (double)aggregate->minimum_ticks * ns_per_tick
                      : 0.0,
                  average_ticks * ns_per_tick,
                  aggregate->valid_count
                      ? (double)aggregate->maximum_ticks * ns_per_tick
                      : 0.0,
                  aggregate->total_ticks * ns_per_tick);
        }
        fputc('\n', file);
      }
      if (!has_aggregate) {
        fprintf(file, "    dispatches=0\n");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_executable_print_jsonl(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "{\"type\":\"executable_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"executables\":%" PRIhsz ",\"exports\":%" PRIhsz
          ",\"matched_dispatches\":%" PRIu64 ",\"valid_dispatches\":%" PRIu64
          ",\"invalid_dispatches\":%" PRIu64 "}\n",
          context->executable_count, context->export_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  for (iree_host_size_t i = 0; i < context->executable_count; ++i) {
    const iree_hal_profile_executable_record_t* executable =
        &context->executables[i].record;
    if (id_filter >= 0 && executable->executable_id != (uint64_t)id_filter) {
      continue;
    }
    const bool has_code_object_hash = iree_all_bits_set(
        executable->flags, IREE_HAL_PROFILE_EXECUTABLE_FLAG_CODE_OBJECT_HASH);
    fprintf(file,
            "{\"type\":\"executable\",\"executable_id\":%" PRIu64
            ",\"flags\":%u,\"export_count\":%u"
            ",\"code_object_hash_present\":%s,\"code_object_hash\":",
            executable->executable_id, executable->flags,
            executable->export_count, has_code_object_hash ? "true" : "false");
    if (has_code_object_hash) {
      fputc('"', file);
      iree_profile_fprint_hash_hex(file, executable->code_object_hash);
      fputc('"', file);
    } else {
      fprintf(file, "null");
    }
    fputs("}\n", file);
    for (iree_host_size_t j = 0; j < context->export_count; ++j) {
      const iree_profile_dispatch_export_t* export_info = &context->exports[j];
      if (export_info->executable_id != executable->executable_id) continue;

      char numeric_buffer[128];
      iree_string_view_t key = iree_profile_dispatch_format_export_key(
          export_info, UINT32_MAX, numeric_buffer, sizeof(numeric_buffer));
      if (!iree_profile_dispatch_key_matches(key, filter)) continue;

      const bool has_pipeline_hash = iree_all_bits_set(
          export_info->flags,
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
      if (has_pipeline_hash) {
        fputc('"', file);
        iree_profile_fprint_hash_hex(file, export_info->pipeline_hash);
        fputc('"', file);
      } else {
        fprintf(file, "null");
      }
      fputs("}\n", file);
      for (iree_host_size_t k = 0; k < context->aggregate_count; ++k) {
        const iree_profile_dispatch_aggregate_t* aggregate =
            &context->aggregates[k];
        if (aggregate->executable_id != export_info->executable_id ||
            aggregate->export_ordinal != export_info->export_ordinal) {
          continue;
        }
        const iree_profile_dispatch_device_t* device =
            iree_profile_dispatch_find_device(
                context, aggregate->physical_device_ordinal);
        double ns_per_tick = 0.0;
        double tick_frequency_hz = 0.0;
        const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
            device, &ns_per_tick, &tick_frequency_hz);
        const double average_ticks =
            aggregate->valid_count
                ? aggregate->total_ticks / (double)aggregate->valid_count
                : 0.0;
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
                ",\"total_ticks\":%.3f,\"clock_fit_available\":%s"
                ",\"min_ns\":%.3f,\"avg_ns\":%.3f,\"max_ns\":%.3f"
                ",\"total_ns\":%.3f}\n",
                aggregate->dispatch_count, aggregate->valid_count,
                aggregate->invalid_count,
                aggregate->valid_count ? aggregate->minimum_ticks : 0,
                average_ticks,
                aggregate->valid_count ? aggregate->maximum_ticks : 0,
                aggregate->total_ticks, has_clock_fit ? "true" : "false",
                has_clock_fit && aggregate->valid_count
                    ? (double)aggregate->minimum_ticks * ns_per_tick
                    : 0.0,
                has_clock_fit ? average_ticks * ns_per_tick : 0.0,
                has_clock_fit && aggregate->valid_count
                    ? (double)aggregate->maximum_ticks * ns_per_tick
                    : 0.0,
                has_clock_fit ? aggregate->total_ticks * ns_per_tick : 0.0);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_command_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_host_size_t matched_command_operation_count = 0;
  IREE_RETURN_IF_ERROR(iree_profile_command_count_matching_operations(
      context, filter, id_filter, &matched_command_operation_count));

  fprintf(file, "IREE HAL profile command-buffer summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "command_buffers=%" PRIhsz " executions=%" PRIhsz
          " command_operations=%" PRIhsz " matched_command_operations=%" PRIhsz
          " matched_dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
          "\n",
          context->command_buffer_count, context->command_aggregate_count,
          context->command_operation_count, matched_command_operation_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < context->command_buffer_count; ++i) {
    const iree_hal_profile_command_buffer_record_t* command_buffer =
        &context->command_buffers[i].record;
    if (id_filter >= 0 &&
        command_buffer->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    fprintf(file,
            "command_buffer %" PRIu64 ": device=%u mode=%" PRIu64
            " categories=%" PRIu64 " queue_affinity=%" PRIu64 "\n",
            command_buffer->command_buffer_id,
            command_buffer->physical_device_ordinal, command_buffer->mode,
            command_buffer->command_categories, command_buffer->queue_affinity);
    for (iree_host_size_t j = 0;
         j < context->command_operation_count && iree_status_is_ok(status);
         ++j) {
      const iree_hal_profile_command_operation_record_t* operation =
          &context->command_operations[j].record;
      if (operation->command_buffer_id != command_buffer->command_buffer_id) {
        continue;
      }
      status = iree_profile_command_print_operation_text(context, operation,
                                                         filter, file);
    }
    for (iree_host_size_t j = 0;
         j < context->command_aggregate_count && iree_status_is_ok(status);
         ++j) {
      const iree_profile_dispatch_command_aggregate_t* aggregate =
          &context->command_aggregates[j];
      if (aggregate->command_buffer_id != command_buffer->command_buffer_id) {
        continue;
      }
      const iree_profile_dispatch_device_t* device =
          iree_profile_dispatch_find_device(context,
                                            aggregate->physical_device_ordinal);
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      const double span_ticks = iree_profile_dispatch_span_ticks(
          aggregate->earliest_start_tick, aggregate->latest_end_tick);
      fprintf(file,
              "  submission=%" PRIu64 " queue=%u stream=%" PRIu64
              " dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
              " span_ticks=%.3f total_dispatch_ticks=%.3f",
              aggregate->submission_id, aggregate->queue_ordinal,
              aggregate->stream_id, aggregate->dispatch_count,
              aggregate->valid_count, aggregate->invalid_count, span_ticks,
              aggregate->total_ticks);
      if (has_clock_fit) {
        fprintf(file, " span_ns=%.3f total_dispatch_ns=%.3f",
                span_ticks * ns_per_tick, aggregate->total_ticks * ns_per_tick);
      }
      fputc('\n', file);
    }
  }
  return status;
}

static iree_status_t iree_profile_command_print_jsonl(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_host_size_t matched_command_operation_count = 0;
  IREE_RETURN_IF_ERROR(iree_profile_command_count_matching_operations(
      context, filter, id_filter, &matched_command_operation_count));

  fprintf(file, "{\"type\":\"command_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"command_buffers\":%" PRIhsz ",\"executions\":%" PRIhsz
          ",\"command_operations\":%" PRIhsz
          ",\"matched_command_operations\":%" PRIhsz
          ",\"matched_dispatches\":%" PRIu64 ",\"valid_dispatches\":%" PRIu64
          ",\"invalid_dispatches\":%" PRIu64 "}\n",
          context->command_buffer_count, context->command_aggregate_count,
          context->command_operation_count, matched_command_operation_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < context->command_buffer_count; ++i) {
    const iree_hal_profile_command_buffer_record_t* command_buffer =
        &context->command_buffers[i].record;
    if (id_filter >= 0 &&
        command_buffer->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    fprintf(file,
            "{\"type\":\"command_buffer\",\"command_buffer_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u,\"mode\":%" PRIu64
            ",\"command_categories\":%" PRIu64 ",\"queue_affinity\":%" PRIu64
            "}\n",
            command_buffer->command_buffer_id,
            command_buffer->physical_device_ordinal, command_buffer->mode,
            command_buffer->command_categories, command_buffer->queue_affinity);
  }
  for (iree_host_size_t i = 0;
       i < context->command_operation_count && iree_status_is_ok(status); ++i) {
    const iree_hal_profile_command_operation_record_t* operation =
        &context->command_operations[i].record;
    if (id_filter >= 0 && operation->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    status = iree_profile_command_print_operation_jsonl(context, operation,
                                                        filter, file);
  }
  for (iree_host_size_t i = 0;
       i < context->command_aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_dispatch_command_aggregate_t* aggregate =
        &context->command_aggregates[i];
    if (id_filter >= 0 && aggregate->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    const iree_profile_dispatch_device_t* device =
        iree_profile_dispatch_find_device(context,
                                          aggregate->physical_device_ordinal);
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    const double span_ticks = iree_profile_dispatch_span_ticks(
        aggregate->earliest_start_tick, aggregate->latest_end_tick);
    fprintf(file,
            "{\"type\":\"command_execution\",\"command_buffer_id\":%" PRIu64
            ",\"submission_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u"
            ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64
            ",\"dispatches\":%" PRIu64 ",\"valid\":%" PRIu64
            ",\"invalid\":%" PRIu64
            ",\"span_ticks\":%.3f"
            ",\"total_dispatch_ticks\":%.3f,\"clock_fit_available\":%s"
            ",\"span_ns\":%.3f,\"total_dispatch_ns\":%.3f}\n",
            aggregate->command_buffer_id, aggregate->submission_id,
            aggregate->physical_device_ordinal, aggregate->queue_ordinal,
            aggregate->stream_id, aggregate->dispatch_count,
            aggregate->valid_count, aggregate->invalid_count, span_ticks,
            aggregate->total_ticks, has_clock_fit ? "true" : "false",
            has_clock_fit ? span_ticks * ns_per_tick : 0.0,
            has_clock_fit ? aggregate->total_ticks * ns_per_tick : 0.0);
  }
  return status;
}

static iree_status_t iree_profile_queue_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "IREE HAL profile queue summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "queues=%" PRIhsz " queue_events=%" PRIhsz
          " queue_device_events=%" PRIhsz " submissions=%" PRIhsz
          " matched_dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
          "\n",
          context->queue_count, context->queue_event_count,
          context->queue_device_event_count, context->queue_aggregate_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  for (iree_host_size_t i = 0; i < context->queue_count; ++i) {
    const iree_hal_profile_queue_record_t* queue = &context->queues[i].record;
    fprintf(file, "queue device=%u ordinal=%u stream=%" PRIu64 "\n",
            queue->physical_device_ordinal, queue->queue_ordinal,
            queue->stream_id);
    for (iree_host_size_t j = 0; j < context->queue_aggregate_count; ++j) {
      const iree_profile_dispatch_queue_aggregate_t* aggregate =
          &context->queue_aggregates[j];
      if (aggregate->physical_device_ordinal !=
              queue->physical_device_ordinal ||
          aggregate->queue_ordinal != queue->queue_ordinal ||
          aggregate->stream_id != queue->stream_id) {
        continue;
      }
      if (id_filter >= 0 && aggregate->submission_id != (uint64_t)id_filter) {
        continue;
      }
      const iree_profile_dispatch_device_t* device =
          iree_profile_dispatch_find_device(context,
                                            aggregate->physical_device_ordinal);
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      const double span_ticks = iree_profile_dispatch_span_ticks(
          aggregate->earliest_start_tick, aggregate->latest_end_tick);
      fprintf(file,
              "  submission=%" PRIu64 " dispatches=%" PRIu64 " valid=%" PRIu64
              " invalid=%" PRIu64 " span_ticks=%.3f total_dispatch_ticks=%.3f",
              aggregate->submission_id, aggregate->dispatch_count,
              aggregate->valid_count, aggregate->invalid_count, span_ticks,
              aggregate->total_ticks);
      if (has_clock_fit) {
        fprintf(file, " span_ns=%.3f total_dispatch_ns=%.3f",
                span_ticks * ns_per_tick, aggregate->total_ticks * ns_per_tick);
      }
      fputc('\n', file);
    }
    for (iree_host_size_t j = 0; j < context->queue_device_event_count; ++j) {
      const iree_hal_profile_queue_device_event_t* event =
          &context->queue_device_events[j].record;
      if (event->physical_device_ordinal != queue->physical_device_ordinal ||
          event->queue_ordinal != queue->queue_ordinal ||
          event->stream_id != queue->stream_id) {
        continue;
      }
      if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
        continue;
      }
      const bool is_valid = event->start_tick != 0 && event->end_tick != 0 &&
                            event->end_tick >= event->start_tick;
      const uint64_t duration_ticks =
          is_valid ? event->end_tick - event->start_tick : 0;
      const iree_profile_dispatch_device_t* device =
          iree_profile_dispatch_find_device(context,
                                            event->physical_device_ordinal);
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      (void)tick_frequency_hz;
      fprintf(file,
              "  device_event=%" PRIu64 " type=%s submission=%" PRIu64
              " command_buffer=%" PRIu64 " allocation=%" PRIu64
              " ops=%u ticks=%" PRIu64 " valid=%s",
              event->event_id, iree_profile_queue_event_type_name(event->type),
              event->submission_id, event->command_buffer_id,
              event->allocation_id, event->operation_count, duration_ticks,
              is_valid ? "true" : "false");
      if (has_clock_fit) {
        fprintf(file, " ns=%.3f", (double)duration_ticks * ns_per_tick);
      }
      fputc('\n', file);
    }
    for (iree_host_size_t j = 0; j < context->queue_event_count; ++j) {
      const iree_hal_profile_queue_event_t* event =
          &context->queue_events[j].record;
      if (event->physical_device_ordinal != queue->physical_device_ordinal ||
          event->queue_ordinal != queue->queue_ordinal ||
          event->stream_id != queue->stream_id) {
        continue;
      }
      if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
        continue;
      }
      fprintf(
          file,
          "  event=%" PRIu64 " type=%s submission=%" PRIu64
          " strategy=%s waits=%u signals=%u barriers=%u ops=%u bytes=%" PRIu64
          "\n",
          event->event_id, iree_profile_queue_event_type_name(event->type),
          event->submission_id,
          iree_profile_queue_dependency_strategy_name(
              event->dependency_strategy),
          event->wait_count, event->signal_count, event->barrier_count,
          event->operation_count, event->payload_length);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_queue_print_jsonl(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "{\"type\":\"queue_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"queues\":%" PRIhsz ",\"queue_events\":%" PRIhsz
          ",\"queue_device_events\":%" PRIhsz ",\"submissions\":%" PRIhsz
          ",\"matched_dispatches\":%" PRIu64 ",\"valid_dispatches\":%" PRIu64
          ",\"invalid_dispatches\":%" PRIu64 "}\n",
          context->queue_count, context->queue_event_count,
          context->queue_device_event_count, context->queue_aggregate_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  for (iree_host_size_t i = 0; i < context->queue_count; ++i) {
    const iree_hal_profile_queue_record_t* queue = &context->queues[i].record;
    fprintf(file,
            "{\"type\":\"queue\",\"physical_device_ordinal\":%u"
            ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64 "}\n",
            queue->physical_device_ordinal, queue->queue_ordinal,
            queue->stream_id);
  }
  for (iree_host_size_t i = 0; i < context->queue_aggregate_count; ++i) {
    const iree_profile_dispatch_queue_aggregate_t* aggregate =
        &context->queue_aggregates[i];
    if (id_filter >= 0 && aggregate->submission_id != (uint64_t)id_filter) {
      continue;
    }
    const iree_profile_dispatch_device_t* device =
        iree_profile_dispatch_find_device(context,
                                          aggregate->physical_device_ordinal);
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    const double span_ticks = iree_profile_dispatch_span_ticks(
        aggregate->earliest_start_tick, aggregate->latest_end_tick);
    fprintf(file,
            "{\"type\":\"queue_submission\",\"submission_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
            ",\"stream_id\":%" PRIu64 ",\"dispatches\":%" PRIu64
            ",\"valid\":%" PRIu64 ",\"invalid\":%" PRIu64
            ",\"span_ticks\":%.3f,\"total_dispatch_ticks\":%.3f"
            ",\"clock_fit_available\":%s,\"span_ns\":%.3f"
            ",\"total_dispatch_ns\":%.3f}\n",
            aggregate->submission_id, aggregate->physical_device_ordinal,
            aggregate->queue_ordinal, aggregate->stream_id,
            aggregate->dispatch_count, aggregate->valid_count,
            aggregate->invalid_count, span_ticks, aggregate->total_ticks,
            has_clock_fit ? "true" : "false",
            has_clock_fit ? span_ticks * ns_per_tick : 0.0,
            has_clock_fit ? aggregate->total_ticks * ns_per_tick : 0.0);
  }
  for (iree_host_size_t i = 0; i < context->queue_device_event_count; ++i) {
    const iree_hal_profile_queue_device_event_t* event =
        &context->queue_device_events[i].record;
    if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
      continue;
    }
    const bool is_valid = event->start_tick != 0 && event->end_tick != 0 &&
                          event->end_tick >= event->start_tick;
    const uint64_t duration_ticks =
        is_valid ? event->end_tick - event->start_tick : 0;
    const iree_profile_dispatch_device_t* device =
        iree_profile_dispatch_find_device(context,
                                          event->physical_device_ordinal);
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    (void)tick_frequency_hz;
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
            ",\"clock_fit_available\":%s,\"duration_ns\":%.3f}\n",
            event->event_id, iree_profile_queue_event_type_name(event->type),
            event->flags, event->submission_id, event->command_buffer_id,
            event->allocation_id, event->physical_device_ordinal,
            event->queue_ordinal, event->stream_id, event->operation_count,
            event->payload_length, event->start_tick, event->end_tick,
            duration_ticks, is_valid ? "true" : "false",
            has_clock_fit ? "true" : "false",
            has_clock_fit ? (double)duration_ticks * ns_per_tick : 0.0);
  }
  for (iree_host_size_t i = 0; i < context->queue_event_count; ++i) {
    const iree_hal_profile_queue_event_t* event =
        &context->queue_events[i].record;
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
        ",\"host_time_domain\":\"iree_host_time_ns\""
        ",\"wait_count\":%u,\"signal_count\":%u,\"barrier_count\":%u"
        ",\"operation_count\":%u,\"payload_length\":%" PRIu64 "}\n",
        event->event_id, iree_profile_queue_event_type_name(event->type),
        event->flags,
        iree_profile_queue_dependency_strategy_name(event->dependency_strategy),
        event->submission_id, event->command_buffer_id, event->allocation_id,
        event->physical_device_ordinal, event->queue_ordinal, event->stream_id,
        event->host_time_ns, event->wait_count, event->signal_count,
        event->barrier_count, event->operation_count, event->payload_length);
  }
  return iree_ok_status();
}

typedef struct iree_profile_projection_parse_context_t {
  // Dispatch aggregation state shared across metadata and event passes.
  iree_profile_dispatch_context_t* dispatch_context;
  // Optional glob filter applied to projected keys.
  iree_string_view_t filter;
  // Projection selected by the command entry point.
  iree_profile_projection_mode_t projection_mode;
  // Optional entity identifier filter, or -1 when disabled.
  int64_t id_filter;
  // True when raw event rows should be streamed while parsing.
  bool emit_events;
  // Output stream receiving raw event rows when enabled.
  FILE* file;
} iree_profile_projection_parse_context_t;

static iree_status_t iree_profile_projection_metadata_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_projection_parse_context_t* context =
      (iree_profile_projection_parse_context_t*)user_data;
  return iree_profile_dispatch_process_metadata_record(
      context->dispatch_context, record);
}

static iree_status_t iree_profile_projection_event_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_projection_parse_context_t* context =
      (iree_profile_projection_parse_context_t*)user_data;
  return iree_profile_dispatch_process_events_record(
      context->dispatch_context, record, context->filter,
      context->projection_mode, context->id_filter, context->emit_events,
      context->file);
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
  iree_profile_projection_parse_context_t parse_context = {
      .dispatch_context = &context,
      .filter = filter,
      .projection_mode = projection_mode,
      .id_filter = id_filter,
      .emit_events = emit_events,
      .file = file,
  };
  iree_status_t status = iree_profile_file_for_each_record(
      &profile_file, iree_profile_projection_metadata_record, &parse_context);
  if (iree_status_is_ok(status)) {
    status = iree_profile_file_for_each_record(
        &profile_file, iree_profile_projection_event_record, &parse_context);
  }

  if (iree_status_is_ok(status)) {
    switch (projection_mode) {
      case IREE_PROFILE_PROJECTION_MODE_DISPATCH:
        if (is_text) {
          status = iree_profile_dispatch_print_text(&context, filter, file);
        } else {
          iree_profile_dispatch_print_jsonl_summary(&context, filter,
                                                    emit_events, file);
          if (!emit_events) {
            status =
                iree_profile_dispatch_print_jsonl_aggregates(&context, file);
          }
        }
        break;
      case IREE_PROFILE_PROJECTION_MODE_EXECUTABLE:
        if (emit_events) {
          iree_profile_dispatch_print_jsonl_summary(&context, filter,
                                                    emit_events, file);
        } else if (is_text) {
          status = iree_profile_executable_print_text(&context, filter,
                                                      id_filter, file);
        } else {
          status = iree_profile_executable_print_jsonl(&context, filter,
                                                       id_filter, file);
        }
        break;
      case IREE_PROFILE_PROJECTION_MODE_COMMAND:
        if (emit_events) {
          iree_profile_dispatch_print_jsonl_summary(&context, filter,
                                                    emit_events, file);
        } else if (is_text) {
          status = iree_profile_command_print_text(&context, filter, id_filter,
                                                   file);
        } else {
          status = iree_profile_command_print_jsonl(&context, filter, id_filter,
                                                    file);
        }
        break;
      case IREE_PROFILE_PROJECTION_MODE_QUEUE:
        if (emit_events) {
          iree_profile_dispatch_print_jsonl_summary(&context, filter,
                                                    emit_events, file);
        } else if (is_text) {
          status =
              iree_profile_queue_print_text(&context, filter, id_filter, file);
        } else {
          status =
              iree_profile_queue_print_jsonl(&context, filter, id_filter, file);
        }
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported profile projection mode %d",
                                  (int)projection_mode);
        break;
    }
  }

  iree_profile_dispatch_context_deinitialize(&context);
  iree_profile_file_close(&profile_file);
  return status;
}
