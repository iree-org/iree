// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/att/att.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>

#include "iree/tooling/profile/att/bundle.h"
#include "iree/tooling/profile/att/comgr.h"
#include "iree/tooling/profile/att/decode.h"
#include "iree/tooling/profile/att/rocm.h"
#include "iree/tooling/profile/att/rocprofiler.h"
#include "iree/tooling/profile/common.h"

//===----------------------------------------------------------------------===//
// Report printing
//===----------------------------------------------------------------------===//

static bool iree_profile_att_trace_matches(
    const iree_profile_att_profile_t* profile,
    const iree_profile_att_trace_t* trace, iree_string_view_t filter,
    int64_t id) {
  if (trace->record.format !=
      IREE_HAL_PROFILE_EXECUTABLE_TRACE_FORMAT_AMDGPU_ATT) {
    return false;
  }
  if (id >= 0 && trace->record.trace_id != (uint64_t)id &&
      trace->record.dispatch_event_id != (uint64_t)id) {
    return false;
  }
  const iree_profile_att_export_t* export_info =
      iree_profile_att_profile_find_export(profile, trace->record.executable_id,
                                           trace->record.export_ordinal);
  if (iree_string_view_is_empty(filter) ||
      iree_string_view_equal(filter, IREE_SV("*"))) {
    return true;
  }
  return export_info &&
         iree_string_view_match_pattern(export_info->name, filter);
}

static void iree_profile_att_print_trace_header_text(
    const iree_profile_att_profile_t* profile,
    const iree_profile_att_trace_t* trace,
    const iree_profile_att_decoded_trace_t* decoded_trace, FILE* file) {
  const iree_profile_att_export_t* export_info =
      iree_profile_att_profile_find_export(profile, trace->record.executable_id,
                                           trace->record.export_ordinal);
  const iree_profile_att_dispatch_t* dispatch =
      iree_profile_att_profile_find_dispatch(profile, trace);
  fprintf(file, "ATT trace %" PRIu64 "\n", trace->record.trace_id);
  fprintf(file, "  executable=%" PRIu64 " export=%" PRIu32 " name=%.*s\n",
          trace->record.executable_id, trace->record.export_ordinal,
          export_info ? (int)export_info->name.size : 9,
          export_info ? export_info->name.data : "<unknown>");
  fprintf(file,
          "  device=%" PRIu32 " queue=%" PRIu32 " submission=%" PRIu64
          " command_buffer=%" PRIu64 " command_index=%" PRIu32 "\n",
          trace->record.physical_device_ordinal, trace->record.queue_ordinal,
          trace->record.submission_id, trace->record.command_buffer_id,
          trace->record.command_index);
  if (dispatch) {
    fprintf(
        file,
        "  workgroups=[%" PRIu32 ",%" PRIu32 ",%" PRIu32
        "] workgroup_size=[%" PRIu32 ",%" PRIu32 ",%" PRIu32
        "] duration_ticks=%" PRIu64 "\n",
        dispatch->record.workgroup_count[0],
        dispatch->record.workgroup_count[1],
        dispatch->record.workgroup_count[2], dispatch->record.workgroup_size[0],
        dispatch->record.workgroup_size[1], dispatch->record.workgroup_size[2],
        dispatch->record.end_tick - dispatch->record.start_tick);
  }
  fprintf(file,
          "  raw_bytes=%" PRIhsz " gfxip=%" PRIu64 " waves=%" PRIu64
          " occupancy=%" PRIu64 " instructions=%" PRIu64 " unique_pcs=%" PRIhsz
          " info=%" PRIu64 "\n\n",
          trace->data.data_length, decoded_trace->gfxip,
          decoded_trace->wave_count, decoded_trace->occupancy_count,
          decoded_trace->instruction_event_count,
          decoded_trace->instruction_count, decoded_trace->info_count);
}

static iree_status_t iree_profile_att_print_trace_text(
    const iree_profile_att_profile_t* profile,
    const iree_profile_att_trace_t* trace,
    iree_profile_att_decoded_trace_t* decoded_trace, FILE* file) {
  iree_profile_att_print_trace_header_text(profile, trace, decoded_trace, file);
  fprintf(file, "%-14s %-10s %-8s %-10s %-10s %s\n", "pc", "category", "hits",
          "duration", "stall", "instruction");
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < decoded_trace->instruction_count; ++i) {
    const iree_profile_att_instruction_stats_t* stats =
        &decoded_trace->instructions[i];
    const uint32_t category =
        iree_profile_att_instruction_stats_dominant_category(stats);
    char instruction_buffer[1024];
    iree_string_view_t instruction = iree_string_view_empty();
    status = iree_profile_att_decoded_trace_disassemble_instruction(
        decoded_trace, &stats->pc, instruction_buffer,
        sizeof(instruction_buffer), &instruction);
    if (iree_status_is_ok(status) && iree_string_view_is_empty(instruction)) {
      instruction = IREE_SV("<unresolved>");
    }
    if (iree_status_is_ok(status)) {
      fprintf(file,
              "0x%012" PRIx64 " %-10s %-8" PRIu64 " %-10" PRIu64 " %-10" PRIu64
              " %.*s\n",
              stats->pc.address,
              iree_profile_att_instruction_category_name(category),
              stats->hit_count, stats->duration, stats->stall,
              (int)instruction.size, instruction.data);
    }
  }
  fputc('\n', file);
  return status;
}

static iree_status_t iree_profile_att_print_trace_jsonl(
    const iree_profile_att_profile_t* profile,
    const iree_profile_att_trace_t* trace,
    iree_profile_att_decoded_trace_t* decoded_trace, FILE* file) {
  const iree_profile_att_export_t* export_info =
      iree_profile_att_profile_find_export(profile, trace->record.executable_id,
                                           trace->record.export_ordinal);
  const iree_profile_att_dispatch_t* dispatch =
      iree_profile_att_profile_find_dispatch(profile, trace);
  fprintf(file,
          "{\"type\":\"att_trace\",\"trace_id\":%" PRIu64
          ",\"dispatch_event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64
          ",\"command_buffer_id\":%" PRIu64 ",\"command_index\":%" PRIu32
          ",\"executable_id\":%" PRIu64 ",\"export_ordinal\":%" PRIu32
          ",\"physical_device_ordinal\":%" PRIu32 ",\"queue_ordinal\":%" PRIu32
          ",\"stream_id\":%" PRIu64 ",\"key\":",
          trace->record.trace_id, trace->record.dispatch_event_id,
          trace->record.submission_id, trace->record.command_buffer_id,
          trace->record.command_index, trace->record.executable_id,
          trace->record.export_ordinal, trace->record.physical_device_ordinal,
          trace->record.queue_ordinal, trace->record.stream_id);
  iree_profile_fprint_json_string(
      file, export_info ? export_info->name : IREE_SV("<unknown>"));
  fprintf(file,
          ",\"raw_bytes\":%" PRIhsz ",\"gfxip\":%" PRIu64 ",\"waves\":%" PRIu64
          ",\"occupancy\":%" PRIu64 ",\"instruction_events\":%" PRIu64
          ",\"unique_pcs\":%" PRIhsz ",\"info\":%" PRIu64,
          trace->data.data_length, decoded_trace->gfxip,
          decoded_trace->wave_count, decoded_trace->occupancy_count,
          decoded_trace->instruction_event_count,
          decoded_trace->instruction_count, decoded_trace->info_count);
  if (dispatch) {
    fprintf(
        file,
        ",\"workgroup_count\":[%" PRIu32 ",%" PRIu32 ",%" PRIu32
        "],\"workgroup_size\":[%" PRIu32 ",%" PRIu32 ",%" PRIu32
        "],\"duration_ticks\":%" PRIu64,
        dispatch->record.workgroup_count[0],
        dispatch->record.workgroup_count[1],
        dispatch->record.workgroup_count[2], dispatch->record.workgroup_size[0],
        dispatch->record.workgroup_size[1], dispatch->record.workgroup_size[2],
        dispatch->record.end_tick - dispatch->record.start_tick);
  }
  fputs("}\n", file);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < decoded_trace->instruction_count; ++i) {
    const iree_profile_att_instruction_stats_t* stats =
        &decoded_trace->instructions[i];
    const uint32_t category =
        iree_profile_att_instruction_stats_dominant_category(stats);
    char instruction_buffer[1024];
    iree_string_view_t instruction = iree_string_view_empty();
    status = iree_profile_att_decoded_trace_disassemble_instruction(
        decoded_trace, &stats->pc, instruction_buffer,
        sizeof(instruction_buffer), &instruction);
    if (iree_status_is_ok(status) && iree_string_view_is_empty(instruction)) {
      instruction = IREE_SV("<unresolved>");
    }
    if (iree_status_is_ok(status)) {
      fprintf(
          file,
          "{\"type\":\"att_instruction\",\"trace_id\":%" PRIu64
          ",\"code_object_id\":%" PRIu64 ",\"pc\":%" PRIu64 ",\"category\":",
          trace->record.trace_id, stats->pc.code_object_id, stats->pc.address);
      iree_profile_fprint_json_string(
          file, iree_make_cstring_view(
                    iree_profile_att_instruction_category_name(category)));
      fprintf(file,
              ",\"hits\":%" PRIu64 ",\"duration\":%" PRIu64
              ",\"stall\":%" PRIu64 ",\"first_time\":%" PRId64
              ",\"last_time\":%" PRId64 ",\"instruction\":",
              stats->hit_count, stats->duration, stats->stall,
              stats->first_time, stats->last_time);
      iree_profile_fprint_json_string(file, instruction);
      fputs("}\n", file);
    }
  }
  return status;
}

static iree_status_t iree_profile_att_report_file(
    const iree_profile_att_profile_t* profile, iree_string_view_t format,
    iree_string_view_t filter, int64_t id, iree_string_view_t rocm_library_path,
    FILE* file, iree_allocator_t host_allocator) {
  if (!iree_string_view_equal(format, IREE_SV("text")) &&
      !iree_string_view_equal(format, IREE_SV("jsonl"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported att format '%.*s'", (int)format.size,
                            format.data);
  }

  rocm_library_path =
      iree_profile_att_rocm_library_path_or_env(rocm_library_path);

  iree_profile_att_rocprofiler_library_t rocprofiler;
  IREE_RETURN_IF_ERROR(iree_profile_att_rocprofiler_load(
      rocm_library_path, host_allocator, &rocprofiler));
  iree_profile_att_comgr_library_t comgr;
  iree_status_t status =
      iree_profile_att_comgr_load(rocm_library_path, host_allocator, &comgr);

  uint64_t matched_trace_count = 0;
  if (iree_status_is_ok(status) &&
      iree_string_view_equal(format, IREE_SV("text"))) {
    fprintf(
        file,
        "profile: code_objects=%" PRIhsz " loads=%" PRIhsz " exports=%" PRIhsz
        " dispatches=%" PRIhsz " traces=%" PRIhsz "\n\n",
        profile->code_object_count, profile->code_object_load_count,
        profile->export_count, profile->dispatch_count, profile->trace_count);
  }

  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < profile->trace_count; ++i) {
    const iree_profile_att_trace_t* trace = &profile->traces[i];
    if (!iree_profile_att_trace_matches(profile, trace, filter, id)) continue;

    iree_profile_att_decoded_trace_t decoded_trace;
    status = iree_profile_att_decode_trace(profile, trace, &rocprofiler, &comgr,
                                           host_allocator, &decoded_trace);
    if (iree_status_is_ok(status)) {
      if (iree_string_view_equal(format, IREE_SV("jsonl"))) {
        status = iree_profile_att_print_trace_jsonl(profile, trace,
                                                    &decoded_trace, file);
      } else {
        status = iree_profile_att_print_trace_text(profile, trace,
                                                   &decoded_trace, file);
      }
      ++matched_trace_count;
      iree_profile_att_decoded_trace_deinitialize(&decoded_trace);
    }
  }

  if (iree_status_is_ok(status) && matched_trace_count == 0) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no AMDGPU ATT trace records matched");
  }

  iree_profile_att_comgr_deinitialize(&comgr);
  iree_profile_att_rocprofiler_deinitialize(host_allocator, &rocprofiler);
  return status;
}

IREE_API_EXPORT iree_status_t iree_profile_att_file(
    iree_string_view_t path, iree_string_view_t format,
    iree_string_view_t filter, int64_t id, iree_string_view_t rocm_library_path,
    FILE* file, iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_profile_att_profile_t profile;
  iree_profile_att_profile_initialize(host_allocator, &profile);
  iree_status_t status = iree_profile_att_profile_parse_file(path, &profile);
  if (iree_status_is_ok(status)) {
    status = iree_profile_att_report_file(
        &profile, format, filter, id, rocm_library_path, file, host_allocator);
  }
  iree_profile_att_profile_deinitialize(&profile);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_profile_att_run(
    const iree_profile_command_invocation_t* invocation) {
  return iree_profile_att_file(
      invocation->input_path, invocation->options->format,
      invocation->options->filter, invocation->options->id,
      invocation->options->rocm_library_path, invocation->output_file,
      invocation->options->host_allocator);
}

static const iree_profile_command_t kIreeProfileAttCommand = {
    .name = "att",
    .summary =
        "Decode AMDGPU ATT/SQTT traces and annotate decoded instructions.",
    .supported_formats =
        IREE_PROFILE_COMMAND_FORMAT_TEXT | IREE_PROFILE_COMMAND_FORMAT_JSONL,
    .accepted_options = IREE_PROFILE_COMMAND_OPTION_FILTER |
                        IREE_PROFILE_COMMAND_OPTION_ID |
                        IREE_PROFILE_COMMAND_OPTION_ROCM_LIBRARY_PATH,
    .run = iree_profile_att_run,
};

const iree_profile_command_t* iree_profile_att_command(void) {
  return &kIreeProfileAttCommand;
}
