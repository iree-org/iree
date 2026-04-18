// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/att/att.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "iree/tooling/profile/att/bundle.h"
#include "iree/tooling/profile/att/comgr.h"
#include "iree/tooling/profile/att/disassembly.h"
#include "iree/tooling/profile/att/rocm.h"
#include "iree/tooling/profile/att/rocprofiler.h"
#include "iree/tooling/profile/att/util.h"
#include "iree/tooling/profile/common.h"

//===----------------------------------------------------------------------===//
// ATT decode state
//===----------------------------------------------------------------------===//

typedef struct iree_profile_att_instruction_stats_t {
  // Program counter key for this aggregate.
  iree_profile_att_pc_key_t pc;
  // Number of decoded instruction events at this PC.
  uint64_t hit_count;
  // Sum of instruction durations in shader-clock cycles.
  uint64_t duration;
  // Sum of instruction stalls in shader-clock cycles.
  uint64_t stall;
  // Earliest decoded instruction timestamp.
  int64_t first_time;
  // Latest decoded instruction completion timestamp.
  int64_t last_time;
  // Per-category hit counters.
  uint64_t categories[IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_COUNT];
} iree_profile_att_instruction_stats_t;

typedef struct iree_profile_att_decode_state_t {
  // Host allocator used for decoded instruction arrays.
  iree_allocator_t host_allocator;
  // Terminal error produced while inside the ROCm decoder callback.
  iree_status_t callback_status;
  // ROCm trace decoder handle for the current trace.
  iree_profile_att_rocprofiler_decoder_id_t decoder;
  // ROCm trace decoder library table.
  const iree_profile_att_rocprofiler_library_t* rocprofiler;
  // Code-object disassembly state for the current trace.
  iree_profile_att_disassembly_context_t* disassembly;
  // Dynamic array of per-PC instruction aggregates.
  iree_profile_att_instruction_stats_t* instructions;
  // Number of valid entries in |instructions|.
  iree_host_size_t instruction_count;
  // Capacity of |instructions| in entries.
  iree_host_size_t instruction_capacity;
  // GFX IP major version reported by the decoder.
  uint64_t gfxip;
  // Number of decoded wave records.
  uint64_t wave_count;
  // Number of decoded occupancy records.
  uint64_t occupancy_count;
  // Number of decoded info records.
  uint64_t info_count;
  // Number of decoded instruction events before per-PC aggregation.
  uint64_t instruction_event_count;
} iree_profile_att_decode_state_t;

//===----------------------------------------------------------------------===//
// Generic helpers
//===----------------------------------------------------------------------===//

static const char* iree_profile_att_instruction_category_name(
    uint32_t category) {
  switch (category) {
    case IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_SMEM:
      return "SMEM";
    case IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_SALU:
      return "SALU";
    case IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_VMEM:
      return "VMEM";
    case IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_FLAT:
      return "FLAT";
    case IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_LDS:
      return "LDS";
    case IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_VALU:
      return "VALU";
    case IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_JUMP:
      return "JUMP";
    case IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_NEXT:
      return "NEXT";
    case IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_IMMED:
      return "IMMED";
    case IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_CONTEXT:
      return "CONTEXT";
    case IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_MESSAGE:
      return "MESSAGE";
    case IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_BVH:
      return "BVH";
    default:
      return "NONE";
  }
}

static uint32_t iree_profile_att_instruction_dominant_category(
    const iree_profile_att_instruction_stats_t* stats) {
  uint32_t best_category = 0;
  uint64_t best_count = 0;
  for (uint32_t i = 0; i < IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_COUNT; ++i) {
    if (stats->categories[i] > best_count) {
      best_category = i;
      best_count = stats->categories[i];
    }
  }
  return best_category;
}

//===----------------------------------------------------------------------===//
// ATT decoding
//===----------------------------------------------------------------------===//

static iree_status_t iree_profile_att_decode_state_initialize(
    iree_allocator_t host_allocator,
    const iree_profile_att_rocprofiler_library_t* rocprofiler,
    const iree_profile_att_comgr_library_t* comgr,
    iree_profile_att_decode_state_t* out_state) {
  memset(out_state, 0, sizeof(*out_state));
  out_state->host_allocator = host_allocator;
  out_state->callback_status = iree_ok_status();
  out_state->rocprofiler = rocprofiler;
  return iree_profile_att_disassembly_context_create(host_allocator, comgr,
                                                     &out_state->disassembly);
}

static void iree_profile_att_decode_state_deinitialize(
    iree_profile_att_decode_state_t* state) {
  iree_profile_att_disassembly_context_destroy(state->disassembly);
  iree_allocator_t host_allocator = state->host_allocator;
  iree_allocator_free(host_allocator, state->instructions);
  iree_status_free(state->callback_status);
  memset(state, 0, sizeof(*state));
}

static iree_status_t iree_profile_att_find_or_append_instruction(
    iree_profile_att_decode_state_t* state, iree_profile_att_pc_key_t pc,
    iree_profile_att_instruction_stats_t** out_stats) {
  *out_stats = NULL;
  for (iree_host_size_t i = 0; i < state->instruction_count; ++i) {
    iree_profile_att_instruction_stats_t* stats = &state->instructions[i];
    if (stats->pc.code_object_id == pc.code_object_id &&
        stats->pc.address == pc.address) {
      *out_stats = stats;
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      state->host_allocator, state->instruction_count + 1,
      sizeof(state->instructions[0]), &state->instruction_capacity,
      (void**)&state->instructions));
  iree_profile_att_instruction_stats_t* stats =
      &state->instructions[state->instruction_count++];
  memset(stats, 0, sizeof(*stats));
  stats->pc = pc;
  stats->first_time = INT64_MAX;
  stats->last_time = INT64_MIN;
  *out_stats = stats;
  return iree_ok_status();
}

static int iree_profile_att_instruction_stats_compare(const void* lhs,
                                                      const void* rhs) {
  const iree_profile_att_instruction_stats_t* lhs_stats =
      (const iree_profile_att_instruction_stats_t*)lhs;
  const iree_profile_att_instruction_stats_t* rhs_stats =
      (const iree_profile_att_instruction_stats_t*)rhs;
  if (lhs_stats->pc.code_object_id < rhs_stats->pc.code_object_id) return -1;
  if (lhs_stats->pc.code_object_id > rhs_stats->pc.code_object_id) return 1;
  if (lhs_stats->pc.address < rhs_stats->pc.address) return -1;
  if (lhs_stats->pc.address > rhs_stats->pc.address) return 1;
  return 0;
}

static void iree_profile_att_decode_callback(
    iree_profile_att_rocprofiler_record_type_t record_type, void* trace_events,
    uint64_t trace_event_count, void* user_data) {
  iree_profile_att_decode_state_t* state =
      (iree_profile_att_decode_state_t*)user_data;
  if (!iree_status_is_ok(state->callback_status)) return;
  switch (record_type) {
    case IREE_PROFILE_ATT_ROCPROFILER_RECORD_GFXIP:
      state->gfxip = (uint64_t)(uintptr_t)trace_events;
      break;
    case IREE_PROFILE_ATT_ROCPROFILER_RECORD_OCCUPANCY:
      state->occupancy_count += trace_event_count;
      break;
    case IREE_PROFILE_ATT_ROCPROFILER_RECORD_INFO:
      state->info_count += trace_event_count;
      break;
    case IREE_PROFILE_ATT_ROCPROFILER_RECORD_WAVE: {
      iree_profile_att_rocprofiler_wave_t* waves =
          (iree_profile_att_rocprofiler_wave_t*)trace_events;
      state->wave_count += trace_event_count;
      for (uint64_t wave_index = 0; wave_index < trace_event_count;
           ++wave_index) {
        const iree_profile_att_rocprofiler_wave_t* wave = &waves[wave_index];
        for (uint64_t instruction_index = 0;
             instruction_index < wave->instructions_size; ++instruction_index) {
          const iree_profile_att_rocprofiler_instruction_t* instruction =
              &wave->instructions_array[instruction_index];
          if (instruction->pc.code_object_id == 0 &&
              instruction->pc.address == 0) {
            continue;
          }
          if (instruction->duration < 0) {
            state->callback_status =
                iree_make_status(IREE_STATUS_DATA_LOSS,
                                 "ATT decoder returned a negative "
                                 "instruction duration");
            return;
          }
          iree_profile_att_instruction_stats_t* stats = NULL;
          iree_status_t status = iree_profile_att_find_or_append_instruction(
              state,
              (iree_profile_att_pc_key_t){
                  .code_object_id = instruction->pc.code_object_id,
                  .address = instruction->pc.address,
              },
              &stats);
          if (!iree_status_is_ok(status)) {
            state->callback_status = status;
            return;
          }
          ++stats->hit_count;
          stats->duration += (uint64_t)instruction->duration;
          stats->stall += (uint64_t)instruction->stall;
          stats->first_time = iree_min(stats->first_time, instruction->time);
          stats->last_time =
              iree_max(stats->last_time,
                       instruction->time + (int64_t)instruction->duration);
          if (instruction->category <
              IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_COUNT) {
            ++stats->categories[instruction->category];
          }
          ++state->instruction_event_count;
        }
      }
      break;
    }
    default:
      break;
  }
}

static iree_status_t iree_profile_att_load_trace_code_objects(
    const iree_profile_att_profile_t* profile,
    const iree_profile_att_trace_t* trace,
    iree_profile_att_decode_state_t* state) {
  iree_host_size_t loaded_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < profile->code_object_load_count; ++i) {
    const iree_profile_att_code_object_load_t* load =
        &profile->code_object_loads[i];
    if (load->physical_device_ordinal !=
        trace->record.physical_device_ordinal) {
      continue;
    }
    const iree_profile_att_code_object_t* code_object =
        iree_profile_att_profile_find_code_object(profile, load->executable_id,
                                                  load->code_object_id);
    if (!code_object) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "missing code-object image for executable %" PRIu64
                           " code-object %" PRIu64,
                           load->executable_id, load->code_object_id);
    }
    if (iree_status_is_ok(status)) {
      status = iree_profile_att_make_rocprofiler_status(
          state->rocprofiler->code_object_load(
              state->decoder, load->code_object_id, (uint64_t)load->load_delta,
              load->load_size, code_object->data.data,
              code_object->data.data_length),
          "rocprofiler_thread_trace_decoder_codeobj_load");
    }
    if (iree_status_is_ok(status)) {
      status = iree_profile_att_disassembly_context_ensure_code_object_loaded(
          state->disassembly, code_object);
    }
    if (iree_status_is_ok(status)) ++loaded_count;
  }
  if (iree_status_is_ok(status) && loaded_count == 0) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "no code-object load records for physical device %" PRIu32,
        trace->record.physical_device_ordinal);
  }
  return status;
}

static iree_status_t iree_profile_att_decode_trace(
    const iree_profile_att_profile_t* profile,
    const iree_profile_att_trace_t* trace,
    const iree_profile_att_rocprofiler_library_t* rocprofiler,
    const iree_profile_att_comgr_library_t* comgr,
    iree_allocator_t host_allocator,
    iree_profile_att_decode_state_t* out_state) {
  iree_status_t status = iree_profile_att_decode_state_initialize(
      host_allocator, rocprofiler, comgr, out_state);
  if (iree_status_is_ok(status)) {
    status = iree_profile_att_make_rocprofiler_status(
        rocprofiler->decoder_create(&out_state->decoder,
                                    rocprofiler->decoder_library_path),
        "rocprofiler_thread_trace_decoder_create");
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_profile_att_load_trace_code_objects(profile, trace, out_state);
  }
  if (iree_status_is_ok(status)) {
    status = iree_profile_att_make_rocprofiler_status(
        rocprofiler->trace_decode(
            out_state->decoder, iree_profile_att_decode_callback,
            (void*)trace->data.data, trace->data.data_length, out_state),
        "rocprofiler_trace_decode");
  }
  if (iree_status_is_ok(status)) {
    status = out_state->callback_status;
    out_state->callback_status = iree_ok_status();
  }
  if (iree_status_is_ok(status) && out_state->instruction_count > 1) {
    qsort(out_state->instructions, (size_t)out_state->instruction_count,
          sizeof(out_state->instructions[0]),
          iree_profile_att_instruction_stats_compare);
  }

  if (out_state->decoder.handle) {
    rocprofiler->decoder_destroy(out_state->decoder);
    out_state->decoder.handle = 0;
  }
  if (!iree_status_is_ok(status)) {
    iree_profile_att_decode_state_deinitialize(out_state);
  }
  return status;
}

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
    const iree_profile_att_decode_state_t* state, FILE* file) {
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
          trace->data.data_length, state->gfxip, state->wave_count,
          state->occupancy_count, state->instruction_event_count,
          state->instruction_count, state->info_count);
}

static iree_status_t iree_profile_att_print_trace_text(
    const iree_profile_att_profile_t* profile,
    const iree_profile_att_trace_t* trace,
    iree_profile_att_decode_state_t* state, FILE* file) {
  iree_profile_att_print_trace_header_text(profile, trace, state, file);
  fprintf(file, "%-14s %-10s %-8s %-10s %-10s %s\n", "pc", "category", "hits",
          "duration", "stall", "instruction");
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < state->instruction_count; ++i) {
    const iree_profile_att_instruction_stats_t* stats = &state->instructions[i];
    const uint32_t category =
        iree_profile_att_instruction_dominant_category(stats);
    char instruction_buffer[1024];
    iree_string_view_t instruction = iree_string_view_empty();
    status = iree_profile_att_disassembly_context_disassemble_instruction(
        state->disassembly, &stats->pc, instruction_buffer,
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
    iree_profile_att_decode_state_t* state, FILE* file) {
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
          trace->data.data_length, state->gfxip, state->wave_count,
          state->occupancy_count, state->instruction_event_count,
          state->instruction_count, state->info_count);
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
       iree_status_is_ok(status) && i < state->instruction_count; ++i) {
    const iree_profile_att_instruction_stats_t* stats = &state->instructions[i];
    const uint32_t category =
        iree_profile_att_instruction_dominant_category(stats);
    char instruction_buffer[1024];
    iree_string_view_t instruction = iree_string_view_empty();
    status = iree_profile_att_disassembly_context_disassemble_instruction(
        state->disassembly, &stats->pc, instruction_buffer,
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

    iree_profile_att_decode_state_t state;
    status = iree_profile_att_decode_trace(profile, trace, &rocprofiler, &comgr,
                                           host_allocator, &state);
    if (iree_status_is_ok(status)) {
      if (iree_string_view_equal(format, IREE_SV("jsonl"))) {
        status =
            iree_profile_att_print_trace_jsonl(profile, trace, &state, file);
      } else {
        status =
            iree_profile_att_print_trace_text(profile, trace, &state, file);
      }
      ++matched_trace_count;
      iree_profile_att_decode_state_deinitialize(&state);
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
