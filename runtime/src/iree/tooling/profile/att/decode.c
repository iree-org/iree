// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/att/decode.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "iree/tooling/profile/att/util.h"

typedef struct iree_profile_att_decode_context_t {
  // Decoded trace result populated by this decode operation.
  iree_profile_att_decoded_trace_t* decoded_trace;
  // Terminal error produced while inside the ROCm decoder callback.
  iree_status_t callback_status;
  // ROCm trace decoder handle for the current trace.
  iree_profile_att_rocprofiler_decoder_id_t decoder;
  // ROCm trace decoder library table.
  const iree_profile_att_rocprofiler_library_t* rocprofiler;
} iree_profile_att_decode_context_t;

const char* iree_profile_att_instruction_category_name(uint32_t category) {
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

uint32_t iree_profile_att_instruction_stats_dominant_category(
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

static iree_status_t iree_profile_att_decoded_trace_initialize(
    iree_allocator_t host_allocator,
    const iree_profile_att_comgr_library_t* comgr,
    iree_profile_att_decoded_trace_t* out_decoded_trace) {
  memset(out_decoded_trace, 0, sizeof(*out_decoded_trace));
  out_decoded_trace->host_allocator = host_allocator;
  return iree_profile_att_disassembly_context_allocate(
      host_allocator, comgr, &out_decoded_trace->disassembly);
}

void iree_profile_att_decoded_trace_deinitialize(
    iree_profile_att_decoded_trace_t* decoded_trace) {
  iree_profile_att_disassembly_context_free(decoded_trace->disassembly);
  iree_allocator_t host_allocator = decoded_trace->host_allocator;
  iree_allocator_free(host_allocator, decoded_trace->instructions);
  memset(decoded_trace, 0, sizeof(*decoded_trace));
}

static iree_status_t iree_profile_att_find_or_append_instruction(
    iree_profile_att_decoded_trace_t* decoded_trace,
    iree_profile_att_pc_key_t pc,
    iree_profile_att_instruction_stats_t** out_stats) {
  *out_stats = NULL;
  for (iree_host_size_t i = 0; i < decoded_trace->instruction_count; ++i) {
    iree_profile_att_instruction_stats_t* stats =
        &decoded_trace->instructions[i];
    if (stats->pc.code_object_id == pc.code_object_id &&
        stats->pc.address == pc.address) {
      *out_stats = stats;
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      decoded_trace->host_allocator, decoded_trace->instruction_count + 1,
      sizeof(decoded_trace->instructions[0]),
      &decoded_trace->instruction_capacity,
      (void**)&decoded_trace->instructions));
  iree_profile_att_instruction_stats_t* stats =
      &decoded_trace->instructions[decoded_trace->instruction_count++];
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
  iree_profile_att_decode_context_t* context =
      (iree_profile_att_decode_context_t*)user_data;
  if (!iree_status_is_ok(context->callback_status)) return;

  iree_profile_att_decoded_trace_t* decoded_trace = context->decoded_trace;
  switch (record_type) {
    case IREE_PROFILE_ATT_ROCPROFILER_RECORD_GFXIP:
      decoded_trace->gfxip = (uint64_t)(uintptr_t)trace_events;
      break;
    case IREE_PROFILE_ATT_ROCPROFILER_RECORD_OCCUPANCY:
      decoded_trace->occupancy_count += trace_event_count;
      break;
    case IREE_PROFILE_ATT_ROCPROFILER_RECORD_INFO:
      decoded_trace->info_count += trace_event_count;
      break;
    case IREE_PROFILE_ATT_ROCPROFILER_RECORD_WAVE: {
      iree_profile_att_rocprofiler_wave_t* waves =
          (iree_profile_att_rocprofiler_wave_t*)trace_events;
      decoded_trace->wave_count += trace_event_count;
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
            context->callback_status =
                iree_make_status(IREE_STATUS_DATA_LOSS,
                                 "ATT decoder returned a negative "
                                 "instruction duration");
            return;
          }
          iree_profile_att_instruction_stats_t* stats = NULL;
          iree_status_t status = iree_profile_att_find_or_append_instruction(
              decoded_trace,
              (iree_profile_att_pc_key_t){
                  .code_object_id = instruction->pc.code_object_id,
                  .address = instruction->pc.address,
              },
              &stats);
          if (!iree_status_is_ok(status)) {
            context->callback_status = status;
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
          ++decoded_trace->instruction_event_count;
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
    iree_profile_att_decode_context_t* context) {
  iree_profile_att_decoded_trace_t* decoded_trace = context->decoded_trace;
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
          context->rocprofiler->code_object_load(
              context->decoder, load->code_object_id,
              (uint64_t)load->load_delta, load->load_size,
              code_object->data.data, code_object->data.data_length),
          "rocprofiler_thread_trace_decoder_codeobj_load");
    }
    if (iree_status_is_ok(status)) {
      status = iree_profile_att_disassembly_context_ensure_code_object_loaded(
          decoded_trace->disassembly, code_object);
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

iree_status_t iree_profile_att_decode_trace(
    const iree_profile_att_profile_t* profile,
    const iree_profile_att_trace_t* trace,
    const iree_profile_att_rocprofiler_library_t* rocprofiler,
    const iree_profile_att_comgr_library_t* comgr,
    iree_allocator_t host_allocator,
    iree_profile_att_decoded_trace_t* out_decoded_trace) {
  iree_status_t status = iree_profile_att_decoded_trace_initialize(
      host_allocator, comgr, out_decoded_trace);

  iree_profile_att_decode_context_t context = {
      .decoded_trace = out_decoded_trace,
      .callback_status = iree_ok_status(),
      .rocprofiler = rocprofiler,
  };
  if (iree_status_is_ok(status)) {
    status = iree_profile_att_make_rocprofiler_status(
        rocprofiler->decoder_create(&context.decoder,
                                    rocprofiler->decoder_library_path),
        "rocprofiler_thread_trace_decoder_create");
  }
  if (iree_status_is_ok(status)) {
    status = iree_profile_att_load_trace_code_objects(profile, trace, &context);
  }
  if (iree_status_is_ok(status)) {
    status = iree_profile_att_make_rocprofiler_status(
        rocprofiler->trace_decode(
            context.decoder, iree_profile_att_decode_callback,
            (void*)trace->data.data, trace->data.data_length, &context),
        "rocprofiler_trace_decode");
  }
  if (iree_status_is_ok(status)) {
    status = context.callback_status;
    context.callback_status = iree_ok_status();
  }
  if (iree_status_is_ok(status) && out_decoded_trace->instruction_count > 1) {
    qsort(out_decoded_trace->instructions,
          (size_t)out_decoded_trace->instruction_count,
          sizeof(out_decoded_trace->instructions[0]),
          iree_profile_att_instruction_stats_compare);
  }

  if (context.decoder.handle) {
    rocprofiler->decoder_destroy(context.decoder);
    context.decoder.handle = 0;
  }
  iree_status_free(context.callback_status);
  if (!iree_status_is_ok(status)) {
    iree_profile_att_decoded_trace_deinitialize(out_decoded_trace);
  }
  return status;
}

iree_status_t iree_profile_att_decoded_trace_disassemble_instruction(
    iree_profile_att_decoded_trace_t* decoded_trace,
    const iree_profile_att_pc_key_t* pc, char* instruction_buffer,
    iree_host_size_t instruction_buffer_capacity,
    iree_string_view_t* out_instruction) {
  return iree_profile_att_disassembly_context_disassemble_instruction(
      decoded_trace->disassembly, pc, instruction_buffer,
      instruction_buffer_capacity, out_instruction);
}
