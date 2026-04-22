// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_ATT_DECODE_H_
#define IREE_TOOLING_PROFILE_ATT_DECODE_H_

#include "iree/tooling/profile/att/bundle.h"
#include "iree/tooling/profile/att/comgr.h"
#include "iree/tooling/profile/att/disassembly.h"
#include "iree/tooling/profile/att/rocprofiler.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

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

typedef struct iree_profile_att_decoded_trace_t {
  // Host allocator used for decoded instruction arrays.
  iree_allocator_t host_allocator;
  // Code-object disassembly state for this trace.
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
} iree_profile_att_decoded_trace_t;

// Decodes one ATT/SQTT trace and aggregates decoded instruction events by PC.
iree_status_t iree_profile_att_decode_trace(
    const iree_profile_att_profile_t* profile,
    const iree_profile_att_trace_t* trace,
    const iree_profile_att_rocprofiler_library_t* rocprofiler,
    const iree_profile_att_comgr_library_t* comgr,
    iree_allocator_t host_allocator,
    iree_profile_att_decoded_trace_t* out_decoded_trace);

// Releases dynamic arrays and disassembly state owned by |decoded_trace|.
void iree_profile_att_decoded_trace_deinitialize(
    iree_profile_att_decoded_trace_t* decoded_trace);

// Returns a stable display name for a ROCprofiler instruction category.
const char* iree_profile_att_instruction_category_name(uint32_t category);

// Returns the most frequently observed instruction category for |stats|.
uint32_t iree_profile_att_instruction_stats_dominant_category(
    const iree_profile_att_instruction_stats_t* stats);

// Disassembles the instruction at |pc| into |instruction_buffer|.
iree_status_t iree_profile_att_decoded_trace_disassemble_instruction(
    iree_profile_att_decoded_trace_t* decoded_trace,
    const iree_profile_att_pc_key_t* pc, char* instruction_buffer,
    iree_host_size_t instruction_buffer_capacity,
    iree_string_view_t* out_instruction);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_ATT_DECODE_H_
