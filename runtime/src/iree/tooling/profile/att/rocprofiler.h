// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_ATT_ROCPROFILER_H_
#define IREE_TOOLING_PROFILE_ATT_ROCPROFILER_H_

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef int32_t iree_profile_att_rocprofiler_status_t;

enum {
  IREE_PROFILE_ATT_ROCPROFILER_STATUS_SUCCESS = 0,
};

typedef struct iree_profile_att_rocprofiler_decoder_id_t {
  // Opaque ROCprofiler decoder handle.
  uint64_t handle;
} iree_profile_att_rocprofiler_decoder_id_t;

typedef enum iree_profile_att_rocprofiler_record_type_e {
  IREE_PROFILE_ATT_ROCPROFILER_RECORD_GFXIP = 0,
  IREE_PROFILE_ATT_ROCPROFILER_RECORD_OCCUPANCY = 1,
  IREE_PROFILE_ATT_ROCPROFILER_RECORD_PERFEVENT = 2,
  IREE_PROFILE_ATT_ROCPROFILER_RECORD_WAVE = 3,
  IREE_PROFILE_ATT_ROCPROFILER_RECORD_INFO = 4,
  IREE_PROFILE_ATT_ROCPROFILER_RECORD_DEBUG = 5,
  IREE_PROFILE_ATT_ROCPROFILER_RECORD_SHADERDATA = 6,
  IREE_PROFILE_ATT_ROCPROFILER_RECORD_REALTIME = 7,
  IREE_PROFILE_ATT_ROCPROFILER_RECORD_RT_FREQUENCY = 8,
} iree_profile_att_rocprofiler_record_type_t;

typedef enum iree_profile_att_rocprofiler_info_e {
  IREE_PROFILE_ATT_ROCPROFILER_INFO_NONE = 0,
  IREE_PROFILE_ATT_ROCPROFILER_INFO_DATA_LOST = 1,
  IREE_PROFILE_ATT_ROCPROFILER_INFO_STITCH_INCOMPLETE = 2,
  IREE_PROFILE_ATT_ROCPROFILER_INFO_WAVE_INCOMPLETE = 3,
} iree_profile_att_rocprofiler_info_t;

typedef enum iree_profile_att_instruction_category_e {
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_NONE = 0,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_SMEM = 1,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_SALU = 2,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_VMEM = 3,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_FLAT = 4,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_LDS = 5,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_VALU = 6,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_JUMP = 7,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_NEXT = 8,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_IMMED = 9,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_CONTEXT = 10,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_MESSAGE = 11,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_BVH = 12,
  IREE_PROFILE_ATT_INSTRUCTION_CATEGORY_COUNT = 13,
} iree_profile_att_instruction_category_t;

typedef struct iree_profile_att_rocprofiler_pc_t {
  // ELF virtual address when |code_object_id| is nonzero.
  uint64_t address;
  // Producer marker id for the code object containing |address|.
  uint64_t code_object_id;
} iree_profile_att_rocprofiler_pc_t;

typedef struct iree_profile_att_rocprofiler_instruction_t {
  // Instruction category from iree_profile_att_instruction_category_t.
  uint32_t category : 8;
  // Stall duration in shader-clock cycles.
  uint32_t stall : 24;
  // Total instruction duration in shader-clock cycles.
  int32_t duration;
  // Timestamp when the wave first attempted to execute this instruction.
  int64_t time;
  // Program counter attributed to this instruction.
  iree_profile_att_rocprofiler_pc_t pc;
} iree_profile_att_rocprofiler_instruction_t;

typedef struct iree_profile_att_rocprofiler_wave_state_t {
  // ROCprofiler wave-state type.
  int32_t type;
  // State duration in shader-clock cycles.
  int32_t duration;
} iree_profile_att_rocprofiler_wave_state_t;

typedef struct iree_profile_att_rocprofiler_wave_t {
  // Target CU id on gfx9, or WGP id on gfx10+.
  uint8_t cu;
  // SIMD id within the target CU/WGP.
  uint8_t simd;
  // Wave slot id within the selected SIMD.
  uint8_t wave_id;
  // Number of CWSR context events in this wave.
  uint8_t contexts;
  // Reserved field in the ROCprofiler ABI.
  uint32_t reserved0;
  // Reserved field in the ROCprofiler ABI.
  uint32_t reserved1;
  // Reserved field in the ROCprofiler ABI.
  uint32_t reserved2;
  // Wave begin timestamp in shader-clock cycles.
  int64_t begin_time;
  // Wave end timestamp in shader-clock cycles.
  int64_t end_time;
  // Number of entries in |timeline_array|.
  uint64_t timeline_size;
  // Number of entries in |instructions_array|.
  uint64_t instructions_size;
  // Decoded wave-state timeline entries.
  iree_profile_att_rocprofiler_wave_state_t* timeline_array;
  // Decoded instruction events for this wave.
  iree_profile_att_rocprofiler_instruction_t* instructions_array;
} iree_profile_att_rocprofiler_wave_t;

typedef void (*iree_profile_att_rocprofiler_callback_fn_t)(
    iree_profile_att_rocprofiler_record_type_t record_type, void* trace_events,
    uint64_t trace_event_count, void* user_data);

typedef struct iree_profile_att_rocprofiler_library_t {
  // Loaded librocprofiler-sdk dynamic library.
  iree_dynamic_library_t* library;
  // Directory passed to rocprofiler_thread_trace_decoder_create.
  char* decoder_library_path;
  // Creates a trace decoder handle.
  iree_profile_att_rocprofiler_status_t (*decoder_create)(
      iree_profile_att_rocprofiler_decoder_id_t* out_decoder,
      const char* library_path);
  // Destroys a trace decoder handle.
  void (*decoder_destroy)(iree_profile_att_rocprofiler_decoder_id_t decoder);
  // Adds a code object to a trace decoder handle.
  iree_profile_att_rocprofiler_status_t (*code_object_load)(
      iree_profile_att_rocprofiler_decoder_id_t decoder, uint64_t load_id,
      uint64_t load_address, uint64_t load_size, const void* data,
      uint64_t data_length);
  // Decodes one ATT/SQTT trace blob.
  //
  // ROCprofiler's ABI passes the callback function and user data separately, so
  // this imported function pointer intentionally uses a *_fn_t callback type
  // instead of IREE's usual callback_t struct convention.
  iree_profile_att_rocprofiler_status_t (*trace_decode)(
      iree_profile_att_rocprofiler_decoder_id_t decoder,
      iree_profile_att_rocprofiler_callback_fn_t callback_fn, void* data,
      uint64_t data_length, void* user_data);
  // Returns a diagnostic string for decoder info records.
  const char* (*info_string)(iree_profile_att_rocprofiler_decoder_id_t decoder,
                             iree_profile_att_rocprofiler_info_t info);
} iree_profile_att_rocprofiler_library_t;

// Loads the ROCprofiler SDK trace decoder entry points.
iree_status_t iree_profile_att_rocprofiler_load(
    iree_string_view_t rocm_library_path, iree_allocator_t host_allocator,
    iree_profile_att_rocprofiler_library_t* out_library);

// Releases dynamic library resources owned by |library|.
void iree_profile_att_rocprofiler_deinitialize(
    iree_allocator_t host_allocator,
    iree_profile_att_rocprofiler_library_t* library);

// Converts a ROCprofiler status code into an IREE status.
iree_status_t iree_profile_att_make_rocprofiler_status(
    iree_profile_att_rocprofiler_status_t status, const char* operation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_ATT_ROCPROFILER_H_
