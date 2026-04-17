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

#include "iree/base/internal/dynamic_library.h"
#include "iree/base/internal/path.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_contents.h"
#include "iree/tooling/profile/common.h"
#include "iree/tooling/profile/reader.h"

//===----------------------------------------------------------------------===//
// ROCm trace-decoder ABI mirrors
//===----------------------------------------------------------------------===//

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

typedef void (*iree_profile_att_rocprofiler_callback_t)(
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
  iree_profile_att_rocprofiler_status_t (*trace_decode)(
      iree_profile_att_rocprofiler_decoder_id_t decoder,
      iree_profile_att_rocprofiler_callback_t callback, void* data,
      uint64_t data_length, void* user_data);
  // Returns a diagnostic string for decoder info records.
  const char* (*info_string)(iree_profile_att_rocprofiler_decoder_id_t decoder,
                             iree_profile_att_rocprofiler_info_t info);
} iree_profile_att_rocprofiler_library_t;

//===----------------------------------------------------------------------===//
// AMD COMGR ABI mirrors
//===----------------------------------------------------------------------===//

typedef int32_t iree_profile_att_comgr_status_t;

enum {
  IREE_PROFILE_ATT_COMGR_STATUS_SUCCESS = 0,
};

typedef enum iree_profile_att_comgr_data_kind_e {
  IREE_PROFILE_ATT_COMGR_DATA_KIND_EXECUTABLE = 0x8,
} iree_profile_att_comgr_data_kind_t;

typedef struct iree_profile_att_comgr_data_t {
  // Opaque AMD COMGR data handle.
  uint64_t handle;
} iree_profile_att_comgr_data_t;

typedef struct iree_profile_att_comgr_disassembly_info_t {
  // Opaque AMD COMGR disassembly handle.
  uint64_t handle;
} iree_profile_att_comgr_disassembly_info_t;

typedef struct iree_profile_att_comgr_library_t {
  // Loaded libamd_comgr dynamic library.
  iree_dynamic_library_t* library;
  // Returns a diagnostic string for AMD COMGR status codes.
  iree_profile_att_comgr_status_t (*status_string)(
      iree_profile_att_comgr_status_t status, const char** out_string);
  // Creates an AMD COMGR data object.
  iree_profile_att_comgr_status_t (*create_data)(
      iree_profile_att_comgr_data_kind_t kind,
      iree_profile_att_comgr_data_t* out_data);
  // Releases an AMD COMGR data object.
  iree_profile_att_comgr_status_t (*release_data)(
      iree_profile_att_comgr_data_t data);
  // Copies raw code-object bytes into an AMD COMGR data object.
  iree_profile_att_comgr_status_t (*set_data)(
      iree_profile_att_comgr_data_t data, size_t data_length,
      const char* data_bytes);
  // Queries the ISA name associated with an executable data object.
  iree_profile_att_comgr_status_t (*get_data_isa_name)(
      iree_profile_att_comgr_data_t data, size_t* inout_isa_name_length,
      char* isa_name);
  // Creates a disassembly context for one ISA.
  iree_profile_att_comgr_status_t (*create_disassembly_info)(
      const char* isa_name,
      uint64_t (*read_memory_callback)(uint64_t from, char* to, uint64_t size,
                                       void* user_data),
      void (*print_instruction_callback)(const char* instruction,
                                         void* user_data),
      void (*print_address_annotation_callback)(uint64_t address,
                                                void* user_data),
      iree_profile_att_comgr_disassembly_info_t* out_info);
  // Destroys a disassembly context.
  iree_profile_att_comgr_status_t (*destroy_disassembly_info)(
      iree_profile_att_comgr_disassembly_info_t info);
  // Disassembles one instruction.
  iree_profile_att_comgr_status_t (*disassemble_instruction)(
      iree_profile_att_comgr_disassembly_info_t info, uint64_t address,
      void* user_data, uint64_t* out_instruction_size);
} iree_profile_att_comgr_library_t;

//===----------------------------------------------------------------------===//
// Profile bundle indexes
//===----------------------------------------------------------------------===//

typedef struct iree_profile_att_code_object_t {
  // Producer-local executable identifier that owns this code object.
  uint64_t executable_id;
  // Producer-local code-object marker identifier used in ATT packets.
  uint64_t code_object_id;
  // Borrowed exact HSACO bytes from the mapped profile bundle.
  iree_const_byte_span_t data;
} iree_profile_att_code_object_t;

typedef struct iree_profile_att_code_object_load_t {
  // Session-local physical device ordinal where the code object was loaded.
  uint32_t physical_device_ordinal;
  // Producer-local executable identifier that owns this code object.
  uint64_t executable_id;
  // Producer-local code-object marker identifier used in ATT packets.
  uint64_t code_object_id;
  // AMD loader delta passed to the ROCm trace decoder as load address.
  int64_t load_delta;
  // Loaded code-object memory size reported by the AMD loader.
  uint64_t load_size;
} iree_profile_att_code_object_load_t;

typedef struct iree_profile_att_export_t {
  // Producer-local executable identifier that owns this export.
  uint64_t executable_id;
  // Export ordinal referenced by dispatch and trace records.
  uint32_t export_ordinal;
  // Borrowed export name bytes from the mapped profile bundle.
  iree_string_view_t name;
} iree_profile_att_export_t;

typedef struct iree_profile_att_dispatch_t {
  // Dispatch event record copied from the profile bundle.
  iree_hal_profile_dispatch_event_t record;
  // Physical device ordinal from the containing chunk metadata.
  uint32_t physical_device_ordinal;
  // Queue ordinal from the containing chunk metadata.
  uint32_t queue_ordinal;
} iree_profile_att_dispatch_t;

typedef struct iree_profile_att_trace_t {
  // Executable trace record copied from the profile bundle.
  iree_hal_profile_executable_trace_record_t record;
  // Borrowed raw ATT/SQTT trace bytes from the mapped profile bundle.
  iree_const_byte_span_t data;
} iree_profile_att_trace_t;

typedef struct iree_profile_att_profile_t {
  // Host allocator used for dynamic index arrays.
  iree_allocator_t host_allocator;
  // Mapped profile file contents retaining all borrowed record data.
  iree_io_file_contents_t* file_contents;
  // Dynamic array of embedded code-object images.
  iree_profile_att_code_object_t* code_objects;
  // Number of valid entries in |code_objects|.
  iree_host_size_t code_object_count;
  // Capacity of |code_objects| in entries.
  iree_host_size_t code_object_capacity;
  // Dynamic array of per-device code-object load records.
  iree_profile_att_code_object_load_t* code_object_loads;
  // Number of valid entries in |code_object_loads|.
  iree_host_size_t code_object_load_count;
  // Capacity of |code_object_loads| in entries.
  iree_host_size_t code_object_load_capacity;
  // Dynamic array of executable export names.
  iree_profile_att_export_t* exports;
  // Number of valid entries in |exports|.
  iree_host_size_t export_count;
  // Capacity of |exports| in entries.
  iree_host_size_t export_capacity;
  // Dynamic array of dispatch events.
  iree_profile_att_dispatch_t* dispatches;
  // Number of valid entries in |dispatches|.
  iree_host_size_t dispatch_count;
  // Capacity of |dispatches| in entries.
  iree_host_size_t dispatch_capacity;
  // Dynamic array of executable trace artifacts.
  iree_profile_att_trace_t* traces;
  // Number of valid entries in |traces|.
  iree_host_size_t trace_count;
  // Capacity of |traces| in entries.
  iree_host_size_t trace_capacity;
} iree_profile_att_profile_t;

typedef struct iree_profile_att_pc_key_t {
  // Producer-local code-object marker identifier.
  uint64_t code_object_id;
  // ELF virtual address within |code_object_id|.
  uint64_t address;
} iree_profile_att_pc_key_t;

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

typedef struct iree_profile_att_code_object_decoder_t {
  // Producer-local code-object marker identifier.
  uint64_t code_object_id;
  // Borrowed exact HSACO bytes from the mapped profile bundle.
  iree_const_byte_span_t data;
  // AMD COMGR data object holding |data|.
  iree_profile_att_comgr_data_t comgr_data;
  // AMD COMGR disassembly context for the code object's ISA.
  iree_profile_att_comgr_disassembly_info_t disassembly_info;
} iree_profile_att_code_object_decoder_t;

typedef struct iree_profile_att_decode_state_t {
  // Host allocator used for decoded instruction arrays.
  iree_allocator_t host_allocator;
  // Terminal error produced while inside the ROCm decoder callback.
  iree_status_t callback_status;
  // ROCm trace decoder handle for the current trace.
  iree_profile_att_rocprofiler_decoder_id_t decoder;
  // ROCm trace decoder library table.
  const iree_profile_att_rocprofiler_library_t* rocprofiler;
  // AMD COMGR library table.
  const iree_profile_att_comgr_library_t* comgr;
  // Dynamic array of disassembly contexts, one per loaded code object.
  iree_profile_att_code_object_decoder_t* code_object_decoders;
  // Number of valid entries in |code_object_decoders|.
  iree_host_size_t code_object_decoder_count;
  // Capacity of |code_object_decoders| in entries.
  iree_host_size_t code_object_decoder_capacity;
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

static iree_status_t iree_profile_att_grow_array(
    iree_allocator_t host_allocator, iree_host_size_t element_count,
    iree_host_size_t element_size, iree_host_size_t* inout_capacity,
    void** inout_ptr) {
  if (element_count <= *inout_capacity) return iree_ok_status();
  return iree_allocator_grow_array(
      host_allocator, iree_max((iree_host_size_t)16, element_count),
      element_size, inout_capacity, inout_ptr);
}

static iree_status_t iree_profile_att_copy_cstring(
    iree_string_view_t value, iree_allocator_t host_allocator,
    char** out_string) {
  *out_string = NULL;
  char* string = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, value.size + 1, (void**)&string));
  if (value.size > 0) memcpy(string, value.data, value.size);
  string[value.size] = 0;
  *out_string = string;
  return iree_ok_status();
}

static iree_string_view_t iree_profile_att_cstring_view_or_empty(
    const char* string) {
  return string ? iree_make_cstring_view(string) : iree_string_view_empty();
}

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
// Dynamic ROCm library loading
//===----------------------------------------------------------------------===//

static iree_string_view_t iree_profile_att_rocm_library_path_or_env(
    iree_string_view_t rocm_library_path) {
  if (!iree_string_view_is_empty(rocm_library_path)) {
    return rocm_library_path;
  }

  // Match the AMDGPU runtime capture path knobs so users can configure ROCm
  // discovery once and use the same environment for capture and decode.
  static const char* const env_names[] = {
      "IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH",
      "IREE_HAL_AMDGPU_LIBHSA_PATH",
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(env_names); ++i) {
    iree_string_view_t env_path = iree_make_cstring_view(getenv(env_names[i]));
    if (!iree_string_view_is_empty(env_path)) {
      return env_path;
    }
  }

  return iree_string_view_empty();
}

static iree_status_t iree_profile_att_load_dynamic_library(
    iree_string_view_t rocm_library_path, const char* library_name,
    iree_allocator_t host_allocator, iree_dynamic_library_t** out_library,
    char** out_loaded_path) {
  *out_library = NULL;
  if (out_loaded_path) *out_loaded_path = NULL;

  char* loaded_path = NULL;
  iree_status_t status = iree_ok_status();
  if (!iree_string_view_is_empty(rocm_library_path)) {
    if (iree_file_path_is_dynamic_library(rocm_library_path)) {
      iree_string_view_t basename = iree_file_path_basename(rocm_library_path);
      if (iree_string_view_equal(basename,
                                 iree_make_cstring_view(library_name))) {
        status = iree_profile_att_copy_cstring(rocm_library_path,
                                               host_allocator, &loaded_path);
      } else {
        status = iree_file_path_join(iree_file_path_dirname(rocm_library_path),
                                     iree_make_cstring_view(library_name),
                                     host_allocator, &loaded_path);
      }
    } else {
      status = iree_file_path_join(rocm_library_path,
                                   iree_make_cstring_view(library_name),
                                   host_allocator, &loaded_path);
    }
  } else {
    status = iree_profile_att_copy_cstring(iree_make_cstring_view(library_name),
                                           host_allocator, &loaded_path);
  }

  if (iree_status_is_ok(status)) {
    status = iree_dynamic_library_load_from_file(loaded_path,
                                                 IREE_DYNAMIC_LIBRARY_FLAG_NONE,
                                                 host_allocator, out_library);
  }

  if (iree_status_is_ok(status)) {
    if (out_loaded_path) {
      *out_loaded_path = loaded_path;
      loaded_path = NULL;
    }
  }

  iree_allocator_free(host_allocator, loaded_path);
  return status;
}

static iree_status_t iree_profile_att_lookup_symbol(
    iree_dynamic_library_t* library, const char* symbol_name, void** out_fn) {
  return iree_dynamic_library_lookup_symbol(library, symbol_name, out_fn);
}

static iree_status_t iree_profile_att_resolve_library_dir(
    void* symbol, iree_string_view_t rocm_library_path,
    iree_allocator_t host_allocator, char** out_directory) {
  *out_directory = NULL;
  if (!iree_string_view_is_empty(rocm_library_path) &&
      !iree_file_path_is_dynamic_library(rocm_library_path)) {
    return iree_profile_att_copy_cstring(rocm_library_path, host_allocator,
                                         out_directory);
  }
  if (!iree_string_view_is_empty(rocm_library_path)) {
    iree_string_view_t dirname = iree_file_path_dirname(rocm_library_path);
    if (!iree_string_view_is_empty(dirname)) {
      return iree_profile_att_copy_cstring(dirname, host_allocator,
                                           out_directory);
    }
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_dynamic_library_append_symbol_path_to_builder(symbol, &builder);
  if (iree_status_is_ok(status)) {
    iree_string_view_t library_path = iree_string_builder_view(&builder);
    iree_string_view_t dirname = iree_file_path_dirname(library_path);
    status =
        iree_profile_att_copy_cstring(dirname, host_allocator, out_directory);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t iree_profile_att_rocprofiler_load(
    iree_string_view_t rocm_library_path, iree_allocator_t host_allocator,
    iree_profile_att_rocprofiler_library_t* out_library) {
  memset(out_library, 0, sizeof(*out_library));
  iree_dynamic_library_t* library = NULL;
  char* loaded_path = NULL;
  iree_status_t status = iree_profile_att_load_dynamic_library(
      rocm_library_path, "librocprofiler-sdk.so", host_allocator, &library,
      &loaded_path);

#define IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN(target, symbol_name)        \
  if (iree_status_is_ok(status)) {                                         \
    status = iree_profile_att_lookup_symbol(library, symbol_name,          \
                                            (void**)&out_library->target); \
  }
  IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN(
      decoder_create, "rocprofiler_thread_trace_decoder_create");
  IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN(
      decoder_destroy, "rocprofiler_thread_trace_decoder_destroy");
  IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN(
      code_object_load, "rocprofiler_thread_trace_decoder_codeobj_load");
  IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN(trace_decode,
                                         "rocprofiler_trace_decode");
  IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN(
      info_string, "rocprofiler_thread_trace_decoder_info_string");
#undef IREE_PROFILE_ATT_LOOKUP_ROCPROFILER_FN

  if (iree_status_is_ok(status)) {
    status = iree_profile_att_resolve_library_dir(
        (void*)out_library->trace_decode, rocm_library_path, host_allocator,
        &out_library->decoder_library_path);
  }

  if (iree_status_is_ok(status)) {
    out_library->library = library;
    library = NULL;
  }

  iree_allocator_free(host_allocator, loaded_path);
  iree_dynamic_library_release(library);
  return status;
}

static void iree_profile_att_rocprofiler_deinitialize(
    iree_allocator_t host_allocator,
    iree_profile_att_rocprofiler_library_t* library) {
  iree_allocator_free(host_allocator, library->decoder_library_path);
  iree_dynamic_library_release(library->library);
  memset(library, 0, sizeof(*library));
}

static iree_status_t iree_profile_att_comgr_load(
    iree_string_view_t rocm_library_path, iree_allocator_t host_allocator,
    iree_profile_att_comgr_library_t* out_library) {
  memset(out_library, 0, sizeof(*out_library));
  iree_dynamic_library_t* library = NULL;
  iree_status_t status = iree_profile_att_load_dynamic_library(
      rocm_library_path, "libamd_comgr.so", host_allocator, &library,
      /*out_loaded_path=*/NULL);

#define IREE_PROFILE_ATT_LOOKUP_COMGR_FN(target, symbol_name)              \
  if (iree_status_is_ok(status)) {                                         \
    status = iree_profile_att_lookup_symbol(library, symbol_name,          \
                                            (void**)&out_library->target); \
  }
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(status_string, "amd_comgr_status_string");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(create_data, "amd_comgr_create_data");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(release_data, "amd_comgr_release_data");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(set_data, "amd_comgr_set_data");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(get_data_isa_name,
                                   "amd_comgr_get_data_isa_name");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(create_disassembly_info,
                                   "amd_comgr_create_disassembly_info");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(destroy_disassembly_info,
                                   "amd_comgr_destroy_disassembly_info");
  IREE_PROFILE_ATT_LOOKUP_COMGR_FN(disassemble_instruction,
                                   "amd_comgr_disassemble_instruction");
#undef IREE_PROFILE_ATT_LOOKUP_COMGR_FN

  if (iree_status_is_ok(status)) {
    out_library->library = library;
    library = NULL;
  }

  iree_dynamic_library_release(library);
  return status;
}

static void iree_profile_att_comgr_deinitialize(
    iree_profile_att_comgr_library_t* library) {
  iree_dynamic_library_release(library->library);
  memset(library, 0, sizeof(*library));
}

static iree_status_t iree_profile_att_make_rocprofiler_status(
    iree_profile_att_rocprofiler_status_t status, const char* operation) {
  if (status == IREE_PROFILE_ATT_ROCPROFILER_STATUS_SUCCESS) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNKNOWN,
                          "%s failed with rocprofiler status %d", operation,
                          (int)status);
}

static iree_status_t iree_profile_att_make_comgr_status(
    const iree_profile_att_comgr_library_t* comgr,
    iree_profile_att_comgr_status_t status, const char* operation) {
  if (status == IREE_PROFILE_ATT_COMGR_STATUS_SUCCESS) return iree_ok_status();
  const char* status_string = NULL;
  if (comgr->status_string) {
    comgr->status_string(status, &status_string);
  }
  return iree_make_status(IREE_STATUS_UNKNOWN,
                          "%s failed with AMD COMGR "
                          "status %d (%s)",
                          operation, (int)status,
                          status_string ? status_string : "unknown");
}

//===----------------------------------------------------------------------===//
// Bundle parsing
//===----------------------------------------------------------------------===//

static void iree_profile_att_initialize(
    iree_allocator_t host_allocator, iree_profile_att_profile_t* out_profile) {
  memset(out_profile, 0, sizeof(*out_profile));
  out_profile->host_allocator = host_allocator;
}

static void iree_profile_att_deinitialize(iree_profile_att_profile_t* profile) {
  iree_allocator_t host_allocator = profile->host_allocator;
  iree_allocator_free(host_allocator, profile->traces);
  iree_allocator_free(host_allocator, profile->dispatches);
  iree_allocator_free(host_allocator, profile->exports);
  iree_allocator_free(host_allocator, profile->code_object_loads);
  iree_allocator_free(host_allocator, profile->code_objects);
  iree_io_file_contents_free(profile->file_contents);
  memset(profile, 0, sizeof(*profile));
}

static iree_status_t iree_profile_att_append_code_object(
    iree_profile_att_profile_t* profile,
    iree_profile_att_code_object_t code_object) {
  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      profile->host_allocator, profile->code_object_count + 1,
      sizeof(profile->code_objects[0]), &profile->code_object_capacity,
      (void**)&profile->code_objects));
  profile->code_objects[profile->code_object_count++] = code_object;
  return iree_ok_status();
}

static iree_status_t iree_profile_att_append_code_object_load(
    iree_profile_att_profile_t* profile,
    iree_profile_att_code_object_load_t code_object_load) {
  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      profile->host_allocator, profile->code_object_load_count + 1,
      sizeof(profile->code_object_loads[0]),
      &profile->code_object_load_capacity,
      (void**)&profile->code_object_loads));
  profile->code_object_loads[profile->code_object_load_count++] =
      code_object_load;
  return iree_ok_status();
}

static iree_status_t iree_profile_att_append_export(
    iree_profile_att_profile_t* profile,
    iree_profile_att_export_t export_info) {
  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      profile->host_allocator, profile->export_count + 1,
      sizeof(profile->exports[0]), &profile->export_capacity,
      (void**)&profile->exports));
  profile->exports[profile->export_count++] = export_info;
  return iree_ok_status();
}

static iree_status_t iree_profile_att_append_dispatch(
    iree_profile_att_profile_t* profile, iree_profile_att_dispatch_t dispatch) {
  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      profile->host_allocator, profile->dispatch_count + 1,
      sizeof(profile->dispatches[0]), &profile->dispatch_capacity,
      (void**)&profile->dispatches));
  profile->dispatches[profile->dispatch_count++] = dispatch;
  return iree_ok_status();
}

static iree_status_t iree_profile_att_append_trace(
    iree_profile_att_profile_t* profile, iree_profile_att_trace_t trace) {
  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      profile->host_allocator, profile->trace_count + 1,
      sizeof(profile->traces[0]), &profile->trace_capacity,
      (void**)&profile->traces));
  profile->traces[profile->trace_count++] = trace;
  return iree_ok_status();
}

static iree_status_t iree_profile_att_parse_code_objects(
    iree_profile_att_profile_t* profile,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_executable_code_object_record_t),
      &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_executable_code_object_record_t code_object_record;
    memcpy(&code_object_record, typed_record.contents.data,
           sizeof(code_object_record));
    if ((iree_host_size_t)code_object_record.data_length !=
        typed_record.inline_payload.data_length) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "profile executable code-object chunk has an invalid record");
    }
    if (iree_status_is_ok(status)) {
      iree_profile_att_code_object_t code_object = {
          .executable_id = code_object_record.executable_id,
          .code_object_id = code_object_record.code_object_id,
          .data = typed_record.inline_payload,
      };
      status = iree_profile_att_append_code_object(profile, code_object);
    }
  }
  return status;
}

static iree_status_t iree_profile_att_parse_code_object_loads(
    iree_profile_att_profile_t* profile,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_iterator_t iterator;
  iree_profile_typed_record_iterator_initialize(
      record, sizeof(iree_hal_profile_executable_code_object_load_record_t),
      &iterator);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_profile_typed_record_t typed_record;
    bool has_record = false;
    status = iree_profile_typed_record_iterator_next(&iterator, &typed_record,
                                                     &has_record);
    if (!iree_status_is_ok(status) || !has_record) break;

    iree_hal_profile_executable_code_object_load_record_t load_record;
    memcpy(&load_record, typed_record.contents.data, sizeof(load_record));
    if (typed_record.record_length != sizeof(load_record)) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "profile executable code-object load record has invalid length");
    }
    if (iree_status_is_ok(status)) {
      iree_profile_att_code_object_load_t code_object_load = {
          .physical_device_ordinal = load_record.physical_device_ordinal,
          .executable_id = load_record.executable_id,
          .code_object_id = load_record.code_object_id,
          .load_delta = load_record.load_delta,
          .load_size = load_record.load_size,
      };
      status =
          iree_profile_att_append_code_object_load(profile, code_object_load);
    }
  }
  return status;
}

static iree_status_t iree_profile_att_parse_exports(
    iree_profile_att_profile_t* profile,
    const iree_hal_profile_file_record_t* record) {
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
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "profile executable export chunk has an invalid record");
    }
    if (iree_status_is_ok(status)) {
      iree_profile_att_export_t export_info = {
          .executable_id = export_record.executable_id,
          .export_ordinal = export_record.export_ordinal,
          .name = iree_make_string_view(
              (const char*)typed_record.inline_payload.data,
              typed_record.inline_payload.data_length),
      };
      status = iree_profile_att_append_export(profile, export_info);
    }
  }
  return status;
}

static iree_status_t iree_profile_att_parse_dispatches(
    iree_profile_att_profile_t* profile,
    const iree_hal_profile_file_record_t* record) {
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

    iree_profile_att_dispatch_t dispatch = {
        .physical_device_ordinal = record->header.physical_device_ordinal,
        .queue_ordinal = record->header.queue_ordinal,
    };
    memcpy(&dispatch.record, typed_record.contents.data,
           sizeof(dispatch.record));
    if (typed_record.record_length != sizeof(dispatch.record)) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "profile dispatch event record has invalid length");
    }
    if (iree_status_is_ok(status)) {
      status = iree_profile_att_append_dispatch(profile, dispatch);
    }
  }
  return status;
}

static iree_status_t iree_profile_att_parse_trace(
    iree_profile_att_profile_t* profile,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_typed_record_t typed_record;
  IREE_RETURN_IF_ERROR(iree_profile_typed_record_parse(
      record, 0, sizeof(iree_hal_profile_executable_trace_record_t), 0,
      &typed_record));
  iree_hal_profile_executable_trace_record_t trace_record;
  memcpy(&trace_record, typed_record.contents.data, sizeof(trace_record));
  if ((iree_host_size_t)trace_record.data_length !=
      typed_record.following_payload.data_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile executable trace chunk has invalid trace record length");
  }
  iree_profile_att_trace_t trace = {
      .record = trace_record,
      .data = typed_record.following_payload,
  };
  return iree_profile_att_append_trace(profile, trace);
}

static iree_status_t iree_profile_att_parse_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_att_profile_t* profile = (iree_profile_att_profile_t*)user_data;
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(
          record->content_type,
          IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECTS)) {
    return iree_profile_att_parse_code_objects(profile, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECT_LOADS)) {
    return iree_profile_att_parse_code_object_loads(profile, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    return iree_profile_att_parse_exports(profile, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    return iree_profile_att_parse_dispatches(profile, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACES)) {
    return iree_profile_att_parse_trace(profile, record);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_att_parse_file(
    iree_string_view_t path, iree_profile_att_profile_t* profile) {
  iree_profile_file_t profile_file;
  IREE_RETURN_IF_ERROR(
      iree_profile_file_open(path, profile->host_allocator, &profile_file));
  // Trace payloads borrow from the mapped bundle, so the ATT profile retains
  // the mapping instead of closing the generic reader at the end of parsing.
  profile->file_contents = profile_file.contents;
  iree_profile_file_record_callback_t record_callback = {
      .fn = iree_profile_att_parse_record,
      .user_data = profile,
  };
  return iree_profile_file_for_each_record(&profile_file, record_callback);
}

static const iree_profile_att_code_object_t* iree_profile_att_find_code_object(
    const iree_profile_att_profile_t* profile, uint64_t executable_id,
    uint64_t code_object_id) {
  for (iree_host_size_t i = 0; i < profile->code_object_count; ++i) {
    const iree_profile_att_code_object_t* code_object =
        &profile->code_objects[i];
    if (code_object->executable_id == executable_id &&
        code_object->code_object_id == code_object_id) {
      return code_object;
    }
  }
  return NULL;
}

static const iree_profile_att_export_t* iree_profile_att_find_export(
    const iree_profile_att_profile_t* profile, uint64_t executable_id,
    uint32_t export_ordinal) {
  for (iree_host_size_t i = 0; i < profile->export_count; ++i) {
    const iree_profile_att_export_t* export_info = &profile->exports[i];
    if (export_info->executable_id == executable_id &&
        export_info->export_ordinal == export_ordinal) {
      return export_info;
    }
  }
  return NULL;
}

static const iree_profile_att_dispatch_t* iree_profile_att_find_dispatch(
    const iree_profile_att_profile_t* profile,
    const iree_profile_att_trace_t* trace) {
  for (iree_host_size_t i = 0; i < profile->dispatch_count; ++i) {
    const iree_profile_att_dispatch_t* dispatch = &profile->dispatches[i];
    if (dispatch->record.event_id == trace->record.dispatch_event_id &&
        dispatch->record.submission_id == trace->record.submission_id &&
        dispatch->record.command_buffer_id == trace->record.command_buffer_id &&
        dispatch->record.command_index == trace->record.command_index &&
        dispatch->record.executable_id == trace->record.executable_id &&
        dispatch->record.export_ordinal == trace->record.export_ordinal) {
      return dispatch;
    }
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// ELF and COMGR disassembly
//===----------------------------------------------------------------------===//

typedef struct iree_profile_att_elf64_header_t {
  // ELF identification bytes.
  uint8_t ident[16];
  // Object file type.
  uint16_t type;
  // Target machine.
  uint16_t machine;
  // ELF object version.
  uint32_t version;
  // Entry point virtual address.
  uint64_t entry;
  // Program header table file offset.
  uint64_t program_header_offset;
  // Section header table file offset.
  uint64_t section_header_offset;
  // Processor-specific flags.
  uint32_t flags;
  // ELF header byte size.
  uint16_t header_size;
  // Program header entry byte size.
  uint16_t program_header_entry_size;
  // Program header entry count.
  uint16_t program_header_count;
  // Section header entry byte size.
  uint16_t section_header_entry_size;
  // Section header entry count.
  uint16_t section_header_count;
  // Section name string table index.
  uint16_t section_name_string_table_index;
} iree_profile_att_elf64_header_t;

typedef struct iree_profile_att_elf64_program_header_t {
  // Program header segment type.
  uint32_t type;
  // Program header flags.
  uint32_t flags;
  // Segment file offset.
  uint64_t offset;
  // Segment virtual address.
  uint64_t virtual_address;
  // Segment physical address.
  uint64_t physical_address;
  // Segment byte length in the file.
  uint64_t file_size;
  // Segment byte length in memory.
  uint64_t memory_size;
  // Segment alignment.
  uint64_t alignment;
} iree_profile_att_elf64_program_header_t;

enum {
  IREE_PROFILE_ATT_ELF_MAGIC0 = 0x7F,
  IREE_PROFILE_ATT_ELF_MAGIC1 = 'E',
  IREE_PROFILE_ATT_ELF_MAGIC2 = 'L',
  IREE_PROFILE_ATT_ELF_MAGIC3 = 'F',
  IREE_PROFILE_ATT_ELF_CLASS_64 = 2,
  IREE_PROFILE_ATT_ELF_DATA_LITTLE = 1,
  IREE_PROFILE_ATT_ELF_PROGRAM_HEADER_LOAD = 1,
};

static bool iree_profile_att_elf_virtual_address_to_file_offset(
    iree_const_byte_span_t image, uint64_t virtual_address,
    uint64_t* out_file_offset) {
  *out_file_offset = 0;
  if (image.data_length < sizeof(iree_profile_att_elf64_header_t)) {
    return false;
  }

  iree_profile_att_elf64_header_t header;
  memcpy(&header, image.data, sizeof(header));
  if (header.ident[0] != IREE_PROFILE_ATT_ELF_MAGIC0 ||
      header.ident[1] != IREE_PROFILE_ATT_ELF_MAGIC1 ||
      header.ident[2] != IREE_PROFILE_ATT_ELF_MAGIC2 ||
      header.ident[3] != IREE_PROFILE_ATT_ELF_MAGIC3 ||
      header.ident[4] != IREE_PROFILE_ATT_ELF_CLASS_64 ||
      header.ident[5] != IREE_PROFILE_ATT_ELF_DATA_LITTLE) {
    return false;
  }
  if (header.program_header_entry_size <
      sizeof(iree_profile_att_elf64_program_header_t)) {
    return false;
  }
  if (header.program_header_offset > image.data_length) {
    return false;
  }

  const uint64_t table_length =
      (uint64_t)header.program_header_entry_size * header.program_header_count;
  if (table_length > image.data_length - header.program_header_offset) {
    return false;
  }

  for (uint16_t i = 0; i < header.program_header_count; ++i) {
    const uint64_t program_header_offset =
        header.program_header_offset +
        (uint64_t)i * header.program_header_entry_size;
    iree_profile_att_elf64_program_header_t program_header;
    memcpy(&program_header, image.data + program_header_offset,
           sizeof(program_header));
    if (program_header.type != IREE_PROFILE_ATT_ELF_PROGRAM_HEADER_LOAD) {
      continue;
    }
    if (virtual_address < program_header.virtual_address) continue;
    const uint64_t segment_offset =
        virtual_address - program_header.virtual_address;
    if (segment_offset >= program_header.file_size) continue;
    if (program_header.offset > image.data_length ||
        segment_offset > image.data_length - program_header.offset) {
      return false;
    }
    *out_file_offset = program_header.offset + segment_offset;
    return true;
  }
  return false;
}

typedef struct iree_profile_att_comgr_disassemble_context_t {
  // Borrowed code-object bytes being disassembled.
  iree_const_byte_span_t image;
  // Last instruction text printed by AMD COMGR.
  char instruction[1024];
} iree_profile_att_comgr_disassemble_context_t;

static uint64_t iree_profile_att_comgr_read_memory(uint64_t from, char* to,
                                                   uint64_t size,
                                                   void* user_data) {
  iree_profile_att_comgr_disassemble_context_t* context =
      (iree_profile_att_comgr_disassemble_context_t*)user_data;
  const uintptr_t begin = (uintptr_t)context->image.data;
  const uintptr_t end = begin + context->image.data_length;
  if (from < begin || from >= end) return 0;
  const uint64_t available = end - from;
  const uint64_t read_length = iree_min(size, available);
  memcpy(to, (const void*)(uintptr_t)from, (size_t)read_length);
  return read_length;
}

static void iree_profile_att_comgr_print_instruction(const char* instruction,
                                                     void* user_data) {
  iree_profile_att_comgr_disassemble_context_t* context =
      (iree_profile_att_comgr_disassemble_context_t*)user_data;
  iree_string_view_to_cstring(
      iree_profile_att_cstring_view_or_empty(instruction), context->instruction,
      sizeof(context->instruction));
}

static void iree_profile_att_comgr_print_address_annotation(uint64_t address,
                                                            void* user_data) {
  (void)address;
  (void)user_data;
}

static iree_status_t iree_profile_att_code_object_decoder_initialize(
    const iree_profile_att_comgr_library_t* comgr,
    const iree_profile_att_code_object_t* code_object,
    iree_profile_att_code_object_decoder_t* out_decoder) {
  memset(out_decoder, 0, sizeof(*out_decoder));
  out_decoder->code_object_id = code_object->code_object_id;
  out_decoder->data = code_object->data;

  iree_status_t status = iree_profile_att_make_comgr_status(
      comgr,
      comgr->create_data(IREE_PROFILE_ATT_COMGR_DATA_KIND_EXECUTABLE,
                         &out_decoder->comgr_data),
      "amd_comgr_create_data");
  if (iree_status_is_ok(status)) {
    status = iree_profile_att_make_comgr_status(
        comgr,
        comgr->set_data(out_decoder->comgr_data, out_decoder->data.data_length,
                        (const char*)out_decoder->data.data),
        "amd_comgr_set_data");
  }

  char isa_name[256];
  memset(isa_name, 0, sizeof(isa_name));
  if (iree_status_is_ok(status)) {
    size_t isa_name_length = sizeof(isa_name);
    status = iree_profile_att_make_comgr_status(
        comgr,
        comgr->get_data_isa_name(out_decoder->comgr_data, &isa_name_length,
                                 isa_name),
        "amd_comgr_get_data_isa_name");
  }
  if (iree_status_is_ok(status)) {
    status = iree_profile_att_make_comgr_status(
        comgr,
        comgr->create_disassembly_info(
            isa_name, iree_profile_att_comgr_read_memory,
            iree_profile_att_comgr_print_instruction,
            iree_profile_att_comgr_print_address_annotation,
            &out_decoder->disassembly_info),
        "amd_comgr_create_disassembly_info");
  }

  if (!iree_status_is_ok(status)) {
    if (out_decoder->disassembly_info.handle) {
      comgr->destroy_disassembly_info(out_decoder->disassembly_info);
    }
    if (out_decoder->comgr_data.handle) {
      comgr->release_data(out_decoder->comgr_data);
    }
    memset(out_decoder, 0, sizeof(*out_decoder));
  }
  return status;
}

static void iree_profile_att_code_object_decoder_deinitialize(
    const iree_profile_att_comgr_library_t* comgr,
    iree_profile_att_code_object_decoder_t* decoder) {
  if (decoder->disassembly_info.handle) {
    comgr->destroy_disassembly_info(decoder->disassembly_info);
  }
  if (decoder->comgr_data.handle) {
    comgr->release_data(decoder->comgr_data);
  }
  memset(decoder, 0, sizeof(*decoder));
}

static iree_profile_att_code_object_decoder_t*
iree_profile_att_find_code_object_decoder(
    iree_profile_att_decode_state_t* state, uint64_t code_object_id) {
  for (iree_host_size_t i = 0; i < state->code_object_decoder_count; ++i) {
    iree_profile_att_code_object_decoder_t* decoder =
        &state->code_object_decoders[i];
    if (decoder->code_object_id == code_object_id) return decoder;
  }
  return NULL;
}

static iree_status_t iree_profile_att_disassemble_instruction(
    iree_profile_att_decode_state_t* state, const iree_profile_att_pc_key_t* pc,
    iree_string_view_t* out_instruction, char* instruction_buffer,
    iree_host_size_t instruction_buffer_capacity) {
  instruction_buffer[0] = 0;
  *out_instruction = iree_string_view_empty();

  iree_profile_att_code_object_decoder_t* decoder =
      iree_profile_att_find_code_object_decoder(state, pc->code_object_id);
  if (!decoder) return iree_ok_status();

  uint64_t file_offset = 0;
  if (!iree_profile_att_elf_virtual_address_to_file_offset(
          decoder->data, pc->address, &file_offset)) {
    return iree_ok_status();
  }
  if (file_offset >= decoder->data.data_length) return iree_ok_status();

  iree_profile_att_comgr_disassemble_context_t context = {
      .image = decoder->data,
  };
  const uint64_t address =
      (uint64_t)(uintptr_t)(decoder->data.data + file_offset);
  uint64_t instruction_size = 0;
  iree_status_t status = iree_profile_att_make_comgr_status(
      state->comgr,
      state->comgr->disassemble_instruction(decoder->disassembly_info, address,
                                            &context, &instruction_size),
      "amd_comgr_disassemble_instruction");
  if (iree_status_is_ok(status)) {
    iree_string_view_to_cstring(iree_make_cstring_view(context.instruction),
                                instruction_buffer,
                                instruction_buffer_capacity);
    *out_instruction = iree_make_cstring_view(instruction_buffer);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// ATT decoding
//===----------------------------------------------------------------------===//

static void iree_profile_att_decode_state_initialize(
    iree_allocator_t host_allocator,
    const iree_profile_att_rocprofiler_library_t* rocprofiler,
    const iree_profile_att_comgr_library_t* comgr,
    iree_profile_att_decode_state_t* out_state) {
  memset(out_state, 0, sizeof(*out_state));
  out_state->host_allocator = host_allocator;
  out_state->callback_status = iree_ok_status();
  out_state->rocprofiler = rocprofiler;
  out_state->comgr = comgr;
}

static void iree_profile_att_decode_state_deinitialize(
    iree_profile_att_decode_state_t* state) {
  for (iree_host_size_t i = 0; i < state->code_object_decoder_count; ++i) {
    iree_profile_att_code_object_decoder_deinitialize(
        state->comgr, &state->code_object_decoders[i]);
  }
  iree_allocator_t host_allocator = state->host_allocator;
  iree_allocator_free(host_allocator, state->code_object_decoders);
  iree_allocator_free(host_allocator, state->instructions);
  iree_status_free(state->callback_status);
  memset(state, 0, sizeof(*state));
}

static iree_status_t iree_profile_att_append_code_object_decoder(
    iree_profile_att_decode_state_t* state,
    const iree_profile_att_code_object_t* code_object) {
  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      state->host_allocator, state->code_object_decoder_count + 1,
      sizeof(state->code_object_decoders[0]),
      &state->code_object_decoder_capacity,
      (void**)&state->code_object_decoders));
  iree_profile_att_code_object_decoder_t* decoder =
      &state->code_object_decoders[state->code_object_decoder_count];
  IREE_RETURN_IF_ERROR(iree_profile_att_code_object_decoder_initialize(
      state->comgr, code_object, decoder));
  ++state->code_object_decoder_count;
  return iree_ok_status();
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
        iree_profile_att_find_code_object(profile, load->executable_id,
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
    if (iree_status_is_ok(status) && !iree_profile_att_find_code_object_decoder(
                                         state, load->code_object_id)) {
      status = iree_profile_att_append_code_object_decoder(state, code_object);
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
  iree_profile_att_decode_state_initialize(host_allocator, rocprofiler, comgr,
                                           out_state);
  iree_status_t status = iree_profile_att_make_rocprofiler_status(
      rocprofiler->decoder_create(&out_state->decoder,
                                  rocprofiler->decoder_library_path),
      "rocprofiler_thread_trace_decoder_create");
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
  const iree_profile_att_export_t* export_info = iree_profile_att_find_export(
      profile, trace->record.executable_id, trace->record.export_ordinal);
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
  const iree_profile_att_export_t* export_info = iree_profile_att_find_export(
      profile, trace->record.executable_id, trace->record.export_ordinal);
  const iree_profile_att_dispatch_t* dispatch =
      iree_profile_att_find_dispatch(profile, trace);
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
    status = iree_profile_att_disassemble_instruction(
        state, &stats->pc, &instruction, instruction_buffer,
        sizeof(instruction_buffer));
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
  const iree_profile_att_export_t* export_info = iree_profile_att_find_export(
      profile, trace->record.executable_id, trace->record.export_ordinal);
  const iree_profile_att_dispatch_t* dispatch =
      iree_profile_att_find_dispatch(profile, trace);
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
    status = iree_profile_att_disassemble_instruction(
        state, &stats->pc, &instruction, instruction_buffer,
        sizeof(instruction_buffer));
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
  iree_profile_att_initialize(host_allocator, &profile);
  iree_status_t status = iree_profile_att_parse_file(path, &profile);
  if (iree_status_is_ok(status)) {
    status = iree_profile_att_report_file(
        &profile, format, filter, id, rocm_library_path, file, host_allocator);
  }
  iree_profile_att_deinitialize(&profile);
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
    "att",
    "Decode AMDGPU ATT/SQTT traces and annotate decoded instructions.",
    iree_profile_att_run,
};

const iree_profile_command_t* iree_profile_att_command(void) {
  return &kIreeProfileAttCommand;
}
