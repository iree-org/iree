// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_ATT_COMGR_H_
#define IREE_TOOLING_PROFILE_ATT_COMGR_H_

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

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

// Reads |size| bytes from the target code object address |from| into |to|.
typedef uint64_t (*iree_profile_att_comgr_read_memory_fn_t)(uint64_t from,
                                                            char* to,
                                                            uint64_t size,
                                                            void* user_data);

// Receives the disassembled instruction text for one COMGR instruction.
typedef void (*iree_profile_att_comgr_print_instruction_fn_t)(
    const char* instruction, void* user_data);

// Receives a COMGR address annotation for the current instruction.
typedef void (*iree_profile_att_comgr_print_address_annotation_fn_t)(
    uint64_t address, void* user_data);

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
  //
  // AMD COMGR's ABI passes callback functions and user data separately, so this
  // imported function pointer intentionally uses *_fn_t callback types instead
  // of IREE's usual callback_t struct convention.
  iree_profile_att_comgr_status_t (*create_disassembly_info)(
      const char* isa_name, iree_profile_att_comgr_read_memory_fn_t read_memory,
      iree_profile_att_comgr_print_instruction_fn_t print_instruction,
      iree_profile_att_comgr_print_address_annotation_fn_t print_address,
      iree_profile_att_comgr_disassembly_info_t* out_info);
  // Destroys a disassembly context.
  iree_profile_att_comgr_status_t (*destroy_disassembly_info)(
      iree_profile_att_comgr_disassembly_info_t info);
  // Disassembles one instruction.
  iree_profile_att_comgr_status_t (*disassemble_instruction)(
      iree_profile_att_comgr_disassembly_info_t info, uint64_t address,
      void* user_data, uint64_t* out_instruction_size);
} iree_profile_att_comgr_library_t;

// Loads the AMD COMGR disassembly entry points.
iree_status_t iree_profile_att_comgr_load(
    iree_string_view_t rocm_library_path, iree_allocator_t host_allocator,
    iree_profile_att_comgr_library_t* out_library);

// Releases dynamic library resources owned by |library|.
void iree_profile_att_comgr_deinitialize(
    iree_profile_att_comgr_library_t* library);

// Converts an AMD COMGR status code into an IREE status.
iree_status_t iree_profile_att_make_comgr_status(
    const iree_profile_att_comgr_library_t* comgr,
    iree_profile_att_comgr_status_t status, const char* operation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_ATT_COMGR_H_
