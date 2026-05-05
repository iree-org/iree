// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_ATT_DISASSEMBLY_H_
#define IREE_TOOLING_PROFILE_ATT_DISASSEMBLY_H_

#include <stdbool.h>

#include "iree/tooling/profile/att/bundle.h"
#include "iree/tooling/profile/att/comgr.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_profile_att_pc_key_t {
  // Producer-local code-object marker identifier.
  uint64_t code_object_id;
  // ELF virtual address within |code_object_id|.
  uint64_t address;
} iree_profile_att_pc_key_t;

typedef struct iree_profile_att_disassembly_context_t
    iree_profile_att_disassembly_context_t;

// Allocates ATT code-object disassembly state.
iree_status_t iree_profile_att_disassembly_context_allocate(
    iree_allocator_t host_allocator,
    const iree_profile_att_comgr_library_t* comgr,
    iree_profile_att_disassembly_context_t** out_context);

// Frees code-object disassembly state owned by |context|.
void iree_profile_att_disassembly_context_free(
    iree_profile_att_disassembly_context_t* context);

// Ensures |context| has a loaded code object for later instruction disassembly.
iree_status_t iree_profile_att_disassembly_context_ensure_code_object_loaded(
    iree_profile_att_disassembly_context_t* context,
    const iree_profile_att_code_object_t* code_object);

// Disassembles the instruction at |pc| into |instruction_buffer|.
//
// |instruction_buffer| must have at least one byte. |out_instruction| receives
// a view into |instruction_buffer| and remains valid until the buffer is
// modified. Empty output means the PC could not be resolved against any loaded
// code object.
iree_status_t iree_profile_att_disassembly_context_disassemble_instruction(
    iree_profile_att_disassembly_context_t* context,
    const iree_profile_att_pc_key_t* pc, char* instruction_buffer,
    iree_host_size_t instruction_buffer_capacity,
    iree_string_view_t* out_instruction);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_ATT_DISASSEMBLY_H_
