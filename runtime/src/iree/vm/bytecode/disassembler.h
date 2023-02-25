// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_DISASSEMBLER_H_
#define IREE_VM_BYTECODE_DISASSEMBLER_H_

#include <stdio.h>

#include "iree/base/string_builder.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/dispatch_util.h"
#include "iree/vm/bytecode/module_impl.h"

// Controls how bytecode disassembly is formatted.
typedef enum iree_vm_bytecode_disassembly_format_e {
  IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_DEFAULT = 0,
  // Includes the input register values inline in the op text.
  // Example: `%i0 <= ShrI32U %i2(5), %i3(6)`
  IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_INLINE_VALUES = 1u << 0,
} iree_vm_bytecode_disassembly_format_t;

// Disassembles the bytecode operation at |pc| using the provided module state.
// Appends the disasembled op to |string_builder| in a format based on |format|.
// If |regs| are available then values can be added using the format mode.
//
// Example: `%i0 <= ShrI32U %i2, %i3`
//
// WARNING: this does not currently perform any verification on the bytecode;
// it's assumed all bytecode is valid. This is a debug tool: you shouldn't be
// running this in production on untrusted inputs anyway.
iree_status_t iree_vm_bytecode_disassemble_op(
    iree_vm_bytecode_module_t* module,
    iree_vm_bytecode_module_state_t* module_state, uint16_t function_ordinal,
    iree_vm_source_offset_t pc, const iree_vm_registers_t* regs,
    iree_vm_bytecode_disassembly_format_t format,
    iree_string_builder_t* string_builder);

iree_status_t iree_vm_bytecode_trace_disassembly(
    iree_vm_stack_frame_t* frame, iree_vm_source_offset_t pc,
    const iree_vm_registers_t* regs, FILE* file);

#endif  // IREE_VM_BYTECODE_DISASSEMBLER_H_
