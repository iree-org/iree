// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_ASSEMBLER_BYTECODE_H_
#define IREE_VM_BYTECODE_ASSEMBLER_BYTECODE_H_

#include "iree/vm/bytecode/assembler/module.h"
#include "iree/vm/bytecode/isa/encoding_table.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Emits a single byte to the module bytecode stream.
iree_status_t iree_vm_bytecode_assembler_emit_u8(
    iree_vm_bytecode_assembler_module_t* module, uint8_t value);

// Pads the module bytecode stream to |alignment| bytes.
iree_status_t iree_vm_bytecode_assembler_align_bytecode(
    iree_vm_bytecode_assembler_module_t* module, iree_host_size_t alignment);

// Parses a textual register and records the function register-file extent.
iree_status_t iree_vm_bytecode_assembler_parse_register(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t value,
    iree_vm_isa_register_bank_t register_bank, uint16_t* out_register);

// Parses and encodes one textual VM instruction.
iree_status_t iree_vm_bytecode_assembler_parse_instruction(
    iree_vm_bytecode_assembler_module_t* module, iree_string_view_t line);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_ASSEMBLER_BYTECODE_H_
