// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_UTILS_ISA_H_
#define IREE_VM_BYTECODE_UTILS_ISA_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/utils/generated/op_table.h"  // IWYU pragma: export

// NOTE: include order matters:
#include "iree/base/internal/flatcc/parsing.h"          // IWYU pragma: export
#include "iree/schemas/bytecode_module_def_reader.h"    // IWYU pragma: export
#include "iree/schemas/bytecode_module_def_verifier.h"  // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Bytecode versioning
//===----------------------------------------------------------------------===//

// Major bytecode version; mismatches on this will fail in either direction.
// This allows coarse versioning of completely incompatible versions.
// Matches BytecodeEncoder::kVersionMajor in the compiler.
#define IREE_VM_BYTECODE_VERSION_MAJOR 16
// Minor bytecode version; lower versions are allowed to enable newer runtimes
// to load older serialized files when there are backwards-compatible changes.
// Higher versions are disallowed as they occur when new ops are added that
// otherwise cannot be executed by older runtimes.
// Matches BytecodeEncoder::kVersionMinor in the compiler.
#define IREE_VM_BYTECODE_VERSION_MINOR 0

//===----------------------------------------------------------------------===//
// Bytecode structural constants
//===----------------------------------------------------------------------===//

// Size of a register ordinal in the bytecode.
#define IREE_REGISTER_ORDINAL_SIZE sizeof(uint16_t)

// Maximum register count per bank.
// This determines the bits required to reference registers in the VM bytecode.
#define IREE_VM_ISA_I32_REGISTER_COUNT 0x7FFF
#define IREE_VM_ISA_REF_REGISTER_COUNT 0x3FFF

#define IREE_VM_ISA_I32_REGISTER_MASK 0x7FFF

#define IREE_VM_ISA_REF_REGISTER_TYPE_BIT 0x8000
#define IREE_VM_ISA_REF_REGISTER_MOVE_BIT 0x4000
#define IREE_VM_ISA_REF_REGISTER_MASK 0x3FFF

// Maximum program counter offset within in a single block.
// This is just so that we can steal bits for flags and such. 16MB (today)
// should be more than enough for a single basic block. If not then we should
// compress better!
#define IREE_VM_ISA_PC_BLOCK_MAX 0x00FFFFFFu

// Bytecode data -offset used when looking for the start of the currently
// dispatched instruction: `instruction_start = pc - OFFSET`
#define IREE_VM_ISA_PC_OFFSET_CORE 1
#define IREE_VM_ISA_PC_OFFSET_EXT_I32 2
#define IREE_VM_ISA_PC_OFFSET_EXT_F32 2
#define IREE_VM_ISA_PC_OFFSET_EXT_F64 2

//===----------------------------------------------------------------------===//
// Common ordinal encodings
//===----------------------------------------------------------------------===//

// Function ordinals are tagged when referencing import functions.
// The low bits are the ordinal in the import table; the high bit indicates
// whether the ordinal is an import reference.
#define IREE_VM_ISA_FUNCTION_ORDINAL_IMPORT_BIT 0x80000000u

static inline bool iree_vm_isa_function_ordinal_is_import(uint32_t ordinal) {
  return (ordinal & IREE_VM_ISA_FUNCTION_ORDINAL_IMPORT_BIT) != 0;
}

static inline uint32_t iree_vm_isa_function_ordinal_as_import(
    uint32_t ordinal) {
  return ordinal & ~IREE_VM_ISA_FUNCTION_ORDINAL_IMPORT_BIT;
}

// Interleaved src-dst register sets for branch register remapping.
// This structure is an overlay for the bytecode that is serialized in a
// matching format.
typedef struct iree_vm_register_remap_list_t {
  uint16_t size;
  struct pair {
    uint16_t src_reg;
    uint16_t dst_reg;
  } pairs[];
} iree_vm_register_remap_list_t;
static_assert(iree_alignof(iree_vm_register_remap_list_t) == 2,
              "Expecting byte alignment (to avoid padding)");
static_assert(offsetof(iree_vm_register_remap_list_t, pairs) == 2,
              "Expect no padding in the struct");

#endif  // IREE_VM_BYTECODE_UTILS_ISA_H_
