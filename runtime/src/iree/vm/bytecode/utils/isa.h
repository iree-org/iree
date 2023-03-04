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
// Misc utilities
//===----------------------------------------------------------------------===//

#define VMMAX(a, b) (((a) > (b)) ? (a) : (b))
#define VMMIN(a, b) (((a) < (b)) ? (a) : (b))

#define VM_AlignPC(pc, alignment) \
  (pc) = ((pc) + ((alignment)-1)) & ~((alignment)-1)

//===----------------------------------------------------------------------===//
// Bytecode versioning
//===----------------------------------------------------------------------===//

// Major bytecode version; mismatches on this will fail in either direction.
// This allows coarse versioning of completely incompatible versions.
// Matches BytecodeEncoder::kVersionMajor in the compiler.
#define IREE_VM_BYTECODE_VERSION_MAJOR 14
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
#define IREE_I32_REGISTER_COUNT 0x7FFF
#define IREE_REF_REGISTER_COUNT 0x3FFF

#define IREE_I32_REGISTER_MASK 0x7FFF

#define IREE_REF_REGISTER_TYPE_BIT 0x8000
#define IREE_REF_REGISTER_MOVE_BIT 0x4000
#define IREE_REF_REGISTER_MASK 0x3FFF

// Maximum program counter offset within in a single block.
// This is just so that we can steal bits for flags and such. 16MB (today)
// should be more than enough for a single basic block. If not then we should
// compress better!
#define IREE_VM_PC_BLOCK_MAX 0x00FFFFFFu

// Bytecode data -offset used when looking for the start of the currently
// dispatched instruction: `instruction_start = pc - OFFSET`
#define IREE_VM_PC_OFFSET_CORE 1
#define IREE_VM_PC_OFFSET_EXT_I32 2
#define IREE_VM_PC_OFFSET_EXT_F32 2
#define IREE_VM_PC_OFFSET_EXT_F64 2

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

//===----------------------------------------------------------------------===//
// Bytecode data reading with little-/big-endian support
//===----------------------------------------------------------------------===//

// Bytecode data access macros for reading values of a given type from a byte
// offset within the current function.
#define OP_I8(i) iree_unaligned_load_le((uint8_t*)&bytecode_data[pc + (i)])
#define OP_I16(i) iree_unaligned_load_le((uint16_t*)&bytecode_data[pc + (i)])
#define OP_I32(i) iree_unaligned_load_le((uint32_t*)&bytecode_data[pc + (i)])
#define OP_I64(i) iree_unaligned_load_le((uint64_t*)&bytecode_data[pc + (i)])
#define OP_F32(i) iree_unaligned_load_le((float*)&bytecode_data[pc + (i)])
#define OP_F64(i) iree_unaligned_load_le((double*)&bytecode_data[pc + (i)])

#endif  // IREE_VM_BYTECODE_UTILS_ISA_H_
