// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ABI_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_ABI_COMMAND_BUFFER_H_

#include "iree/hal/drivers/amdgpu/abi/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Command Buffer Program ABI
//===----------------------------------------------------------------------===//

enum {
  // Magic value stored in every command-buffer block header.
  IREE_HAL_AMDGPU_COMMAND_BUFFER_BLOCK_MAGIC = 0x444D4342u,
  // Version of the block ABI defined in this header.
  IREE_HAL_AMDGPU_COMMAND_BUFFER_BLOCK_VERSION_0 = 0,
  // Required alignment for all command records and fixup records.
  IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT = 8,
};

// Opcodes in the AMDGPU command-buffer program.
typedef enum iree_hal_amdgpu_command_buffer_opcode_e {
  IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_INVALID = 0,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER = 1,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH = 2,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL = 3,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY = 4,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE = 5,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_PROFILE_MARKER = 6,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH = 7,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH = 8,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN = 9,
} iree_hal_amdgpu_command_buffer_opcode_t;

// Command flags shared by all command records.
typedef enum iree_hal_amdgpu_command_buffer_command_flag_bits_e {
  IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER = 1u << 0,
} iree_hal_amdgpu_command_buffer_command_flag_bits_t;

// Fixup kinds used to resolve command operands at execution time.
typedef enum iree_hal_amdgpu_command_buffer_fixup_kind_e {
  IREE_HAL_AMDGPU_COMMAND_BUFFER_FIXUP_KIND_INVALID = 0,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_FIXUP_KIND_DYNAMIC_BINDING = 1,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_FIXUP_KIND_STATIC_BINDING = 2,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_FIXUP_KIND_DEFERRED_STATIC_BINDING = 3,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_FIXUP_KIND_RODATA = 4,
} iree_hal_amdgpu_command_buffer_fixup_kind_t;

// Header stored at byte 0 of every command-buffer block.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_block_header_t {
  // Magic value identifying this as an AMDGPU command-buffer block.
  uint32_t magic;
  // ABI version used to interpret this block.
  uint16_t version;
  // Byte length of this header.
  uint16_t header_length;
  // Ordinal of this block within the command-buffer program.
  uint32_t block_ordinal;
  // Block flags reserved for replay strategy selection.
  uint32_t flags;
  // Total byte capacity of this block, including this header.
  uint32_t block_length;
  // Byte offset from this header to the first command record.
  uint32_t command_offset;
  // Total bytes occupied by command records.
  uint32_t command_length;
  // Byte offset from this header to the first fixup record.
  uint32_t fixup_offset;
  // Number of command records in this block, including terminators.
  uint16_t command_count;
  // Number of fixup records in this block.
  uint16_t fixup_count;
  // Worst-case AQL packets emitted when replaying this block.
  uint32_t aql_packet_count;
  // Worst-case kernarg bytes emitted when replaying this block.
  uint32_t kernarg_length;
  // Byte offset from this header to block-local read-only payload data.
  uint32_t rodata_offset;
  // Total bytes occupied by block-local read-only payload data.
  uint32_t rodata_length;
  // Reserved bytes that must be zero in version 0.
  uint32_t reserved0[3];
} iree_hal_amdgpu_command_buffer_block_header_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_block_header_t) == 64,
    "command-buffer block header must stay cache-line sized");

// Header common to every command record.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_command_header_t {
  // Opcode from iree_hal_amdgpu_command_buffer_opcode_t.
  uint8_t opcode;
  // Command flags from iree_hal_amdgpu_command_buffer_command_flag_bits_t.
  uint8_t flags;
  // Command length in 8-byte qwords, including this header.
  uint16_t length_qwords;
  // Program-global command index used for profiling/source attribution.
  uint32_t command_index;
  // Byte offset from the block header to this command's first fixup record.
  uint32_t fixup_offset;
  // Number of fixup records owned by this command.
  uint16_t fixup_count;
  // Reserved bytes that must be zero in version 0.
  uint16_t reserved0;
} iree_hal_amdgpu_command_buffer_command_header_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_command_header_t) == 16,
    "command header size is part of the command-buffer ABI");

// Fixup record used to patch a binding, rodata pointer, or condition operand.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_command_buffer_fixup_t {
  // Byte offset from the resolved binding or rodata base.
  uint64_t source_offset;
  // Binding or rodata ordinal interpreted according to |kind|.
  uint32_t ordinal;
  // Byte offset inside the command's template payload to patch.
  uint32_t patch_offset;
  // Fixup kind from iree_hal_amdgpu_command_buffer_fixup_kind_t.
  uint16_t kind;
  // Fixup flags reserved for replay policy.
  uint16_t flags;
  // Reserved bytes that must be zero in version 0.
  uint32_t reserved0;
} iree_hal_amdgpu_command_buffer_fixup_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_command_buffer_fixup_t) == 24,
                          "fixup size is part of the command-buffer ABI");

// Barrier metadata command. Replay normally folds this into the next
// packet-bearing command instead of emitting a standalone packet.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_barrier_command_t {
  // Common command record header.
  iree_hal_amdgpu_command_buffer_command_header_t header;
  // Acquire fence scope requested by the barrier.
  uint8_t acquire_scope;
  // Release fence scope requested by the barrier.
  uint8_t release_scope;
  // Barrier flags reserved for visibility-debt lowering.
  uint16_t barrier_flags;
  // Reserved bytes that must be zero in version 0.
  uint32_t reserved0;
} iree_hal_amdgpu_command_buffer_barrier_command_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_barrier_command_t) == 24,
    "barrier command size must remain qword aligned");

// Dispatch command record.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_dispatch_command_t {
  // Common command record header.
  iree_hal_amdgpu_command_buffer_command_header_t header;
  // Command-buffer executable table ordinal.
  uint32_t executable_ordinal;
  // Executable export ordinal.
  uint32_t export_ordinal;
  // Workgroup count along X, Y, and Z.
  uint32_t workgroup_count[3];
  // Dynamic workgroup local memory in bytes.
  uint32_t dynamic_shared_memory;
  // Byte offset to the kernarg template payload.
  uint32_t kernarg_template_offset;
  // Byte length of the kernarg template payload.
  uint32_t kernarg_template_length;
} iree_hal_amdgpu_command_buffer_dispatch_command_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t) == 48,
    "dispatch command size must remain qword aligned");

// Fill command record.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_fill_command_t {
  // Common command record header.
  iree_hal_amdgpu_command_buffer_command_header_t header;
  // Byte length of the target range.
  uint64_t length;
  // Repeated fill pattern stored in the low bytes.
  uint64_t pattern;
  // Byte length of the fill pattern.
  uint8_t pattern_length;
  // Reserved bytes that must be zero in version 0.
  uint8_t reserved0[7];
} iree_hal_amdgpu_command_buffer_fill_command_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_fill_command_t) == 40,
    "fill command size must remain qword aligned");

// Copy command record.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_copy_command_t {
  // Common command record header.
  iree_hal_amdgpu_command_buffer_command_header_t header;
  // Byte length of the copied range.
  uint64_t length;
} iree_hal_amdgpu_command_buffer_copy_command_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_copy_command_t) == 24,
    "copy command size must remain qword aligned");

// Update command record.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_update_command_t {
  // Common command record header.
  iree_hal_amdgpu_command_buffer_command_header_t header;
  // Byte offset to the update payload in rodata storage.
  uint32_t rodata_offset;
  // Byte length of the update payload.
  uint32_t rodata_length;
} iree_hal_amdgpu_command_buffer_update_command_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_update_command_t) == 24,
    "update command size must remain qword aligned");

// Unconditional branch terminator.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_branch_command_t {
  // Common command record header.
  iree_hal_amdgpu_command_buffer_command_header_t header;
  // Target block ordinal.
  uint32_t target_block_ordinal;
  // Reserved bytes that must be zero in version 0.
  uint32_t reserved0;
} iree_hal_amdgpu_command_buffer_branch_command_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_branch_command_t) == 24,
    "branch command size must remain qword aligned");

// Conditional branch terminator.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_cond_branch_command_t {
  // Common command record header.
  iree_hal_amdgpu_command_buffer_command_header_t header;
  // Target block ordinal when the loaded condition value is non-zero.
  uint32_t true_block_ordinal;
  // Target block ordinal when the loaded condition value is zero.
  uint32_t false_block_ordinal;
  // Condition load width in bytes.
  uint8_t condition_width;
  // Reserved bytes that must be zero in version 0.
  uint8_t reserved0[7];
} iree_hal_amdgpu_command_buffer_cond_branch_command_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_cond_branch_command_t) == 32,
    "conditional branch command size must remain qword aligned");

// Return terminator.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_return_command_t {
  // Common command record header.
  iree_hal_amdgpu_command_buffer_command_header_t header;
} iree_hal_amdgpu_command_buffer_return_command_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_return_command_t) == 16,
    "return command size must remain qword aligned");

// Returns the byte length encoded in a command header.
static inline size_t iree_hal_amdgpu_command_buffer_command_length(
    const iree_hal_amdgpu_command_buffer_command_header_t* command) {
  return (size_t)command->length_qwords *
         IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT;
}

// Returns the first command in |block|.
static inline iree_hal_amdgpu_command_buffer_command_header_t*
iree_hal_amdgpu_command_buffer_block_commands(
    iree_hal_amdgpu_command_buffer_block_header_t* block) {
  uint8_t* block_base = (uint8_t*)block;
  uint8_t* command_data = block_base + block->command_offset;
  return (iree_hal_amdgpu_command_buffer_command_header_t*)command_data;
}

// Returns the first command in |block|.
static inline const iree_hal_amdgpu_command_buffer_command_header_t*
iree_hal_amdgpu_command_buffer_block_commands_const(
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  const uint8_t* block_base = (const uint8_t*)block;
  const uint8_t* command_data = block_base + block->command_offset;
  return (const iree_hal_amdgpu_command_buffer_command_header_t*)command_data;
}

// Returns the command following |command|.
static inline iree_hal_amdgpu_command_buffer_command_header_t*
iree_hal_amdgpu_command_buffer_command_next(
    iree_hal_amdgpu_command_buffer_command_header_t* command) {
  uint8_t* command_base = (uint8_t*)command;
  uint8_t* next_command =
      command_base + iree_hal_amdgpu_command_buffer_command_length(command);
  return (iree_hal_amdgpu_command_buffer_command_header_t*)next_command;
}

// Returns the command following |command|.
static inline const iree_hal_amdgpu_command_buffer_command_header_t*
iree_hal_amdgpu_command_buffer_command_next_const(
    const iree_hal_amdgpu_command_buffer_command_header_t* command) {
  const uint8_t* command_base = (const uint8_t*)command;
  const uint8_t* next_command =
      command_base + iree_hal_amdgpu_command_buffer_command_length(command);
  return (const iree_hal_amdgpu_command_buffer_command_header_t*)next_command;
}

// Returns the first fixup record in |block|.
static inline iree_hal_amdgpu_command_buffer_fixup_t*
iree_hal_amdgpu_command_buffer_block_fixups(
    iree_hal_amdgpu_command_buffer_block_header_t* block) {
  uint8_t* block_base = (uint8_t*)block;
  uint8_t* fixup_data = block_base + block->fixup_offset;
  return (iree_hal_amdgpu_command_buffer_fixup_t*)fixup_data;
}

// Returns the first fixup record in |block|.
static inline const iree_hal_amdgpu_command_buffer_fixup_t*
iree_hal_amdgpu_command_buffer_block_fixups_const(
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  const uint8_t* block_base = (const uint8_t*)block;
  const uint8_t* fixup_data = block_base + block->fixup_offset;
  return (const iree_hal_amdgpu_command_buffer_fixup_t*)fixup_data;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_COMMAND_BUFFER_H_
