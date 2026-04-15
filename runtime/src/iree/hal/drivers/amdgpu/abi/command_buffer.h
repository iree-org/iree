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
  // Required alignment for all command records and binding source records.
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

// Binding source flags used to form HAL dispatch kernarg pointer prefixes.
typedef enum iree_hal_amdgpu_command_buffer_binding_source_flag_bits_e {
  IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC = 1u << 0,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS = 1u
                                                                           << 1,
} iree_hal_amdgpu_command_buffer_binding_source_flag_bits_t;

// Dispatch command flags.
typedef enum iree_hal_amdgpu_command_buffer_dispatch_flag_bits_e {
  IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_NONE = 0u,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS = 1u << 0,
} iree_hal_amdgpu_command_buffer_dispatch_flag_bits_t;

// Kernarg formation strategy for a dispatch command.
typedef enum iree_hal_amdgpu_command_buffer_kernarg_strategy_e {
  IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_HAL = 0,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT = 1,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_INDIRECT = 2,
} iree_hal_amdgpu_command_buffer_kernarg_strategy_t;

// Binding reference kind constants embedded in command records.
enum iree_hal_amdgpu_command_buffer_binding_kind_e {
  IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_INVALID = 0,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_STATIC = 1,
  IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_DYNAMIC = 2,
};
// Compact binding reference kind storage.
typedef uint8_t iree_hal_amdgpu_command_buffer_binding_kind_t;

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
  // Byte offset from this header to the first binding source record.
  uint32_t binding_source_offset;
  // Number of command records in this block, including terminators.
  uint16_t command_count;
  // Number of binding source records in this block.
  uint16_t binding_source_count;
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
} iree_hal_amdgpu_command_buffer_command_header_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_command_header_t) == 8,
    "command header size is part of the command-buffer ABI");

// Source record used to emit one HAL ABI dispatch binding pointer.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_binding_source_t {
  // Static source: final raw device pointer. Dynamic source: byte offset added
  // to the queue_execute binding table slot.
  uint64_t offset_or_pointer;
  // Dynamic source binding table slot. Must be zero for static sources.
  uint32_t slot;
  // Source flags from
  // iree_hal_amdgpu_command_buffer_binding_source_flag_bits_t.
  uint8_t flags;
  // Reserved bytes that must be zero in version 0.
  uint8_t reserved0[3];
} iree_hal_amdgpu_command_buffer_binding_source_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_binding_source_t) == 16,
    "binding source size is part of the command-buffer ABI");

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
    sizeof(iree_hal_amdgpu_command_buffer_barrier_command_t) == 16,
    "barrier command size must remain qword aligned");

// Dispatch command record.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_dispatch_command_t {
  // Common command record header.
  iree_hal_amdgpu_command_buffer_command_header_t header;
  // HSA kernel object for the command buffer's selected physical device.
  uint64_t kernel_object;
  // Byte offset from the block header to this dispatch's first binding source.
  uint32_t binding_source_offset;
  // Byte offset from this command record to constants/implicit tail bytes.
  uint32_t tail_payload_offset;
  // Number of HAL ABI binding pointer slots emitted before the tail payload.
  uint16_t binding_count;
  // Total kernarg reservation size in 8-byte qwords.
  uint16_t kernarg_length_qwords;
  // Tail payload size in 8-byte qwords.
  uint16_t tail_length_qwords;
  // Kernarg strategy from iree_hal_amdgpu_command_buffer_kernarg_strategy_t.
  uint8_t kernarg_strategy;
  // Dispatch flags from iree_hal_amdgpu_command_buffer_dispatch_flag_bits_t.
  uint8_t dispatch_flags;
  // AQL dispatch packet setup field.
  uint16_t setup;
  // AQL dispatch packet workgroup size fields.
  uint16_t workgroup_size[3];
  // Kernarg qword offset of implicit args, or UINT16_MAX when absent.
  uint16_t implicit_args_offset_qwords;
  // AQL dispatch packet grid size fields.
  uint32_t grid_size[3];
  // AQL dispatch packet private segment size field.
  uint32_t private_segment_size;
  // AQL dispatch packet group segment size field.
  uint32_t group_segment_size;
} iree_hal_amdgpu_command_buffer_dispatch_command_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t) == 64,
    "dispatch command size must remain qword aligned");

// Fill command record.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_fill_command_t {
  // Common command record header.
  iree_hal_amdgpu_command_buffer_command_header_t header;
  // Byte offset into the target buffer reference.
  uint64_t target_offset;
  // Byte length of the target range.
  uint64_t length;
  // Repeated fill pattern stored in the low bytes.
  uint64_t pattern;
  // Static buffer ordinal or dynamic binding-table slot.
  uint32_t target_ordinal;
  // Binding reference kind from iree_hal_amdgpu_command_buffer_binding_kind_t.
  iree_hal_amdgpu_command_buffer_binding_kind_t target_kind;
  // Byte length of the fill pattern.
  uint8_t pattern_length;
  // Reserved bytes that must be zero in version 0.
  uint8_t reserved0[2];
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
  // Byte offset into the source buffer reference.
  uint64_t source_offset;
  // Byte offset into the target buffer reference.
  uint64_t target_offset;
  // Static buffer ordinal or dynamic binding-table slot for the source.
  uint32_t source_ordinal;
  // Static buffer ordinal or dynamic binding-table slot for the target.
  uint32_t target_ordinal;
  // Source reference kind from iree_hal_amdgpu_command_buffer_binding_kind_t.
  iree_hal_amdgpu_command_buffer_binding_kind_t source_kind;
  // Target reference kind from iree_hal_amdgpu_command_buffer_binding_kind_t.
  iree_hal_amdgpu_command_buffer_binding_kind_t target_kind;
  // Reserved bytes that must be zero in version 0.
  uint8_t reserved0[6];
} iree_hal_amdgpu_command_buffer_copy_command_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_copy_command_t) == 48,
    "copy command size must remain qword aligned");

// Update command record.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_update_command_t {
  // Common command record header.
  iree_hal_amdgpu_command_buffer_command_header_t header;
  // Command-buffer rodata segment ordinal containing the update payload.
  uint64_t rodata_ordinal;
  // Byte offset into the target buffer reference.
  uint64_t target_offset;
  // Byte length of the update payload and target range.
  uint32_t length;
  // Static buffer ordinal or dynamic binding-table slot for the target.
  uint32_t target_ordinal;
  // Target reference kind from iree_hal_amdgpu_command_buffer_binding_kind_t.
  iree_hal_amdgpu_command_buffer_binding_kind_t target_kind;
  // Reserved bytes that must be zero in version 0.
  uint8_t reserved0[7];
} iree_hal_amdgpu_command_buffer_update_command_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_update_command_t) == 40,
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
    sizeof(iree_hal_amdgpu_command_buffer_branch_command_t) == 16,
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
    sizeof(iree_hal_amdgpu_command_buffer_cond_branch_command_t) == 24,
    "conditional branch command size must remain qword aligned");

// Return terminator.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_return_command_t {
  // Common command record header.
  iree_hal_amdgpu_command_buffer_command_header_t header;
} iree_hal_amdgpu_command_buffer_return_command_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_return_command_t) == 8,
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

// Returns the first binding source record in |block|.
static inline iree_hal_amdgpu_command_buffer_binding_source_t*
iree_hal_amdgpu_command_buffer_block_binding_sources(
    iree_hal_amdgpu_command_buffer_block_header_t* block) {
  uint8_t* block_base = (uint8_t*)block;
  uint8_t* binding_source_data = block_base + block->binding_source_offset;
  return (iree_hal_amdgpu_command_buffer_binding_source_t*)binding_source_data;
}

// Returns the first binding source record in |block|.
static inline const iree_hal_amdgpu_command_buffer_binding_source_t*
iree_hal_amdgpu_command_buffer_block_binding_sources_const(
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  const uint8_t* block_base = (const uint8_t*)block;
  const uint8_t* binding_source_data =
      block_base + block->binding_source_offset;
  return (const iree_hal_amdgpu_command_buffer_binding_source_t*)
      binding_source_data;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_COMMAND_BUFFER_H_
