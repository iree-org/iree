// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_AQL_PROGRAM_BUILDER_H_
#define IREE_HAL_DRIVERS_AMDGPU_AQL_PROGRAM_BUILDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amdgpu/abi/command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Block Pool Utilities
//===----------------------------------------------------------------------===//

enum {
  // Default usable bytes per command-buffer block.
  IREE_HAL_AMDGPU_AQL_PROGRAM_DEFAULT_BLOCK_SIZE = 16 * 1024,
  // Minimum usable bytes per command-buffer block.
  IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE = 256,
};

// Initializes |out_block_pool| so each acquired block has exactly |block_size|
// usable bytes. |block_size| must be a non-zero power of two.
iree_status_t iree_hal_amdgpu_aql_program_block_pool_initialize(
    iree_host_size_t block_size, iree_allocator_t host_allocator,
    iree_arena_block_pool_t* out_block_pool);

//===----------------------------------------------------------------------===//
// Recording Output
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_aql_program_t {
  // Block pool that owns all blocks in this program.
  iree_arena_block_pool_t* block_pool;
  // First finalized block in program order.
  iree_hal_amdgpu_command_buffer_block_header_t* first_block;
  // Number of finalized blocks in the program.
  uint32_t block_count;
  // Number of command records in the program, including terminators.
  uint32_t command_count;
  // Worst-case AQL packet count across all blocks.
  uint32_t max_block_aql_packet_count;
  // Worst-case kernarg byte count across all blocks.
  uint32_t max_block_kernarg_length;
} iree_hal_amdgpu_aql_program_t;

// Releases all blocks in |program| back to its block pool.
void iree_hal_amdgpu_aql_program_release(
    iree_hal_amdgpu_aql_program_t* program);

// Returns the block following |block| in program order.
iree_hal_amdgpu_command_buffer_block_header_t*
iree_hal_amdgpu_aql_program_block_next(
    iree_arena_block_pool_t* block_pool,
    const iree_hal_amdgpu_command_buffer_block_header_t* block);

//===----------------------------------------------------------------------===//
// Program Builder
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_aql_program_builder_t {
  // Block pool used to acquire fixed-capacity recording blocks.
  iree_arena_block_pool_t* block_pool;
  // First finalized block in program order.
  iree_hal_amdgpu_command_buffer_block_header_t* first_block;
  // Last finalized block in program order.
  iree_hal_amdgpu_command_buffer_block_header_t* last_block;
  // Current block being recorded.
  iree_hal_amdgpu_command_buffer_block_header_t* current_block;
  // Forward cursor used to append command records.
  uint8_t* command_cursor;
  // Backward cursor used to append binding source records.
  uint8_t* binding_source_cursor;
  // Number of finalized and current blocks.
  uint32_t block_count;
  // Number of command records emitted into finalized and current blocks.
  uint32_t command_count;
  // Number of command records emitted into the current block.
  uint16_t current_block_command_count;
  // Number of binding source records emitted into the current block.
  uint16_t current_block_binding_source_count;
  // Worst-case AQL packet count emitted by the current block.
  uint32_t current_block_aql_packet_count;
  // Worst-case kernarg byte count emitted by the current block.
  uint32_t current_block_kernarg_length;
  // Worst-case AQL packet count across finalized blocks.
  uint32_t max_block_aql_packet_count;
  // Worst-case kernarg byte count across finalized blocks.
  uint32_t max_block_kernarg_length;
} iree_hal_amdgpu_aql_program_builder_t;

// Initializes |out_builder|. No blocks are acquired until begin().
void iree_hal_amdgpu_aql_program_builder_initialize(
    iree_arena_block_pool_t* block_pool,
    iree_hal_amdgpu_aql_program_builder_t* out_builder);

// Deinitializes |builder| and releases any blocks not transferred by end().
void iree_hal_amdgpu_aql_program_builder_deinitialize(
    iree_hal_amdgpu_aql_program_builder_t* builder);

// Begins a recording session by acquiring the first block.
iree_status_t iree_hal_amdgpu_aql_program_builder_begin(
    iree_hal_amdgpu_aql_program_builder_t* builder);

// Finalizes recording with a return terminator and transfers blocks to
// |out_program|.
iree_status_t iree_hal_amdgpu_aql_program_builder_end(
    iree_hal_amdgpu_aql_program_builder_t* builder,
    iree_hal_amdgpu_aql_program_t* out_program);

// Appends a command record and optional binding source records.
//
// |command_length| must be qword-aligned and include the common command header.
// |aql_packet_count| and |kernarg_length| are worst-case replay resource
// requirements contributed by this command. The builder automatically splits
// blocks and inserts a branch terminator when the current block cannot fit the
// command while preserving room for a terminator.
iree_status_t iree_hal_amdgpu_aql_program_builder_append_command(
    iree_hal_amdgpu_aql_program_builder_t* builder, uint8_t opcode,
    uint8_t flags, iree_host_size_t command_length,
    uint16_t binding_source_count, uint32_t aql_packet_count,
    uint32_t kernarg_length,
    iree_hal_amdgpu_command_buffer_command_header_t** out_command,
    iree_hal_amdgpu_command_buffer_binding_source_t** out_binding_sources);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_AQL_PROGRAM_BUILDER_H_
