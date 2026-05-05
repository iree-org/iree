// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_program_builder.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "iree/hal/drivers/amdgpu/abi/queue.h"

//===----------------------------------------------------------------------===//
// Block Pool Utilities
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_aql_program_block_pool_initialize(
    iree_host_size_t block_size, iree_allocator_t host_allocator,
    iree_arena_block_pool_t* out_block_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, block_size);
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(block_size) ||
                    block_size < IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "command-buffer block size must be a "
                             "power-of-two >= %u bytes",
                             IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE));
  }
  if (IREE_UNLIKELY(block_size > UINT32_MAX)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "command-buffer block size must fit in the block ABI"));
  }

  iree_host_size_t total_block_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          block_size, sizeof(iree_arena_block_t), &total_block_size))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "command-buffer block size overflow"));
  }

  iree_arena_block_pool_initialize(total_block_size, host_allocator,
                                   out_block_pool);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Arena Block Helpers
//===----------------------------------------------------------------------===//

static iree_arena_block_t* iree_hal_amdgpu_aql_program_arena_block(
    iree_arena_block_pool_t* block_pool,
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  return iree_arena_block_trailer(block_pool, (void*)block);
}

static iree_hal_amdgpu_command_buffer_block_header_t*
iree_hal_amdgpu_aql_program_block_from_arena(
    iree_arena_block_pool_t* block_pool, iree_arena_block_t* arena_block) {
  return arena_block ? (iree_hal_amdgpu_command_buffer_block_header_t*)
                           iree_arena_block_ptr(block_pool, arena_block)
                     : NULL;
}

iree_hal_amdgpu_command_buffer_block_header_t*
iree_hal_amdgpu_aql_program_block_next(
    iree_arena_block_pool_t* block_pool,
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  iree_arena_block_t* arena_block =
      iree_hal_amdgpu_aql_program_arena_block(block_pool, block);
  return iree_hal_amdgpu_aql_program_block_from_arena(block_pool,
                                                      arena_block->next);
}

//===----------------------------------------------------------------------===//
// Recording Output
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_aql_program_release(
    iree_hal_amdgpu_aql_program_t* program) {
  if (!program->first_block) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_arena_block_t* first_arena_block =
      iree_hal_amdgpu_aql_program_arena_block(program->block_pool,
                                              program->first_block);
  iree_arena_block_t* last_arena_block = first_arena_block;
  while (last_arena_block->next) {
    last_arena_block = last_arena_block->next;
  }

  iree_arena_block_pool_release(program->block_pool, first_arena_block,
                                last_arena_block);
  memset(program, 0, sizeof(*program));
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_aql_program_builder_initialize(
    iree_arena_block_pool_t* block_pool,
    iree_hal_amdgpu_aql_program_builder_t* out_builder) {
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->block_pool = block_pool;
}

void iree_hal_amdgpu_aql_program_builder_deinitialize(
    iree_hal_amdgpu_aql_program_builder_t* builder) {
  if (!builder->block_pool) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  if (builder->current_block.header) {
    iree_arena_block_t* arena_block = iree_hal_amdgpu_aql_program_arena_block(
        builder->block_pool, builder->current_block.header);
    arena_block->next = NULL;
    iree_arena_block_pool_release(builder->block_pool, arena_block,
                                  arena_block);
    builder->current_block.header = NULL;
  }

  if (builder->first_block) {
    iree_hal_amdgpu_aql_program_t program = {
        .block_pool = builder->block_pool,
        .first_block = builder->first_block,
    };
    iree_hal_amdgpu_aql_program_release(&program);
  }

  memset(builder, 0, sizeof(*builder));
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Block Management
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_aql_program_builder_begin_block(
    iree_hal_amdgpu_aql_program_builder_t* builder) {
  if (IREE_UNLIKELY(builder->block_count == UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command-buffer block count overflow");
  }

  iree_arena_block_t* arena_block = NULL;
  void* block_data = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_block_pool_acquire(builder->block_pool, &arena_block,
                                        &block_data));

  arena_block->next = NULL;
  iree_hal_amdgpu_command_buffer_block_header_t* block =
      (iree_hal_amdgpu_command_buffer_block_header_t*)block_data;
  memset(block, 0, sizeof(*block));
  block->magic = IREE_HAL_AMDGPU_COMMAND_BUFFER_BLOCK_MAGIC;
  block->version = IREE_HAL_AMDGPU_COMMAND_BUFFER_BLOCK_VERSION_0;
  block->header_length = sizeof(*block);
  block->block_ordinal = builder->block_count;
  block->block_length = (uint32_t)builder->block_pool->usable_block_size;
  block->command_offset = sizeof(*block);
  block->rodata_offset = block->block_length;

  builder->current_block.header = block;
  builder->current_block.command_cursor = (uint8_t*)block + sizeof(*block);
  builder->current_block.binding_source_cursor =
      (uint8_t*)block + builder->block_pool->usable_block_size;
  builder->current_block.command_count = 0;
  builder->current_block.binding_source_count = 0;
  builder->current_block.dispatch_count = 0;
  builder->current_block.indirect_dispatch_count = 0;
  builder->current_block.profile_marker_count = 0;
  builder->current_block.aql_packet_count = 0;
  builder->current_block.kernarg_length = 0;
  builder->current_block.initial_barrier_packet_count = 0;
  builder->current_block.pending_barrier_acquire_scope =
      IREE_HSA_FENCE_SCOPE_NONE;
  builder->current_block.flags = IREE_HAL_AMDGPU_AQL_PROGRAM_BUILDER_FLAG_NONE;
  ++builder->block_count;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdgpu_aql_program_builder_finalize_block(
    iree_hal_amdgpu_aql_program_builder_t* builder) {
  iree_hal_amdgpu_command_buffer_block_header_t* block =
      builder->current_block.header;
  block->command_length = (uint32_t)(builder->current_block.command_cursor -
                                     ((uint8_t*)block + block->command_offset));
  block->binding_source_offset =
      (uint32_t)(builder->current_block.binding_source_cursor -
                 (uint8_t*)block);
  block->command_count = builder->current_block.command_count;
  block->binding_source_count = builder->current_block.binding_source_count;
  block->aql_packet_count = builder->current_block.aql_packet_count;
  block->kernarg_length = builder->current_block.kernarg_length;
  block->dispatch_count = builder->current_block.dispatch_count;
  block->indirect_dispatch_count =
      builder->current_block.indirect_dispatch_count;
  block->profile_marker_count = builder->current_block.profile_marker_count;
  block->initial_barrier_packet_count =
      iree_any_bit_set(
          builder->current_block.flags,
          IREE_HAL_AMDGPU_AQL_PROGRAM_BUILDER_FLAG_HAS_INITIAL_BARRIER_PACKET)
          ? builder->current_block.initial_barrier_packet_count
          : block->aql_packet_count;

  if (block->aql_packet_count > builder->max_block_aql_packet_count) {
    builder->max_block_aql_packet_count = block->aql_packet_count;
  }
  if (block->kernarg_length > builder->max_block_kernarg_length) {
    builder->max_block_kernarg_length = block->kernarg_length;
  }

  iree_arena_block_t* arena_block =
      iree_hal_amdgpu_aql_program_arena_block(builder->block_pool, block);
  if (builder->last_block) {
    iree_arena_block_t* last_arena_block =
        iree_hal_amdgpu_aql_program_arena_block(builder->block_pool,
                                                builder->last_block);
    last_arena_block->next = arena_block;
  } else {
    builder->first_block = block;
  }
  builder->last_block = block;

  builder->current_block.header = NULL;
  builder->current_block.command_cursor = NULL;
  builder->current_block.binding_source_cursor = NULL;
}

static iree_host_size_t iree_hal_amdgpu_aql_program_builder_remaining(
    const iree_hal_amdgpu_aql_program_builder_t* builder) {
  return (iree_host_size_t)(builder->current_block.binding_source_cursor -
                            builder->current_block.command_cursor);
}

static iree_status_t iree_hal_amdgpu_aql_program_builder_append_terminator(
    iree_hal_amdgpu_aql_program_builder_t* builder, uint8_t opcode,
    uint32_t target_block_ordinal) {
  if (IREE_UNLIKELY(builder->current_block.command_count == UINT16_MAX ||
                    builder->command_count == UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command-buffer command count overflow");
  }

  iree_host_size_t command_length = 0;
  if (opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH) {
    command_length = sizeof(iree_hal_amdgpu_command_buffer_branch_command_t);
  } else if (opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN) {
    command_length = sizeof(iree_hal_amdgpu_command_buffer_return_command_t);
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported command-buffer terminator opcode %u",
                            opcode);
  }

  if (IREE_UNLIKELY(iree_hal_amdgpu_aql_program_builder_remaining(builder) <
                    command_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command-buffer block has no terminator space");
  }

  iree_hal_amdgpu_command_buffer_command_header_t* header =
      (iree_hal_amdgpu_command_buffer_command_header_t*)
          builder->current_block.command_cursor;
  memset(header, 0, command_length);
  header->opcode = opcode;
  header->length_qwords =
      (uint16_t)(command_length /
                 IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT);
  header->command_index = builder->command_count;

  if (opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH) {
    iree_hal_amdgpu_command_buffer_branch_command_t* branch_command =
        (iree_hal_amdgpu_command_buffer_branch_command_t*)header;
    branch_command->target_block_ordinal = target_block_ordinal;
  }
  builder->current_block.header->terminator_opcode = opcode;
  builder->current_block.header->terminator_target_block_ordinal =
      opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH
          ? target_block_ordinal
          : 0;

  builder->current_block.command_cursor += command_length;
  ++builder->command_count;
  ++builder->current_block.command_count;
  return iree_ok_status();
}

static iree_host_size_t iree_hal_amdgpu_aql_program_terminator_reserve(void) {
  return sizeof(iree_hal_amdgpu_command_buffer_branch_command_t);
}

static iree_status_t iree_hal_amdgpu_aql_program_builder_split_block(
    iree_hal_amdgpu_aql_program_builder_t* builder) {
  if (IREE_UNLIKELY(builder->block_count == UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command-buffer block count overflow");
  }
  const iree_hal_amdgpu_aql_program_builder_flags_t carried_flags =
      builder->current_block.flags &
      IREE_HAL_AMDGPU_AQL_PROGRAM_BUILDER_FLAG_HAS_PENDING_EXECUTION_BARRIER;
  const uint8_t carried_acquire_scope =
      carried_flags ? builder->current_block.pending_barrier_acquire_scope
                    : IREE_HSA_FENCE_SCOPE_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_terminator(
      builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH,
      builder->block_count));
  iree_hal_amdgpu_aql_program_builder_finalize_block(builder);
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_program_builder_begin_block(builder));
  builder->current_block.flags |= carried_flags;
  builder->current_block.pending_barrier_acquire_scope = carried_acquire_scope;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_program_builder_validate_command(
    const iree_hal_amdgpu_aql_program_builder_t* builder, uint8_t opcode,
    iree_host_size_t command_length, uint16_t binding_source_count,
    iree_host_size_t* out_required_length,
    iree_host_size_t* out_binding_source_length) {
  if (IREE_UNLIKELY(!builder->current_block.header)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command-buffer builder is not recording");
  }
  if (IREE_UNLIKELY(opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_INVALID ||
                    opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH ||
                    opcode ==
                        IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH ||
                    opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "opcode %u cannot be appended as a work command",
                            opcode);
  }
  if (IREE_UNLIKELY(
          command_length <
              sizeof(iree_hal_amdgpu_command_buffer_command_header_t) ||
          !iree_host_size_has_alignment(
              command_length,
              IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT) ||
          command_length / IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT >
              UINT16_MAX)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command length must be qword-aligned and fit in "
                            "uint16 qword units");
  }

  iree_host_size_t binding_source_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &binding_source_length,
      IREE_STRUCT_FIELD(binding_source_count,
                        iree_hal_amdgpu_command_buffer_binding_source_t,
                        NULL)));

  iree_host_size_t required_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          command_length, binding_source_length, &required_length))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command record size overflow");
  }
  *out_required_length = required_length;
  *out_binding_source_length = binding_source_length;
  return iree_ok_status();
}

static bool iree_hal_amdgpu_aql_program_command_fits_empty_block(
    const iree_hal_amdgpu_aql_program_builder_t* builder,
    iree_host_size_t required_length) {
  const iree_host_size_t empty_available =
      builder->block_pool->usable_block_size -
      sizeof(iree_hal_amdgpu_command_buffer_block_header_t);
  iree_host_size_t required_with_terminator = 0;
  if (!iree_host_size_checked_add(
          required_length, iree_hal_amdgpu_aql_program_terminator_reserve(),
          &required_with_terminator)) {
    return false;
  }
  return required_with_terminator <= empty_available;
}

static bool iree_hal_amdgpu_aql_program_command_fits_current_block(
    const iree_hal_amdgpu_aql_program_builder_t* builder,
    uint16_t binding_source_count, uint32_t aql_packet_count,
    uint32_t kernarg_length) {
  if (builder->current_block.command_count > UINT16_MAX - 2) return false;
  if (binding_source_count >
      UINT16_MAX - builder->current_block.binding_source_count) {
    return false;
  }
  if (aql_packet_count > UINT32_MAX - builder->current_block.aql_packet_count) {
    return false;
  }
  if (kernarg_length > UINT32_MAX - builder->current_block.kernarg_length) {
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Recording
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_aql_program_builder_begin(
    iree_hal_amdgpu_aql_program_builder_t* builder) {
  if (IREE_UNLIKELY(!builder->block_pool)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-buffer builder requires a block pool");
  }
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(
                        builder->block_pool->usable_block_size) ||
                    builder->block_pool->usable_block_size <
                        IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-buffer block pool must have power-of-two "
                            "usable blocks >= %u bytes",
                            IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE);
  }
  if (IREE_UNLIKELY(builder->block_pool->usable_block_size > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command-buffer block pool usable size must fit in the block ABI");
  }
  if (IREE_UNLIKELY(builder->current_block.header || builder->first_block)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command-buffer builder already has a recording");
  }
  builder->last_payload_command = NULL;
  return iree_hal_amdgpu_aql_program_builder_begin_block(builder);
}

iree_status_t iree_hal_amdgpu_aql_program_builder_end(
    iree_hal_amdgpu_aql_program_builder_t* builder,
    iree_hal_amdgpu_aql_program_t* out_program) {
  if (IREE_UNLIKELY(!out_program)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-buffer program output is required");
  }
  memset(out_program, 0, sizeof(*out_program));

  if (IREE_UNLIKELY(!builder->current_block.header)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command-buffer builder is not recording");
  }

  iree_status_t status = iree_hal_amdgpu_aql_program_builder_append_terminator(
      builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN,
      /*target_block_ordinal=*/0);
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_aql_program_builder_finalize_block(builder);
    *out_program = (iree_hal_amdgpu_aql_program_t){
        .block_pool = builder->block_pool,
        .first_block = builder->first_block,
        .block_count = builder->block_count,
        .command_count = builder->command_count,
        .max_block_aql_packet_count = builder->max_block_aql_packet_count,
        .max_block_kernarg_length = builder->max_block_kernarg_length,
    };
    builder->first_block = NULL;
    builder->last_block = NULL;
    builder->last_payload_command = NULL;
    builder->block_count = 0;
    builder->command_count = 0;
    builder->max_block_aql_packet_count = 0;
    builder->max_block_kernarg_length = 0;
  }
  return status;
}

iree_status_t iree_hal_amdgpu_aql_program_builder_append_command(
    iree_hal_amdgpu_aql_program_builder_t* builder, uint8_t opcode,
    uint8_t flags, iree_host_size_t command_length,
    uint16_t binding_source_count, uint32_t aql_packet_count,
    uint32_t kernarg_length,
    iree_hal_amdgpu_command_buffer_command_header_t** out_command,
    iree_hal_amdgpu_command_buffer_binding_source_t** out_binding_sources) {
  if (IREE_UNLIKELY(!out_command ||
                    (binding_source_count > 0 && !out_binding_sources))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command output pointers are required");
  }
  *out_command = NULL;
  if (out_binding_sources) *out_binding_sources = NULL;

  if (IREE_UNLIKELY(builder->command_count == UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command-buffer command count overflow");
  }

  iree_host_size_t required_length = 0;
  iree_host_size_t binding_source_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_validate_command(
      builder, opcode, command_length, binding_source_count, &required_length,
      &binding_source_length));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_aql_program_command_fits_empty_block(
          builder, required_length))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command record and binding sources cannot fit in "
                            "one command-buffer block");
  }

  iree_host_size_t required_with_terminator = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          required_length, iree_hal_amdgpu_aql_program_terminator_reserve(),
          &required_with_terminator))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command record size overflow");
  }
  if (iree_hal_amdgpu_aql_program_builder_remaining(builder) <
          required_with_terminator ||
      !iree_hal_amdgpu_aql_program_command_fits_current_block(
          builder, binding_source_count, aql_packet_count, kernarg_length)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_program_builder_split_block(builder));
    if (IREE_UNLIKELY(builder->command_count == UINT32_MAX)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "command-buffer command count overflow");
    }
  }

  iree_hal_amdgpu_command_buffer_command_header_t* command =
      (iree_hal_amdgpu_command_buffer_command_header_t*)
          builder->current_block.command_cursor;
  memset(command, 0, command_length);
  builder->current_block.command_cursor += command_length;

  iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources = NULL;
  if (binding_source_length > 0) {
    builder->current_block.binding_source_cursor -= binding_source_length;
    binding_sources = (iree_hal_amdgpu_command_buffer_binding_source_t*)
                          builder->current_block.binding_source_cursor;
    memset(binding_sources, 0, binding_source_length);
  }

  uint8_t command_flags = flags;
  if (kernarg_length != 0) {
    command_flags |=
        IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS;
  }
  const bool command_has_payload = aql_packet_count != 0;
  const bool has_pending_execution_barrier = iree_any_bit_set(
      builder->current_block.flags,
      IREE_HAL_AMDGPU_AQL_PROGRAM_BUILDER_FLAG_HAS_PENDING_EXECUTION_BARRIER);
  const bool command_forces_barrier = iree_any_bit_set(
      command_flags, IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER);
  const bool command_has_barrier_packet =
      command_has_payload &&
      (has_pending_execution_barrier || command_forces_barrier);

  command->opcode = opcode;
  if (command_has_barrier_packet) {
    uint8_t acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
    uint8_t release_scope = IREE_HSA_FENCE_SCOPE_NONE;
    if (has_pending_execution_barrier) {
      acquire_scope = builder->current_block.pending_barrier_acquire_scope;
    }
    if (command_forces_barrier) {
      if (acquire_scope < IREE_HSA_FENCE_SCOPE_AGENT) {
        acquire_scope = IREE_HSA_FENCE_SCOPE_AGENT;
      }
      release_scope = IREE_HSA_FENCE_SCOPE_AGENT;
    }
    command_flags |= IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER;
    command_flags =
        iree_hal_amdgpu_command_buffer_command_flags_set_fence_scopes(
            command_flags, acquire_scope, release_scope);
  }
  command->flags = command_flags;
  command->length_qwords =
      (uint16_t)(command_length /
                 IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT);
  command->command_index = builder->command_count;

  if (command_has_barrier_packet &&
      !iree_any_bit_set(
          builder->current_block.flags,
          IREE_HAL_AMDGPU_AQL_PROGRAM_BUILDER_FLAG_HAS_INITIAL_BARRIER_PACKET)) {
    builder->current_block.initial_barrier_packet_count =
        builder->current_block.aql_packet_count + 1;
    builder->current_block.flags |=
        IREE_HAL_AMDGPU_AQL_PROGRAM_BUILDER_FLAG_HAS_INITIAL_BARRIER_PACKET;
  }
  if (opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER) {
    builder->current_block.flags |=
        IREE_HAL_AMDGPU_AQL_PROGRAM_BUILDER_FLAG_HAS_PENDING_EXECUTION_BARRIER;
  } else if (command_has_payload) {
    builder->current_block.flags &=
        ~IREE_HAL_AMDGPU_AQL_PROGRAM_BUILDER_FLAG_HAS_PENDING_EXECUTION_BARRIER;
    builder->current_block.pending_barrier_acquire_scope =
        IREE_HSA_FENCE_SCOPE_NONE;
  }
  if (command_has_payload) {
    builder->last_payload_command = command;
  }

  ++builder->command_count;
  ++builder->current_block.command_count;
  if (opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH) {
    ++builder->current_block.dispatch_count;
  } else if (opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_PROFILE_MARKER) {
    ++builder->current_block.profile_marker_count;
  }
  builder->current_block.binding_source_count += binding_source_count;
  builder->current_block.aql_packet_count += aql_packet_count;
  builder->current_block.kernarg_length += kernarg_length;

  *out_command = command;
  if (out_binding_sources) *out_binding_sources = binding_sources;
  return iree_ok_status();
}

void iree_hal_amdgpu_aql_program_builder_set_pending_barrier_scopes(
    iree_hal_amdgpu_aql_program_builder_t* builder, uint8_t acquire_scope,
    uint8_t release_scope) {
  if (!iree_any_bit_set(
          builder->current_block.flags,
          IREE_HAL_AMDGPU_AQL_PROGRAM_BUILDER_FLAG_HAS_PENDING_EXECUTION_BARRIER)) {
    return;
  }
  if (builder->last_payload_command) {
    const uint8_t current_acquire_scope =
        iree_hal_amdgpu_command_buffer_command_flags_acquire_scope(
            builder->last_payload_command->flags);
    const uint8_t current_release_scope =
        iree_hal_amdgpu_command_buffer_command_flags_release_scope(
            builder->last_payload_command->flags);
    if (release_scope > current_release_scope) {
      builder->last_payload_command->flags =
          iree_hal_amdgpu_command_buffer_command_flags_set_fence_scopes(
              builder->last_payload_command->flags, current_acquire_scope,
              release_scope);
    }
  }
  if (acquire_scope > builder->current_block.pending_barrier_acquire_scope) {
    builder->current_block.pending_barrier_acquire_scope = acquire_scope;
  }
}
