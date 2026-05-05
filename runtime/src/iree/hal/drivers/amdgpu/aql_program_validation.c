// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_program_validation.h"

iree_status_t iree_hal_amdgpu_aql_program_validate_block_terminator(
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  switch (block->terminator_opcode) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH:
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
      return iree_ok_status();
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "conditional AQL command-buffer branch replay not yet wired");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AQL command-buffer block %" PRIu32
                              " has no terminator",
                              block->block_ordinal);
  }
}

iree_status_t iree_hal_amdgpu_aql_program_next_linear_block(
    const iree_hal_amdgpu_aql_program_t* program,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    uint32_t target_block_ordinal,
    const iree_hal_amdgpu_command_buffer_block_header_t** out_next_block) {
  *out_next_block = NULL;
  const iree_hal_amdgpu_command_buffer_block_header_t* next_block =
      iree_hal_amdgpu_aql_program_block_next(program->block_pool, block);
  if (IREE_UNLIKELY(!next_block ||
                    next_block->block_ordinal != target_block_ordinal)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "non-linear AQL command-buffer branch replay not yet wired");
  }
  *out_next_block = next_block;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_aql_program_validate_metadata_block_commands(
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(block);
  iree_status_t status = iree_ok_status();
  bool reached_terminator = false;
  for (uint16_t i = 0; i < block->command_count && iree_status_is_ok(status) &&
                       !reached_terminator;
       ++i) {
    const bool is_final_command = i + 1 == block->command_count;
    switch (command->opcode) {
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER:
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH: {
        const iree_hal_amdgpu_command_buffer_branch_command_t* branch_command =
            (const iree_hal_amdgpu_command_buffer_branch_command_t*)command;
        if (IREE_UNLIKELY(!is_final_command)) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "AQL command-buffer block %" PRIu32
                                    " has a branch before the final command",
                                    block->block_ordinal);
          break;
        }
        if (IREE_UNLIKELY(block->terminator_opcode !=
                              IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH ||
                          branch_command->target_block_ordinal !=
                              block->terminator_target_block_ordinal)) {
          status =
              iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                               "AQL command-buffer block %" PRIu32
                               " has mismatched branch terminator metadata",
                               block->block_ordinal);
          break;
        }
        reached_terminator = true;
        break;
      }
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
        if (IREE_UNLIKELY(!is_final_command)) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "AQL command-buffer block %" PRIu32
                                    " has a return before the final command",
                                    block->block_ordinal);
          break;
        }
        if (IREE_UNLIKELY(block->terminator_opcode !=
                          IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN)) {
          status =
              iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                               "AQL command-buffer block %" PRIu32
                               " has mismatched return terminator metadata",
                               block->block_ordinal);
          break;
        }
        reached_terminator = true;
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_PROFILE_MARKER:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AQL command-buffer opcode %u metadata-only replay not yet wired",
            command->opcode);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "malformed AQL command-buffer opcode %u",
                                  command->opcode);
        break;
    }
    if (iree_status_is_ok(status) && !reached_terminator) {
      command = iree_hal_amdgpu_command_buffer_command_next_const(command);
    }
  }
  if (iree_status_is_ok(status) && !reached_terminator) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AQL command-buffer block %" PRIu32
                              " has no terminator",
                              block->block_ordinal);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_aql_program_validate_metadata_only(
    const iree_hal_amdgpu_aql_program_t* program) {
  if (IREE_UNLIKELY(!program->first_block)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AQL command-buffer program has no blocks");
  }

  const iree_hal_amdgpu_command_buffer_block_header_t* block =
      program->first_block;
  bool reached_return = false;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && !reached_return && block) {
    if (IREE_UNLIKELY(block->aql_packet_count != 0)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "metadata-only AQL command-buffer block %" PRIu32
                           " declares %" PRIu32 " AQL packets",
                           block->block_ordinal, block->aql_packet_count);
      break;
    }
    status = iree_hal_amdgpu_aql_program_validate_block_terminator(block);
    if (!iree_status_is_ok(status)) break;
    status =
        iree_hal_amdgpu_aql_program_validate_metadata_block_commands(block);
    if (!iree_status_is_ok(status)) break;

    switch (block->terminator_opcode) {
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH:
        status = iree_hal_amdgpu_aql_program_next_linear_block(
            program, block, block->terminator_target_block_ordinal, &block);
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
        reached_return = true;
        break;
      default:
        IREE_ASSERT_UNREACHABLE("block terminator was already validated");
        break;
    }
  }
  if (iree_status_is_ok(status) && !reached_return) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AQL command-buffer program has no return");
  }
  return status;
}
