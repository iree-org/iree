// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_command_buffer_profile.h"

#include "iree/base/alignment.h"

static iree_hal_profile_command_operation_flags_t
iree_hal_amdgpu_aql_command_buffer_profile_binding_kind_flags(
    iree_hal_amdgpu_command_buffer_binding_kind_t kind) {
  switch (kind) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_STATIC:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS;
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_DYNAMIC:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS;
    default:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_NONE;
  }
}

static iree_hal_profile_command_operation_type_t
iree_hal_amdgpu_aql_command_buffer_profile_operation_type(uint8_t opcode) {
  switch (opcode) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BARRIER;
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL;
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COPY;
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_UPDATE;
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_PROFILE_MARKER:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_PROFILE_MARKER;
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BRANCH;
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COND_BRANCH;
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_RETURN;
    default:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_NONE;
  }
}

static iree_hal_profile_command_operation_flags_t
iree_hal_amdgpu_aql_command_buffer_profile_dispatch_binding_flags(
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  if (dispatch_command->binding_count == 0) {
    return IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_NONE;
  }

  iree_hal_profile_command_operation_flags_t flags =
      IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_NONE;
  switch (dispatch_command->kernarg_strategy) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS;
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PATCHED_TEMPLATE:
      if (dispatch_command->payload.patch_source_count != 0) {
        flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS;
      }
      if (dispatch_command->payload.patch_source_count <
          dispatch_command->binding_count) {
        flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS;
      }
      return flags;
    default:
      break;
  }
  if (dispatch_command->binding_source_offset == 0) {
    return IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS;
  }

  const uint8_t* block_base = (const uint8_t*)block;
  const uint32_t binding_source_offset =
      dispatch_command->binding_source_offset;
  const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
      (const iree_hal_amdgpu_command_buffer_binding_source_t*)(block_base +
                                                               binding_source_offset);
  for (uint16_t binding_ordinal = 0;
       binding_ordinal < dispatch_command->binding_count; ++binding_ordinal) {
    flags |= iree_any_bit_set(
                 binding_sources[binding_ordinal].flags,
                 IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC)
                 ? IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS
                 : IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS;
  }
  return flags;
}

static void iree_hal_amdgpu_aql_command_buffer_initialize_profile_operation(
    uint64_t command_buffer_id,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    uint32_t block_command_ordinal,
    const iree_hal_amdgpu_command_buffer_command_header_t* command,
    iree_hal_profile_command_operation_record_t* out_record) {
  iree_hal_profile_command_operation_record_t record =
      iree_hal_profile_command_operation_record_default();
  record.type = iree_hal_amdgpu_aql_command_buffer_profile_operation_type(
      command->opcode);
  record.command_buffer_id = command_buffer_id;
  record.command_index = command->command_index;
  record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_BLOCK_STRUCTURE;
  record.block_ordinal = block->block_ordinal;
  record.block_command_ordinal = block_command_ordinal;
  if (iree_any_bit_set(
          command->flags,
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER)) {
    record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_EXECUTION_BARRIER;
  }

  switch (command->opcode) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER:
      record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_EXECUTION_BARRIER;
      break;
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH: {
      const iree_hal_amdgpu_command_buffer_dispatch_command_t*
          dispatch_command =
              (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)command;
      record.flags |=
          iree_hal_amdgpu_aql_command_buffer_profile_dispatch_binding_flags(
              block, dispatch_command);
      if (iree_any_bit_set(
              dispatch_command->dispatch_flags,
              IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS)) {
        record.flags |=
            IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_INDIRECT_PARAMETERS;
      }
      if (dispatch_command->kernarg_strategy ==
          IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED) {
        record.flags |=
            IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_PREPUBLISHED_ARGUMENTS;
      }
      record.executable_id = dispatch_command->executable_id;
      record.export_ordinal = dispatch_command->export_ordinal;
      record.binding_count = dispatch_command->binding_count;
      record.workgroup_size[0] = dispatch_command->workgroup_size[0];
      record.workgroup_size[1] = dispatch_command->workgroup_size[1];
      record.workgroup_size[2] = dispatch_command->workgroup_size[2];
      if (!iree_any_bit_set(
              dispatch_command->dispatch_flags,
              IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS)) {
        for (iree_host_size_t dimension_ordinal = 0;
             dimension_ordinal < IREE_ARRAYSIZE(record.workgroup_count);
             ++dimension_ordinal) {
          record.workgroup_count[dimension_ordinal] =
              dispatch_command->workgroup_size[dimension_ordinal] == 0
                  ? 0
                  : dispatch_command->grid_size[dimension_ordinal] /
                        dispatch_command->workgroup_size[dimension_ordinal];
        }
      }
      break;
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL: {
      const iree_hal_amdgpu_command_buffer_fill_command_t* fill_command =
          (const iree_hal_amdgpu_command_buffer_fill_command_t*)command;
      record.flags |=
          iree_hal_amdgpu_aql_command_buffer_profile_binding_kind_flags(
              fill_command->target_kind);
      record.target_offset = fill_command->target_offset;
      record.length = fill_command->length;
      record.target_ordinal = fill_command->target_ordinal;
      break;
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY: {
      const iree_hal_amdgpu_command_buffer_copy_command_t* copy_command =
          (const iree_hal_amdgpu_command_buffer_copy_command_t*)command;
      record.flags |=
          iree_hal_amdgpu_aql_command_buffer_profile_binding_kind_flags(
              copy_command->source_kind);
      record.flags |=
          iree_hal_amdgpu_aql_command_buffer_profile_binding_kind_flags(
              copy_command->target_kind);
      record.source_offset = copy_command->source_offset;
      record.target_offset = copy_command->target_offset;
      record.length = copy_command->length;
      record.source_ordinal = copy_command->source_ordinal;
      record.target_ordinal = copy_command->target_ordinal;
      break;
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE: {
      const iree_hal_amdgpu_command_buffer_update_command_t* update_command =
          (const iree_hal_amdgpu_command_buffer_update_command_t*)command;
      record.flags |=
          iree_hal_amdgpu_aql_command_buffer_profile_binding_kind_flags(
              update_command->target_kind);
      record.target_offset = update_command->target_offset;
      record.length = update_command->length;
      record.target_ordinal = update_command->target_ordinal;
      break;
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH: {
      const iree_hal_amdgpu_command_buffer_branch_command_t* branch_command =
          (const iree_hal_amdgpu_command_buffer_branch_command_t*)command;
      record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_CONTROL_FLOW;
      record.target_block_ordinal = branch_command->target_block_ordinal;
      break;
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH: {
      const iree_hal_amdgpu_command_buffer_cond_branch_command_t*
          branch_command =
              (const iree_hal_amdgpu_command_buffer_cond_branch_command_t*)
                  command;
      record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_CONTROL_FLOW;
      record.target_block_ordinal = branch_command->true_block_ordinal;
      record.alternate_block_ordinal = branch_command->false_block_ordinal;
      break;
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
      record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_CONTROL_FLOW;
      break;
    default:
      break;
  }
  *out_record = record;
}

iree_status_t iree_hal_amdgpu_aql_command_buffer_register_profile_operations(
    iree_hal_amdgpu_profile_metadata_registry_t* profile_metadata,
    uint64_t command_buffer_id, const iree_hal_amdgpu_aql_program_t* program,
    iree_allocator_t host_allocator) {
  if (program->command_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &byte_length,
      IREE_STRUCT_FIELD(program->command_count,
                        iree_hal_profile_command_operation_record_t, NULL)));

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, program->command_count);

  iree_hal_profile_command_operation_record_t* records = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, byte_length, (void**)&records);

  iree_host_size_t record_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_hal_amdgpu_command_buffer_block_header_t* block =
             program->first_block;
         block; block = iree_hal_amdgpu_aql_program_block_next(
                    program->block_pool, block)) {
      const iree_hal_amdgpu_command_buffer_command_header_t* command =
          iree_hal_amdgpu_command_buffer_block_commands_const(block);
      for (uint16_t command_ordinal = 0;
           command_ordinal < block->command_count &&
           record_count < program->command_count;
           ++command_ordinal) {
        iree_hal_amdgpu_aql_command_buffer_initialize_profile_operation(
            command_buffer_id, block, command_ordinal, command,
            &records[record_count++]);
        command = iree_hal_amdgpu_command_buffer_command_next_const(command);
      }
    }
  }
  if (iree_status_is_ok(status) && record_count != program->command_count) {
    status =
        iree_make_status(IREE_STATUS_INTERNAL,
                         "profile command-operation count mismatch: expected "
                         "%u but got %" PRIhsz,
                         program->command_count, record_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_metadata_register_command_operations(
        profile_metadata, record_count, records);
  }

  iree_allocator_free(host_allocator, records);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
