// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/block_command_ops.h"

#include <stddef.h>
#include <string.h>

#include "iree/hal/local/local_executable.h"

//===----------------------------------------------------------------------===//
// Rollback support
//===----------------------------------------------------------------------===//

void iree_hal_cmd_build_rollback(iree_hal_cmd_block_builder_t* builder,
                                 iree_hal_cmd_build_token_t token) {
  iree_hal_cmd_block_builder_pop_cmd(builder, token.cmd_bytes,
                                     token.fixup_count, token.binding_count,
                                     token.tile_count);
}

//===----------------------------------------------------------------------===//
// FILL
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_cmd_build_fill(iree_hal_cmd_block_builder_t* builder,
                                      iree_device_size_t length,
                                      const void* pattern,
                                      iree_host_size_t pattern_length,
                                      iree_hal_cmd_fixup_t** out_fixups,
                                      iree_hal_cmd_build_token_t* out_token) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(pattern);
  IREE_ASSERT_ARGUMENT(out_fixups);
  IREE_ASSERT_ARGUMENT(out_token);

  uint16_t binding_data_base = builder->total_binding_count;

  iree_hal_cmd_fill_t* cmd = NULL;
  iree_hal_cmd_fixup_t* fixups = NULL;
  out_token->cmd_bytes = sizeof(iree_hal_cmd_fill_t);
  out_token->fixup_count = 1;
  out_token->binding_count = 1;
  out_token->tile_count = 1;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_append_cmd(
      builder, IREE_HAL_CMD_FILL, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_fill_t), 1, 1, 1, (void**)&cmd, &fixups));

  cmd->target_binding = binding_data_base;
  cmd->pattern_length = (uint8_t)pattern_length;
  cmd->params.direct.target_offset = 0;
  cmd->params.direct.length = length;
  cmd->params.direct.pattern = 0;
  memcpy(&cmd->params.direct.pattern, pattern, pattern_length);

  // Pre-fill fixup data_index. Caller sets span.
  fixups[0].data_index = binding_data_base;

  *out_fixups = fixups;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// COPY
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_cmd_build_copy(iree_hal_cmd_block_builder_t* builder,
                                      iree_device_size_t length,
                                      iree_hal_cmd_fixup_t** out_fixups,
                                      iree_hal_cmd_build_token_t* out_token) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_fixups);
  IREE_ASSERT_ARGUMENT(out_token);

  uint16_t binding_data_base = builder->total_binding_count;

  iree_hal_cmd_copy_t* cmd = NULL;
  iree_hal_cmd_fixup_t* fixups = NULL;
  out_token->cmd_bytes = sizeof(iree_hal_cmd_copy_t);
  out_token->fixup_count = 2;
  out_token->binding_count = 2;
  out_token->tile_count = 1;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_append_cmd(
      builder, IREE_HAL_CMD_COPY, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_copy_t), 2, 2, 1, (void**)&cmd, &fixups));

  cmd->source_binding = binding_data_base;
  cmd->target_binding = (uint16_t)(binding_data_base + 1);
  cmd->params.direct.source_offset = 0;
  cmd->params.direct.target_offset = 0;
  cmd->params.direct.length = length;

  // Pre-fill fixup data_indices. Caller sets spans.
  fixups[0].data_index = binding_data_base;
  fixups[1].data_index = (uint16_t)(binding_data_base + 1);

  *out_fixups = fixups;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// UPDATE
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_cmd_build_update(iree_hal_cmd_block_builder_t* builder,
                                        const void* source_buffer,
                                        iree_host_size_t source_offset,
                                        iree_device_size_t length,
                                        iree_hal_cmd_fixup_t** out_fixups,
                                        iree_hal_cmd_build_token_t* out_token) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(out_fixups);
  IREE_ASSERT_ARGUMENT(out_token);

  uint16_t binding_data_base = builder->total_binding_count;

  // Command includes trailing inline source data, 8-byte aligned.
  iree_host_size_t cmd_bytes =
      iree_host_align(offsetof(iree_hal_cmd_update_t, source_data) + length, 8);

  iree_hal_cmd_update_t* cmd = NULL;
  iree_hal_cmd_fixup_t* fixups = NULL;
  out_token->cmd_bytes = cmd_bytes;
  out_token->fixup_count = 1;
  out_token->binding_count = 1;
  out_token->tile_count = 1;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_append_cmd(
      builder, IREE_HAL_CMD_UPDATE, IREE_HAL_CMD_FLAG_NONE, cmd_bytes, 1, 1, 1,
      (void**)&cmd, &fixups));

  cmd->target_binding = binding_data_base;
  cmd->target_offset = 0;
  cmd->length = length;

  // Copy inline source data into the FAM.
  memcpy(cmd->source_data, (const uint8_t*)source_buffer + source_offset,
         (size_t)length);

  // Pre-fill fixup data_index. Caller sets span.
  fixups[0].data_index = binding_data_base;

  *out_fixups = fixups;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// DISPATCH
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_cmd_build_dispatch(
    iree_hal_cmd_block_builder_t* builder, iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_host_size_t binding_count, iree_hal_dispatch_flags_t flags,
    iree_hal_cmd_fixup_t** out_fixups, iree_hal_cmd_build_token_t* out_token) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_fixups);
  IREE_ASSERT_ARGUMENT(out_token);

  // Reject features not yet supported in the block ISA.
  if (iree_hal_dispatch_uses_custom_arguments(flags)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "direct/indirect arguments are not supported in the block ISA");
  }
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "indirect dispatch not yet supported in the block ISA");
  }

  iree_hal_local_executable_t* local_executable =
      iree_hal_local_executable_cast(executable);

  // Block ISA resolves function pointers at build time — the raw pointer
  // is baked into .text for zero-indirection execution. VMVX dispatches
  // through the VM and has dispatch_ptrs == NULL.
  if (!local_executable->dispatch_ptrs) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "block ISA requires direct dispatch (dispatch_ptrs must be non-NULL); "
        "VMVX is not supported");
  }

  iree_hal_executable_dispatch_attrs_v0_t dispatch_attrs = {0};
  if (local_executable->dispatch_attrs) {
    dispatch_attrs = local_executable->dispatch_attrs[export_ordinal];
  }

  // Validate constants.
  if (IREE_UNLIKELY((constants.data_length % sizeof(uint32_t)) != 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "constants must be 4-byte aligned");
  }
  if (IREE_UNLIKELY(constants.data_length !=
                    dispatch_attrs.constant_count * sizeof(uint32_t))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "constant count mismatch, expected %u but was provided %" PRIhsz,
        (uint32_t)dispatch_attrs.constant_count,
        constants.data_length / sizeof(uint32_t));
  }

  // Validate bindings.
  if (IREE_UNLIKELY(binding_count != dispatch_attrs.binding_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "binding count mismatch, expected %u but was provided %" PRIhsz,
        (uint32_t)dispatch_attrs.binding_count, binding_count);
  }

  uint16_t binding_data_base = builder->total_binding_count;

  // Compute command size: fixed header + trailing constants, 8-byte aligned.
  iree_host_size_t cmd_bytes =
      iree_host_align(offsetof(iree_hal_cmd_dispatch_t, constants) +
                          dispatch_attrs.constant_count * sizeof(uint32_t),
                      8);

  // Compute tile count from static workgroup count.
  uint32_t tile_count = config.workgroup_count[0] * config.workgroup_count[1] *
                        config.workgroup_count[2];

  // Append the command and reserve fixup storage for all bindings.
  iree_hal_cmd_dispatch_t* cmd = NULL;
  iree_hal_cmd_fixup_t* fixups = NULL;
  out_token->cmd_bytes = cmd_bytes;
  out_token->fixup_count = (uint16_t)binding_count;
  out_token->binding_count = (uint16_t)binding_count;
  out_token->tile_count = tile_count;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_append_cmd(
      builder, IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FLAG_NONE, cmd_bytes,
      (uint16_t)binding_count, (uint16_t)binding_count, tile_count,
      (void**)&cmd, &fixups));

  // Fill dispatch command fields.
  cmd->constant_count = dispatch_attrs.constant_count;
  cmd->binding_count = dispatch_attrs.binding_count;
  cmd->binding_data_base = binding_data_base;
  cmd->function = local_executable->dispatch_ptrs[export_ordinal];
  cmd->environment = &local_executable->environment;
  cmd->workgroup_size[0] = config.workgroup_size[0];
  cmd->workgroup_size[1] = config.workgroup_size[1];
  cmd->workgroup_size[2] = config.workgroup_size[2];
  cmd->params.direct.workgroup_count[0] = config.workgroup_count[0];
  cmd->params.direct.workgroup_count[1] = config.workgroup_count[1];
  cmd->params.direct.workgroup_count[2] = config.workgroup_count[2];
  cmd->tile_count = tile_count;
  cmd->tiles_per_reservation = 1;
  cmd->local_memory_size =
      (uint32_t)dispatch_attrs.local_memory_pages *
          IREE_HAL_EXECUTABLE_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE +
      config.dynamic_workgroup_local_memory;

  // Copy constants into the FAM.
  if (dispatch_attrs.constant_count > 0) {
    memcpy(cmd->constants, constants.data,
           dispatch_attrs.constant_count * sizeof(uint32_t));
  }

  // Pre-fill fixup data_indices. Caller sets spans.
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    fixups[i].data_index = (uint16_t)(binding_data_base + i);
  }

  *out_fixups = fixups;
  return iree_ok_status();
}
