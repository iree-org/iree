// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/block_command_buffer.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/local_task/block_builder.h"
#include "iree/hal/drivers/local_task/block_isa.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/local_executable.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// iree_hal_block_command_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_block_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  iree_arena_block_pool_t* block_pool;

  // Retains resources (buffers, executables) used during recording.
  iree_hal_resource_set_t* resource_set;

  // Arena for recording-time allocations (spans for direct fixups).
  // Separate from the builder's block pool usage: the arena allocates
  // small objects (24 bytes per binding) that live until the CB is destroyed.
  iree_arena_allocator_t arena;

  // Block builder compiling HAL commands into block ISA.
  iree_hal_cmd_block_builder_t builder;

  // Recording produced by end(). Consumed by the queue or released on destroy.
  iree_hal_cmd_block_recording_t recording;
} iree_hal_block_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_block_command_buffer_vtable;

static iree_hal_block_command_buffer_t* iree_hal_block_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_block_command_buffer_vtable);
  return (iree_hal_block_command_buffer_t*)base_value;
}

//===----------------------------------------------------------------------===//
// Binding helpers
//===----------------------------------------------------------------------===//

// Maps a HAL buffer binding to a host pointer and creates a direct fixup entry.
// The fixup references an arena-allocated span that lives until CB destroy.
//
// |data_index| is the absolute index into .data binding_ptrs[] where the
// resolved pointer will be stored at block entry.
static iree_status_t iree_hal_block_command_buffer_map_binding(
    iree_hal_block_command_buffer_t* command_buffer, iree_hal_buffer_t* buffer,
    iree_device_size_t offset, iree_device_size_t length, uint16_t data_index,
    iree_hal_cmd_fixup_t* out_fixup) {
  // Map the buffer to get a stable host pointer. For local_task buffers
  // (heap-backed), persistent mapping always succeeds.
  iree_hal_buffer_mapping_t mapping = {{0}};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_PERSISTENT, IREE_HAL_MEMORY_ACCESS_ANY,
      offset, length, &mapping));

  // Allocate a span from the CB's arena. The span stores the host pointer
  // and remains valid until the CB is destroyed (after execution completes).
  iree_async_span_t* span = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&command_buffer->arena,
                                           sizeof(*span), (void**)&span));
  *span = iree_async_span_from_ptr(mapping.contents.data,
                                   mapping.contents.data_length);

  // Build the direct fixup entry. The offset is 0 because the map_range
  // call already applied the buffer offset — the span pointer points to
  // exactly the right location.
  memset(out_fixup, 0, sizeof(*out_fixup));
  out_fixup->span = span;
  out_fixup->offset = 0;
  out_fixup->data_index = data_index;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_block_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only one-shot command buffer usage is supported");
  }
  if (binding_capacity > 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t total_size = 0;
  iree_host_size_t validation_state_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_hal_block_command_buffer_t), &total_size,
              IREE_STRUCT_FIELD(iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                                uint8_t, &validation_state_offset)));

  iree_hal_block_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_allocator_malloc(host_allocator, total_size,
                                               (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        device_allocator, mode, command_categories, queue_affinity,
        binding_capacity, (uint8_t*)command_buffer + validation_state_offset,
        &iree_hal_block_command_buffer_vtable, &command_buffer->base);
    command_buffer->host_allocator = host_allocator;
    command_buffer->block_pool = block_pool;
    iree_arena_initialize(block_pool, &command_buffer->arena);
    iree_hal_cmd_block_builder_initialize(block_pool, &command_buffer->builder);
    memset(&command_buffer->recording, 0, sizeof(command_buffer->recording));
    status = iree_hal_resource_set_allocate(block_pool,
                                            &command_buffer->resource_set);
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    if (command_buffer) {
      iree_hal_cmd_block_builder_deinitialize(&command_buffer->builder);
      iree_arena_deinitialize(&command_buffer->arena);
      iree_allocator_free(host_allocator, command_buffer);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_block_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cmd_block_recording_release(&command_buffer->recording);
  iree_hal_cmd_block_builder_deinitialize(&command_buffer->builder);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_block_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_block_command_buffer_vtable);
}

const iree_hal_cmd_block_recording_t* iree_hal_block_command_buffer_recording(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  return &command_buffer->recording;
}

//===----------------------------------------------------------------------===//
// Recording session
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  return iree_hal_cmd_block_builder_begin(&command_buffer->builder);
}

static iree_status_t iree_hal_block_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  return iree_hal_cmd_block_builder_end(&command_buffer->builder,
                                        &command_buffer->recording);
}

//===----------------------------------------------------------------------===//
// Debug groups (no-op)
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_block_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Barriers and events
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  // Block ISA barriers are global: all prior work in the region must complete
  // before the next region begins. Fine-grained memory/buffer barriers are
  // not applicable (CPU execution is cache-coherent).
  return iree_hal_cmd_block_builder_barrier(&command_buffer->builder);
}

static iree_status_t iree_hal_block_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_ok_status();
}

static iree_status_t iree_hal_block_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_ok_status();
}

static iree_status_t iree_hal_block_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  // Treat event waits as global barriers (same as the task CB).
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  return iree_hal_cmd_block_builder_barrier(&command_buffer->builder);
}

//===----------------------------------------------------------------------===//
// Buffer advise (no-op)
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_fill_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_ref.buffer));

  // Claim a .data binding slot before append_cmd (which increments the count).
  uint16_t binding_data_base = command_buffer->builder.total_binding_count;

  // Append the command and reserve fixup storage in one step.
  iree_hal_cmd_fill_t* cmd = NULL;
  iree_hal_cmd_fixup_t* fixups = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_append_cmd(
      &command_buffer->builder, IREE_HAL_CMD_FILL, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_fill_t), 1, 1, 1, (void**)&cmd, &fixups));

  // Map the binding directly into the fixup storage. On failure, roll back
  // the command (cold path — map_range on heap-backed buffers always succeeds).
  iree_status_t status = iree_hal_block_command_buffer_map_binding(
      command_buffer, target_ref.buffer, target_ref.offset, target_ref.length,
      binding_data_base, &fixups[0]);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_cmd_block_builder_pop_cmd(&command_buffer->builder,
                                       sizeof(iree_hal_cmd_fill_t), 1, 1, 1);
    return status;
  }

  cmd->target_binding = binding_data_base;
  cmd->pattern_length = (uint8_t)pattern_length;
  // Offset is 0: already applied by map_range. Length is the fill region size.
  cmd->params.direct.target_offset = 0;
  cmd->params.direct.length = target_ref.length;
  cmd->params.direct.pattern = 0;
  memcpy(&cmd->params.direct.pattern, pattern, pattern_length);

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_update_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_ref.buffer));

  uint16_t binding_data_base = command_buffer->builder.total_binding_count;

  // Command includes trailing inline source data, 8-byte aligned.
  iree_host_size_t cmd_bytes = iree_host_align(
      offsetof(iree_hal_cmd_update_t, source_data) + target_ref.length, 8);

  iree_hal_cmd_update_t* cmd = NULL;
  iree_hal_cmd_fixup_t* fixups = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_append_cmd(
      &command_buffer->builder, IREE_HAL_CMD_UPDATE, IREE_HAL_CMD_FLAG_NONE,
      cmd_bytes, 1, 1, 1, (void**)&cmd, &fixups));

  iree_status_t status = iree_hal_block_command_buffer_map_binding(
      command_buffer, target_ref.buffer, target_ref.offset, target_ref.length,
      binding_data_base, &fixups[0]);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_cmd_block_builder_pop_cmd(&command_buffer->builder, cmd_bytes, 1,
                                       1, 1);
    return status;
  }

  cmd->target_binding = binding_data_base;
  cmd->target_offset = 0;
  cmd->length = target_ref.length;

  // Copy inline source data into the FAM.
  memcpy(cmd->source_data, (const uint8_t*)source_buffer + source_offset,
         (size_t)target_ref.length);

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_copy_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);

  const iree_hal_buffer_t* buffers[2] = {source_ref.buffer, target_ref.buffer};
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, IREE_ARRAYSIZE(buffers), buffers));

  uint16_t binding_data_base = command_buffer->builder.total_binding_count;

  iree_hal_cmd_copy_t* cmd = NULL;
  iree_hal_cmd_fixup_t* fixups = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_append_cmd(
      &command_buffer->builder, IREE_HAL_CMD_COPY, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_copy_t), 2, 2, 1, (void**)&cmd, &fixups));

  iree_status_t status = iree_hal_block_command_buffer_map_binding(
      command_buffer, source_ref.buffer, source_ref.offset, source_ref.length,
      binding_data_base, &fixups[0]);
  if (iree_status_is_ok(status)) {
    status = iree_hal_block_command_buffer_map_binding(
        command_buffer, target_ref.buffer, target_ref.offset, target_ref.length,
        (uint16_t)(binding_data_base + 1), &fixups[1]);
  }
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_cmd_block_builder_pop_cmd(&command_buffer->builder,
                                       sizeof(iree_hal_cmd_copy_t), 2, 2, 1);
    return status;
  }

  cmd->source_binding = binding_data_base;
  cmd->target_binding = (uint16_t)(binding_data_base + 1);
  // Offsets are 0: already applied by map_range.
  cmd->params.direct.source_offset = 0;
  cmd->params.direct.target_offset = 0;
  cmd->params.direct.length = target_ref.length;

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_collective
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not yet implemented on the block ISA");
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_dispatch
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);

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

  // Block ISA resolves function pointers at recording time — the raw pointer
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
  if (IREE_UNLIKELY(bindings.count != dispatch_attrs.binding_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "binding count mismatch, expected %u but was provided %" PRIhsz,
        (uint32_t)dispatch_attrs.binding_count, bindings.count);
  }

  // Retain executable and all bound buffers.
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &executable));
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, bindings.count, bindings.values,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));

  // Validate bindings before touching the builder — all must be non-NULL.
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (IREE_UNLIKELY(!bindings.values[i].buffer)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "required binding %" PRIhsz
                              " is NULL; all bindings must have a valid buffer",
                              i);
    }
  }

  // Claim .data binding slots for all bindings in this dispatch.
  uint16_t binding_data_base = command_buffer->builder.total_binding_count;

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
  IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_append_cmd(
      &command_buffer->builder, IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FLAG_NONE,
      cmd_bytes, (uint16_t)bindings.count, (uint16_t)bindings.count, tile_count,
      (void**)&cmd, &fixups));

  // Map each binding directly into the fixup storage. If any mapping fails,
  // roll back the command (cold path — map_range on heap-backed buffers
  // always succeeds).
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < bindings.count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_block_command_buffer_map_binding(
        command_buffer, bindings.values[i].buffer, bindings.values[i].offset,
        bindings.values[i].length, (uint16_t)(binding_data_base + i),
        &fixups[i]);
  }
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_cmd_block_builder_pop_cmd(&command_buffer->builder, cmd_bytes,
                                       (uint16_t)bindings.count,
                                       (uint16_t)bindings.count, tile_count);
    return status;
  }

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

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_block_command_buffer_vtable = {
        .destroy = iree_hal_block_command_buffer_destroy,
        .begin = iree_hal_block_command_buffer_begin,
        .end = iree_hal_block_command_buffer_end,
        .begin_debug_group = iree_hal_block_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_block_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_block_command_buffer_execution_barrier,
        .signal_event = iree_hal_block_command_buffer_signal_event,
        .reset_event = iree_hal_block_command_buffer_reset_event,
        .wait_events = iree_hal_block_command_buffer_wait_events,
        .advise_buffer = iree_hal_block_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_block_command_buffer_fill_buffer,
        .update_buffer = iree_hal_block_command_buffer_update_buffer,
        .copy_buffer = iree_hal_block_command_buffer_copy_buffer,
        .collective = iree_hal_block_command_buffer_collective,
        .dispatch = iree_hal_block_command_buffer_dispatch,
};
