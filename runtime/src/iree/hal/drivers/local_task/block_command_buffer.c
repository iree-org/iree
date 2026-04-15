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
#include "iree/hal/drivers/local_task/block_command_ops.h"
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

// Resolves |count| buffer references into fixup entries. For each ref:
//   - Direct (buffer != NULL): maps the buffer persistently and stores the
//     host pointer inline in the fixup (no span indirection).
//   - Indirect (buffer == NULL): records the binding table slot + offset
//     for runtime resolution by the processor.
//
// Direct buffers use PERSISTENT mapping: the buffer is retained by the
// resource_set for the CB's lifetime, so the pointer is stable. Buffers
// that require SCOPED mapping must use the indirect path (binding table
// provided at submit time, where the mapping lifecycle is bounded).
//
// The fixup data_index fields are pre-filled by the builder and preserved.
static iree_status_t iree_hal_block_command_buffer_resolve_refs(
    iree_hal_block_command_buffer_t* command_buffer, iree_host_size_t count,
    const iree_hal_buffer_ref_t* buffer_refs, iree_hal_cmd_fixup_t* fixups) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    // data_index is pre-filled by the builder — write only the other fields.
    fixups[i].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
    if (buffer_refs[i].buffer) {
      // Direct: try to map the buffer now. If the buffer can't be mapped
      // yet (e.g. transient buffer from queue_alloca not yet committed),
      // defer the mapping to drain time — the buffer will be committed by
      // then (semaphore ordering guarantees this).
      iree_hal_buffer_mapping_t mapping = {{0}};
      iree_status_t map_status = iree_hal_buffer_map_range(
          buffer_refs[i].buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
          IREE_HAL_MEMORY_ACCESS_ANY, buffer_refs[i].offset,
          buffer_refs[i].length, &mapping);
      if (iree_status_is_ok(map_status)) {
        fixups[i].host_ptr = mapping.contents.data;
        fixups[i].offset = 0;  // map_range already applied the offset.
        fixups[i].length = mapping.contents.data_length;
        fixups[i].slot = 0;
      } else {
        iree_status_ignore(map_status);
        fixups[i].buffer = buffer_refs[i].buffer;
        fixups[i].offset = buffer_refs[i].offset;
        fixups[i].length = buffer_refs[i].length;
        fixups[i].slot = 0;
        fixups[i].flags = IREE_HAL_CMD_FIXUP_FLAG_DEFERRED;
      }
    } else {
      // Indirect: record binding table slot for runtime resolution.
      fixups[i].host_ptr = NULL;
      fixups[i].offset = buffer_refs[i].offset;
      fixups[i].length = buffer_refs[i].length;
      fixups[i].slot = (uint16_t)buffer_refs[i].buffer_slot;
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

static void iree_hal_block_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer);

iree_status_t iree_hal_block_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (iree_any_bit_set(mode,
                       IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION) &&
      !iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ALLOW_INLINE_EXECUTION requires ONE_SHOT mode");
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
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                (void**)&command_buffer));
  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + validation_state_offset,
      &iree_hal_block_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->block_pool = block_pool;
  iree_hal_cmd_block_builder_initialize(block_pool, &command_buffer->builder);
  memset(&command_buffer->recording, 0, sizeof(command_buffer->recording));

  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);
  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_block_command_buffer_destroy(&command_buffer->base);
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
// Buffer advise
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

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, 1, &target_ref,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  IREE_RETURN_IF_ERROR(
      iree_hal_cmd_build_fill(&command_buffer->builder, target_ref.length,
                              pattern, pattern_length, &fixups, &token));

  iree_status_t status = iree_hal_block_command_buffer_resolve_refs(
      command_buffer, 1, &target_ref, fixups);
  if (!iree_status_is_ok(status)) {
    iree_hal_cmd_build_rollback(&command_buffer->builder, token);
  }
  return status;
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

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, 1, &target_ref,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_build_update(
      &command_buffer->builder, source_buffer, source_offset, target_ref.length,
      &fixups, &token));

  iree_status_t status = iree_hal_block_command_buffer_resolve_refs(
      command_buffer, 1, &target_ref, fixups);
  if (!iree_status_is_ok(status)) {
    iree_hal_cmd_build_rollback(&command_buffer->builder, token);
  }
  return status;
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

  // Retain direct buffer references. Indirect refs (buffer == NULL) are
  // skipped by the strided insert and resolved from the binding table at
  // submit time.
  const iree_hal_buffer_ref_t refs[2] = {source_ref, target_ref};
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, IREE_ARRAYSIZE(refs), refs,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_build_copy(
      &command_buffer->builder, target_ref.length, &fixups, &token));

  iree_status_t status = iree_hal_block_command_buffer_resolve_refs(
      command_buffer, IREE_ARRAYSIZE(refs), refs, fixups);
  if (!iree_status_is_ok(status)) {
    iree_hal_cmd_build_rollback(&command_buffer->builder, token);
  }
  return status;
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

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &executable));
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, bindings.count, bindings.values,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, 1, &config.workgroup_count_ref.buffer));
  }

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_build_dispatch(
      &command_buffer->builder, executable, export_ordinal, config, constants,
      bindings.count, flags, &fixups, &token));

  iree_status_t status = iree_hal_block_command_buffer_resolve_refs(
      command_buffer, bindings.count, bindings.values, fixups);
  if (iree_status_is_ok(status) &&
      iree_hal_dispatch_uses_indirect_parameters(flags)) {
    status = iree_hal_block_command_buffer_resolve_refs(
        command_buffer, 1, &config.workgroup_count_ref,
        &fixups[bindings.count]);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_cmd_build_rollback(&command_buffer->builder, token);
  }
  return status;
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
