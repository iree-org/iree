// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_aql_command_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_aql_command_buffer_t {
  // Base HAL command-buffer resource.
  iree_hal_command_buffer_t base;
  // Host allocator used to allocate the command-buffer object.
  iree_allocator_t host_allocator;
  // Block pool used for durable command-buffer program blocks.
  iree_arena_block_pool_t* block_pool;
  // Builder used only during begin/end recording.
  iree_hal_amdgpu_aql_program_builder_t builder;
  // Program produced by end() and consumed by queue execution.
  iree_hal_amdgpu_aql_program_t program;
} iree_hal_amdgpu_aql_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_amdgpu_aql_command_buffer_vtable;

static iree_hal_amdgpu_aql_command_buffer_t*
iree_hal_amdgpu_aql_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_aql_command_buffer_vtable);
  return (iree_hal_amdgpu_aql_command_buffer_t*)base_value;
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_aql_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer);

iree_status_t iree_hal_amdgpu_aql_command_buffer_create(
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
  if (IREE_UNLIKELY(!block_pool)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-buffer block pool is required");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t total_size = 0;
  iree_host_size_t validation_state_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_hal_amdgpu_aql_command_buffer_t), &total_size,
              IREE_STRUCT_FIELD(iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                                uint8_t, &validation_state_offset)));

  iree_hal_amdgpu_aql_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                (void**)&command_buffer));
  memset(command_buffer, 0, sizeof(*command_buffer));
  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + validation_state_offset,
      &iree_hal_amdgpu_aql_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->block_pool = block_pool;
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool,
                                                 &command_buffer->builder);

  *out_command_buffer = &command_buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdgpu_aql_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_aql_program_release(&command_buffer->program);
  iree_hal_amdgpu_aql_program_builder_deinitialize(&command_buffer->builder);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdgpu_aql_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_amdgpu_aql_command_buffer_vtable);
}

const iree_hal_amdgpu_aql_program_t* iree_hal_amdgpu_aql_command_buffer_program(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  return &command_buffer->program;
}

//===----------------------------------------------------------------------===//
// Recording Session
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_aql_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  iree_hal_amdgpu_aql_program_release(&command_buffer->program);
  return iree_hal_amdgpu_aql_program_builder_begin(&command_buffer->builder);
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);
  return iree_hal_amdgpu_aql_program_builder_end(&command_buffer->builder,
                                                 &command_buffer->program);
}

//===----------------------------------------------------------------------===//
// Debug Groups
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_aql_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Barriers and Events
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_aql_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_amdgpu_aql_command_buffer_t* command_buffer =
      iree_hal_amdgpu_aql_command_buffer_cast(base_command_buffer);

  iree_hal_amdgpu_command_buffer_command_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
      &command_buffer->builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_barrier_command_t),
      /*fixup_count=*/0, /*aql_packet_count=*/0, /*kernarg_length=*/0, &header,
      /*out_fixups=*/NULL));

  iree_hal_amdgpu_command_buffer_barrier_command_t* barrier =
      (iree_hal_amdgpu_command_buffer_barrier_command_t*)header;
  barrier->acquire_scope = 0;
  barrier->release_scope = 0;
  barrier->barrier_flags = 0;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU command-buffer events not implemented");
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU command-buffer events not implemented");
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU command-buffer events not implemented");
}

//===----------------------------------------------------------------------===//
// Buffer Commands
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_aql_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU command-buffer fill replay not implemented");
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU command-buffer update replay not implemented");
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU command-buffer copy replay not implemented");
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU collectives not implemented");
}

static iree_status_t iree_hal_amdgpu_aql_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU command-buffer dispatch replay not implemented");
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_amdgpu_aql_command_buffer_vtable = {
        .destroy = iree_hal_amdgpu_aql_command_buffer_destroy,
        .begin = iree_hal_amdgpu_aql_command_buffer_begin,
        .end = iree_hal_amdgpu_aql_command_buffer_end,
        .begin_debug_group =
            iree_hal_amdgpu_aql_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_amdgpu_aql_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_amdgpu_aql_command_buffer_execution_barrier,
        .signal_event = iree_hal_amdgpu_aql_command_buffer_signal_event,
        .reset_event = iree_hal_amdgpu_aql_command_buffer_reset_event,
        .wait_events = iree_hal_amdgpu_aql_command_buffer_wait_events,
        .advise_buffer = iree_hal_amdgpu_aql_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_amdgpu_aql_command_buffer_fill_buffer,
        .update_buffer = iree_hal_amdgpu_aql_command_buffer_update_buffer,
        .copy_buffer = iree_hal_amdgpu_aql_command_buffer_copy_buffer,
        .collective = iree_hal_amdgpu_aql_command_buffer_collective,
        .dispatch = iree_hal_amdgpu_aql_command_buffer_dispatch,
};
