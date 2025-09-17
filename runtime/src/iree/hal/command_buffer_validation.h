// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_COMMAND_BUFFER_VALIDATION_H_
#define IREE_HAL_COMMAND_BUFFER_VALIDATION_H_

#include "iree/base/api.h"
#include "iree/hal/command_buffer.h"

// Requirements for a buffer resource used within a command buffer.
// Buffers bound to must have all bits set from the included bitfields and
// support the given min/max byte offsets as in-range.
typedef struct iree_hal_buffer_binding_requirements_t {
  iree_hal_buffer_compatibility_t required_compatibility;
  iree_hal_buffer_usage_t usage;
  iree_hal_memory_access_t access;
  iree_hal_memory_type_t type;
  // Maximum offset in the binding referenced by any command.
  iree_device_size_t max_byte_offset;
  // Minimum required alignment by at least one command.
  iree_device_size_t min_byte_alignment;
} iree_hal_buffer_binding_requirements_t;

// Storage for command buffer validation state.
// Designed to be embedded in concrete implementations that want validation.
typedef struct iree_hal_command_buffer_validation_state_t {
  // Allocator from the device the command buffer is targeting.
  // Used to verify buffer compatibility.
  iree_hal_allocator_t* device_allocator;
  // 1 when begin has been called.
  int32_t has_began : 1;
  // 1 when end has been called.
  int32_t has_ended : 1;
  // Debug group depth for tracking proper begin/end pairing.
  int32_t debug_group_depth : 30;
  // TODO(benvanik): current pipeline layout/descriptor set layout info.
  // TODO(benvanik): valid push constant bit ranges.
  // Requirements for each binding table entry.
  // Unused slots in the binding table will have IREE_HAL_BUFFER_USAGE_NONE and
  // are ignored if set when executed.
  iree_hal_buffer_binding_requirements_t binding_requirements[0];
} iree_hal_command_buffer_validation_state_t;

void iree_hal_command_buffer_initialize_validation(
    iree_hal_allocator_t* device_allocator,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* out_validation_state);

iree_status_t iree_hal_command_buffer_begin_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state);

iree_status_t iree_hal_command_buffer_end_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state);

iree_status_t iree_hal_command_buffer_begin_debug_group_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_string_view_t label, iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location);

iree_status_t iree_hal_command_buffer_end_debug_group_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state);

iree_status_t iree_hal_command_buffer_execution_barrier_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers);

iree_status_t iree_hal_command_buffer_signal_event_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_event_t* event, iree_hal_execution_stage_t source_stage_mask);

iree_status_t iree_hal_command_buffer_reset_event_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_event_t* event, iree_hal_execution_stage_t source_stage_mask);

iree_status_t iree_hal_command_buffer_wait_events_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers);

iree_status_t iree_hal_command_buffer_advise_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1);

iree_status_t iree_hal_command_buffer_fill_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags);

iree_status_t iree_hal_command_buffer_update_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_ref_t target_ref, iree_hal_update_flags_t flags);

iree_status_t iree_hal_command_buffer_copy_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags);

iree_status_t iree_hal_command_buffer_collective_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_channel_t* channel, iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_ref_t send_ref, iree_hal_buffer_ref_t recv_ref,
    iree_device_size_t element_count);

iree_status_t iree_hal_command_buffer_dispatch_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags);

iree_status_t iree_hal_command_buffer_submission_validation(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_command_buffer_validation_state_t* validation_state);

iree_status_t iree_hal_command_buffer_binding_table_validation(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_binding_table_t binding_table);

#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_H_
