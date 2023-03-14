// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_COMMAND_BUFFER_VALIDATION_H_
#define IREE_HAL_COMMAND_BUFFER_VALIDATION_H_

#include "iree/base/api.h"
#include "iree/hal/command_buffer.h"

void iree_hal_command_buffer_initialize_validation(
    iree_hal_device_t* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* out_validation_state);

iree_status_t iree_hal_command_buffer_begin_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state);

iree_status_t iree_hal_command_buffer_end_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state);

void iree_hal_command_buffer_begin_debug_group_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_string_view_t label, iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location);

void iree_hal_command_buffer_end_debug_group_validation(
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

iree_status_t iree_hal_command_buffer_discard_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_t* buffer);

iree_status_t iree_hal_command_buffer_fill_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length);

iree_status_t iree_hal_command_buffer_update_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length);

iree_status_t iree_hal_command_buffer_copy_buffer_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length);

iree_status_t iree_hal_command_buffer_collective_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_channel_t* channel, iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count);

iree_status_t iree_hal_command_buffer_push_constants_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length);

iree_status_t iree_hal_command_buffer_push_descriptor_set_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings);

iree_status_t iree_hal_command_buffer_dispatch_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z);

iree_status_t iree_hal_command_buffer_dispatch_indirect_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer, iree_device_size_t workgroups_offset);

iree_status_t iree_hal_command_buffer_execute_commands_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_buffer_validation_state_t* validation_state,
    iree_hal_command_buffer_t* commands,
    iree_hal_buffer_binding_table_t binding_table);

#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_H_
