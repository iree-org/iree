// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_profile.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/profile_traces.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

void iree_hal_amdgpu_host_queue_commit_command_buffer_profile_start(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot =
      &queue->pm4_ib_slots[packet_id & queue->aql_ring.mask];
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(pm4_ib_slot, &builder);
  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_copy_timestamp_to_memory(
          &builder, &queue_device_event->start_tick);
  IREE_ASSERT(did_emit, "PM4 start timestamp must fit profiling IB slot");
  (void)did_emit;
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_pm4_ib(
      &packet->pm4_ib, pm4_ib_slot,
      iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), packet_control,
      iree_hsa_signal_null(), &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}

void iree_hal_amdgpu_host_queue_commit_command_buffer_profile_end(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal,
    iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot =
      &queue->pm4_ib_slots[packet_id & queue->aql_ring.mask];
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(pm4_ib_slot, &builder);
  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_release_mem_timestamp_to_memory(
          &builder, &queue_device_event->end_tick);
  IREE_ASSERT(did_emit, "PM4 end timestamp must fit profiling IB slot");
  (void)did_emit;
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_pm4_ib(
      &packet->pm4_ib, pm4_ib_slot,
      iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), packet_control,
      completion_signal, &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}

void iree_hal_amdgpu_host_queue_commit_command_buffer_profile_timestamp_range(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal,
    iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot =
      &queue->pm4_ib_slots[packet_id & queue->aql_ring.mask];
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(pm4_ib_slot, &builder);
  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_range_to_memory(
          &builder, &queue_device_event->start_tick,
          &queue_device_event->end_tick);
  IREE_ASSERT(did_emit, "PM4 timestamp range must fit profiling IB slot");
  (void)did_emit;
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_pm4_ib(
      &packet->pm4_ib, pm4_ib_slot,
      iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder), packet_control,
      completion_signal, &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}

bool iree_hal_amdgpu_host_queue_should_profile_command_buffer_dispatch(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  if (!queue->profiling.hsa_queue_timestamps_enabled) return false;
  const uint32_t physical_device_ordinal = queue->device_ordinal <= UINT32_MAX
                                               ? (uint32_t)queue->device_ordinal
                                               : UINT32_MAX;
  const uint32_t queue_ordinal = iree_async_axis_queue_index(queue->axis);
  iree_hal_amdgpu_logical_device_t* logical_device =
      (iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  return iree_hal_amdgpu_logical_device_should_profile_dispatch(
      logical_device, dispatch_command->executable_id,
      dispatch_command->export_ordinal, command_buffer_id,
      dispatch_command->header.command_index, physical_device_ordinal,
      queue_ordinal);
}

uint32_t
iree_hal_amdgpu_host_queue_count_command_buffer_profile_dispatch_events(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id,
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  if (!queue->profiling.hsa_queue_timestamps_enabled) return 0;

  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(block);
  uint32_t dispatch_event_count = 0;
  for (uint16_t command_ordinal = 0; command_ordinal < block->command_count;
       ++command_ordinal) {
    switch (command->opcode) {
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH: {
        const iree_hal_amdgpu_command_buffer_dispatch_command_t*
            dispatch_command =
                (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)
                    command;
        if (iree_hal_amdgpu_host_queue_should_profile_command_buffer_dispatch(
                queue, command_buffer_id, dispatch_command)) {
          ++dispatch_event_count;
        }
        command = iree_hal_amdgpu_command_buffer_command_next_const(command);
        break;
      }
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
        return dispatch_event_count;
      default:
        command = iree_hal_amdgpu_command_buffer_command_next_const(command);
        break;
    }
  }
  return dispatch_event_count;
}

static bool iree_hal_amdgpu_command_buffer_dispatch_uses_indirect_parameters(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return iree_any_bit_set(
      dispatch_command->dispatch_flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS);
}

static void iree_hal_amdgpu_host_queue_initialize_command_buffer_dispatch_event(
    iree_hal_amdgpu_profile_dispatch_event_t* event, uint64_t command_buffer_id,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  const uint64_t event_id = event->event_id;
  memset(event, 0, sizeof(*event));
  event->record_length = sizeof(*event);
  event->flags = IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
  event->event_id = event_id;
  event->command_buffer_id = command_buffer_id;
  event->executable_id = dispatch_command->executable_id;
  if (iree_hal_amdgpu_command_buffer_dispatch_uses_indirect_parameters(
          dispatch_command)) {
    event->flags |=
        IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS;
  }
  event->command_index = dispatch_command->header.command_index;
  event->export_ordinal = dispatch_command->export_ordinal;
  for (iree_host_size_t dimension_ordinal = 0;
       dimension_ordinal < IREE_ARRAYSIZE(event->workgroup_size);
       ++dimension_ordinal) {
    event->workgroup_size[dimension_ordinal] =
        dispatch_command->workgroup_size[dimension_ordinal];
    if (!iree_hal_amdgpu_command_buffer_dispatch_uses_indirect_parameters(
            dispatch_command) &&
        dispatch_command->workgroup_size[dimension_ordinal] != 0) {
      event->workgroup_count[dimension_ordinal] =
          dispatch_command->grid_size[dimension_ordinal] /
          dispatch_command->workgroup_size[dimension_ordinal];
    }
  }
}

void iree_hal_amdgpu_host_queue_record_command_buffer_profile_dispatch_source(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events,
    iree_hal_amdgpu_profile_dispatch_harvest_source_t* profile_harvest_sources,
    bool profile_dispatch_packet, uint32_t* inout_profile_event_index) {
  if (!profile_dispatch_packet) return;
  const uint32_t profile_event_index = *inout_profile_event_index;
  const uint64_t profile_event_position =
      profile_events.first_event_position + profile_event_index;
  iree_hal_amdgpu_profile_dispatch_event_t* event =
      iree_hal_amdgpu_host_queue_profile_dispatch_event_at(
          queue, profile_event_position);
  iree_hal_amdgpu_host_queue_initialize_command_buffer_dispatch_event(
      event, command_buffer_id, dispatch_command);
  profile_harvest_sources[profile_event_index].completion_signal =
      iree_hal_amdgpu_host_queue_profiling_completion_signal_ptr(
          queue, profile_event_position);
  profile_harvest_sources[profile_event_index].event = event;
  *inout_profile_event_index = profile_event_index + 1;
}

iree_status_t
iree_hal_amdgpu_host_queue_prepare_command_buffer_profile_trace_code_objects(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events) {
  if (profile_events.event_count == 0 || !queue->profiling.trace_session) {
    return iree_ok_status();
  }

  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(block);
  uint32_t profile_event_index = 0;
  bool reached_terminator = false;
  iree_status_t status = iree_ok_status();
  for (uint16_t command_ordinal = 0;
       command_ordinal < block->command_count && iree_status_is_ok(status) &&
       !reached_terminator;
       ++command_ordinal) {
    switch (command->opcode) {
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH: {
        const iree_hal_amdgpu_command_buffer_dispatch_command_t*
            dispatch_command =
                (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)
                    command;
        if (iree_hal_amdgpu_host_queue_should_profile_command_buffer_dispatch(
                queue, command_buffer_id, dispatch_command)) {
          const uint64_t event_position =
              profile_events.first_event_position + profile_event_index++;
          status = iree_hal_amdgpu_host_queue_prepare_profile_trace_code_object(
              queue, event_position, dispatch_command->executable_id);
        }
        command = iree_hal_amdgpu_command_buffer_command_next_const(command);
        break;
      }
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
        reached_terminator = true;
        break;
      default:
        command = iree_hal_amdgpu_command_buffer_command_next_const(command);
        break;
    }
  }
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(profile_event_index != profile_events.event_count)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "profile command-buffer dispatch event count changed during trace "
        "preparation");
  }
  return status;
}
