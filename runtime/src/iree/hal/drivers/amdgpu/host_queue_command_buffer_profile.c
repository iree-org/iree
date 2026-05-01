// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_profile.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/profile_traces.h"

static bool iree_hal_amdgpu_host_queue_profiles_command_buffer_dispatches(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return queue->profiling.dispatch_profiling_enabled;
}

static bool
iree_hal_amdgpu_host_queue_should_profile_all_command_buffer_dispatches(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id) {
  if (command_buffer_id == 0) return false;
  if (!queue->profiling.hsa_queue_timestamps_enabled) return false;

  iree_hal_amdgpu_logical_device_t* logical_device =
      (iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  const iree_hal_profile_capture_filter_t* filter =
      &logical_device->profiling.options.capture_filter;
  if (iree_any_bit_set(
          filter->flags,
          IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_INDEX |
              IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_EXPORT_PATTERN)) {
    return false;
  }

  const uint32_t physical_device_ordinal = queue->device_ordinal <= UINT32_MAX
                                               ? (uint32_t)queue->device_ordinal
                                               : UINT32_MAX;
  const uint32_t queue_ordinal = iree_async_axis_queue_index(queue->axis);
  return iree_hal_profile_capture_filter_matches_location(
      filter, command_buffer_id, /*command_index=*/0, physical_device_ordinal,
      queue_ordinal);
}

static bool
iree_hal_amdgpu_host_queue_should_profile_command_buffer_dispatch_summary(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id,
    const iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* summary) {
  if (command_buffer_id == 0) return false;
  if (!queue->profiling.hsa_queue_timestamps_enabled) return false;
  const uint32_t physical_device_ordinal = queue->device_ordinal <= UINT32_MAX
                                               ? (uint32_t)queue->device_ordinal
                                               : UINT32_MAX;
  const uint32_t queue_ordinal = iree_async_axis_queue_index(queue->axis);
  iree_hal_amdgpu_logical_device_t* logical_device =
      (iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  return iree_hal_amdgpu_logical_device_should_profile_dispatch(
      logical_device, summary->metadata.executable_id,
      summary->metadata.export_ordinal, command_buffer_id,
      summary->metadata.command_index, physical_device_ordinal, queue_ordinal);
}

iree_status_t
iree_hal_amdgpu_host_queue_select_command_buffer_profile_dispatches(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_arena_allocator_t* scratch_arena,
    iree_hal_amdgpu_aql_block_processor_profile_dispatch_list_t*
        out_dispatches) {
  *out_dispatches =
      (iree_hal_amdgpu_aql_block_processor_profile_dispatch_list_t){0};
  const uint64_t command_buffer_id =
      iree_hal_amdgpu_aql_command_buffer_profile_id(command_buffer);
  if (command_buffer_id == 0) return iree_ok_status();
  if (!queue->profiling.hsa_queue_timestamps_enabled) return iree_ok_status();
  if (!iree_hal_amdgpu_host_queue_profiles_command_buffer_dispatches(queue)) {
    return iree_ok_status();
  }
  const bool profile_all_dispatches =
      iree_hal_amdgpu_host_queue_should_profile_all_command_buffer_dispatches(
          queue, command_buffer_id);

  uint32_t summary_count = 0;
  const iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* summary =
      iree_hal_amdgpu_aql_command_buffer_dispatch_summaries(
          command_buffer, block, &summary_count);
  if (IREE_UNLIKELY(summary_count != block->dispatch_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "retained dispatch summary count mismatch: expected %u but got %u",
        block->dispatch_count, summary_count);
  }
  if (summary_count == 0) return iree_ok_status();

  iree_host_size_t dispatch_storage_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &dispatch_storage_size,
      IREE_STRUCT_FIELD(summary_count,
                        iree_hal_amdgpu_aql_block_processor_profile_dispatch_t,
                        NULL)));
  iree_hal_amdgpu_aql_block_processor_profile_dispatch_t* dispatches = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(scratch_arena, dispatch_storage_size,
                                           (void**)&dispatches));

  uint32_t selected_count = 0;
  for (uint32_t summary_ordinal = 0; summary_ordinal < summary_count;
       ++summary_ordinal) {
    if (IREE_UNLIKELY(!summary)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "retained dispatch summary list ended after %u of %u entries",
          summary_ordinal, summary_count);
    }
    if (profile_all_dispatches ||
        iree_hal_amdgpu_host_queue_should_profile_command_buffer_dispatch_summary(
            queue, command_buffer_id, summary)) {
      dispatches[selected_count++].summary = summary;
    }
    summary = summary->next;
  }
  out_dispatches->values = selected_count ? dispatches : NULL;
  out_dispatches->count = selected_count;
  return iree_ok_status();
}

static bool iree_hal_amdgpu_command_buffer_dispatch_summary_uses_indirect(
    const iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* summary) {
  return iree_any_bit_set(
      summary->metadata.dispatch_flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS);
}

static void
iree_hal_amdgpu_host_queue_initialize_command_buffer_dispatch_summary_event(
    iree_hal_amdgpu_profile_dispatch_event_t* event, uint64_t command_buffer_id,
    const iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* summary) {
  const uint64_t event_id = event->event_id;
  memset(event, 0, sizeof(*event));
  event->record_length = sizeof(*event);
  event->flags = IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
  event->event_id = event_id;
  event->command_buffer_id = command_buffer_id;
  event->executable_id = summary->metadata.executable_id;
  if (iree_hal_amdgpu_command_buffer_dispatch_summary_uses_indirect(summary)) {
    event->flags |=
        IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS;
  }
  event->command_index = summary->metadata.command_index;
  event->export_ordinal = summary->metadata.export_ordinal;
  for (iree_host_size_t dimension_ordinal = 0;
       dimension_ordinal < IREE_ARRAYSIZE(event->workgroup_size);
       ++dimension_ordinal) {
    event->workgroup_size[dimension_ordinal] =
        summary->workgroup.size[dimension_ordinal];
    if (!iree_hal_amdgpu_command_buffer_dispatch_summary_uses_indirect(
            summary) &&
        summary->workgroup.size[dimension_ordinal] != 0) {
      event->workgroup_count[dimension_ordinal] =
          summary->workgroup.count[dimension_ordinal];
    }
  }
}

void iree_hal_amdgpu_host_queue_record_command_buffer_profile_dispatch_source(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id,
    const iree_hal_amdgpu_aql_block_processor_profile_dispatch_t* dispatch,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events,
    iree_hal_amdgpu_profile_dispatch_harvest_source_t* profile_harvest_sources,
    uint32_t* inout_profile_event_index) {
  const uint32_t profile_event_index = *inout_profile_event_index;
  const uint64_t profile_event_position =
      profile_events.first_event_position + profile_event_index;
  iree_hal_amdgpu_profile_dispatch_event_t* event =
      iree_hal_amdgpu_host_queue_profile_dispatch_event_at(
          queue, profile_event_position);
  iree_hal_amdgpu_host_queue_initialize_command_buffer_dispatch_summary_event(
      event, command_buffer_id, dispatch->summary);
  profile_harvest_sources[profile_event_index].completion_signal =
      iree_hal_amdgpu_host_queue_profiling_completion_signal_ptr(
          queue, profile_event_position);
  profile_harvest_sources[profile_event_index].ticks =
      iree_hal_amdgpu_profile_dispatch_event_ticks(event);
  *inout_profile_event_index = profile_event_index + 1;
}

iree_status_t
iree_hal_amdgpu_host_queue_prepare_command_buffer_profile_trace_code_objects(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_aql_block_processor_profile_dispatch_list_t dispatches,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events) {
  if (profile_events.event_count == 0 || !queue->profiling.traces.session) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(dispatches.count != profile_events.event_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "profile command-buffer dispatch selection count %u does not match "
        "reserved event count %u",
        dispatches.count, profile_events.event_count);
  }

  iree_status_t status = iree_ok_status();
  for (uint32_t dispatch_ordinal = 0;
       dispatch_ordinal < dispatches.count && iree_status_is_ok(status);
       ++dispatch_ordinal) {
    const iree_hal_amdgpu_aql_block_processor_profile_dispatch_t* dispatch =
        &dispatches.values[dispatch_ordinal];
    if (IREE_UNLIKELY(!dispatch->summary)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "profile command-buffer dispatch selection %u has no summary",
          dispatch_ordinal);
      break;
    }
    const uint64_t event_position =
        profile_events.first_event_position + dispatch_ordinal;
    status = iree_hal_amdgpu_host_queue_prepare_profile_trace_code_object(
        queue, event_position, dispatch->summary->metadata.executable_id);
  }
  return status;
}
