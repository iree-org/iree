// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/aql_program_validation.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_block.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_replay.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/pm4_command_buffer.h"
#include "iree/hal/utils/resource_set.h"

iree_status_t iree_hal_amdgpu_host_queue_validate_execute_flags(
    iree_hal_execute_flags_t flags) {
  const iree_hal_execute_flags_t supported_flags =
      IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME;
  if (IREE_UNLIKELY(iree_any_bit_set(flags, ~supported_flags))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported execute flags: 0x%" PRIx64, flags);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_create_binding_table_resource_set(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t execute_flags,
    iree_hal_resource_set_t** out_resource_set) {
  *out_resource_set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_execute_flags(execute_flags));
  if (!command_buffer || command_buffer->binding_count == 0) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(binding_table.count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indirect command buffer requires at least %u "
                            "bindings but no binding table was provided",
                            command_buffer->binding_count);
  }
  if (IREE_UNLIKELY(binding_table.count < command_buffer->binding_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "indirect command buffer requires at least %u bindings but only "
        "%" PRIhsz " were provided",
        command_buffer->binding_count, binding_table.count);
  }
  if (IREE_UNLIKELY(!binding_table.bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indirect command buffer binding table storage is "
                            "NULL for %" PRIhsz " bindings",
                            binding_table.count);
  }
  if (iree_any_bit_set(execute_flags,
                       IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME)) {
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, command_buffer->binding_count);
  iree_hal_resource_set_t* resource_set = NULL;
  iree_status_t status =
      iree_hal_resource_set_allocate(queue->block_pool, &resource_set);
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert_strided(
        resource_set, command_buffer->binding_count, binding_table.bindings,
        offsetof(iree_hal_buffer_binding_t, buffer),
        sizeof(iree_hal_buffer_binding_t));
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_set_freeze(resource_set);
    *out_resource_set = resource_set;
  } else {
    iree_hal_resource_set_free(resource_set);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_host_queue_retire_pm4_publication_reference(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    iree_status_t status) {
  (void)entry;
  iree_hal_amdgpu_pm4_command_buffer_retire_publication_reference(
      (iree_hal_command_buffer_t*)user_data, status);
}

static iree_hal_amdgpu_reclaim_action_t
iree_hal_amdgpu_host_queue_make_pm4_publication_retire_action(
    iree_hal_command_buffer_t* command_buffer,
    hsa_signal_t publication_signal) {
  if (iree_hsa_signal_is_null(publication_signal)) {
    return (iree_hal_amdgpu_reclaim_action_t){0};
  }
  iree_hal_amdgpu_reclaim_action_t action = {
      .fn = iree_hal_amdgpu_host_queue_retire_pm4_publication_reference,
      .user_data = command_buffer,
  };
  return action;
}

static void iree_hal_amdgpu_host_queue_cancel_pm4_publication_reference(
    iree_hal_command_buffer_t* command_buffer,
    hsa_signal_t publication_signal) {
  if (iree_hsa_signal_is_null(publication_signal)) return;
  iree_hal_amdgpu_pm4_command_buffer_cancel_publication_reference(
      command_buffer);
}

static iree_status_t
iree_hal_amdgpu_host_queue_verify_pm4_command_buffer_profiling_supported(
    const iree_hal_amdgpu_host_queue_t* queue) {
  if (!queue->profiling.dispatch_profiling_enabled) {
    return iree_ok_status();
  }
  if (queue->profiling.counters.session) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU PM4 command-buffer profiling currently supports "
        "dispatch-timestamp events but not dispatch-attributed counter "
        "samples");
  }
  if (queue->profiling.traces.session) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU PM4 command-buffer profiling currently supports "
        "dispatch-timestamp events but not dispatch-attributed executable "
        "traces");
  }
  return iree_ok_status();
}

static iree_hal_amdgpu_host_queue_profile_event_info_t
iree_hal_amdgpu_host_queue_pm4_command_buffer_profile_event_info(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_pm4_program_t* program) {
  return (iree_hal_amdgpu_host_queue_profile_event_info_t){
      .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE,
      .command_buffer_id =
          iree_hal_amdgpu_pm4_command_buffer_profile_id(command_buffer),
      .payload_length = (uint64_t)program->dword_count * sizeof(uint32_t),
      .operation_count =
          iree_hal_amdgpu_pm4_command_buffer_operation_count(command_buffer),
  };
}

typedef struct iree_hal_amdgpu_host_queue_pm4_profile_dispatch_selection_t {
  const iree_hal_profile_command_operation_record_t* operations;
  uint32_t operation_count;
  uint32_t* selected_ordinals;
  uint32_t selected_count;
} iree_hal_amdgpu_host_queue_pm4_profile_dispatch_selection_t;

static bool
iree_hal_amdgpu_host_queue_should_profile_all_pm4_command_buffer_dispatches(
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
iree_hal_amdgpu_host_queue_should_profile_pm4_command_buffer_dispatch(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t command_buffer_id,
    const iree_hal_profile_command_operation_record_t* operation) {
  const uint32_t physical_device_ordinal = queue->device_ordinal <= UINT32_MAX
                                               ? (uint32_t)queue->device_ordinal
                                               : UINT32_MAX;
  const uint32_t queue_ordinal = iree_async_axis_queue_index(queue->axis);
  iree_hal_amdgpu_logical_device_t* logical_device =
      (iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  return iree_hal_amdgpu_logical_device_should_profile_dispatch(
      logical_device, operation->executable_id, operation->export_ordinal,
      command_buffer_id, operation->command_index, physical_device_ordinal,
      queue_ordinal);
}

static iree_status_t
iree_hal_amdgpu_host_queue_select_pm4_command_buffer_profile_dispatches(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    iree_arena_allocator_t* scratch_arena,
    iree_hal_amdgpu_host_queue_pm4_profile_dispatch_selection_t*
        out_selection) {
  *out_selection =
      (iree_hal_amdgpu_host_queue_pm4_profile_dispatch_selection_t){0};
  const uint64_t command_buffer_id =
      iree_hal_amdgpu_pm4_command_buffer_profile_id(command_buffer);
  if (command_buffer_id == 0) return iree_ok_status();
  if (!queue->profiling.hsa_queue_timestamps_enabled) return iree_ok_status();
  if (!queue->profiling.dispatch_profiling_enabled) return iree_ok_status();

  uint32_t operation_count = 0;
  const iree_hal_profile_command_operation_record_t* operations =
      iree_hal_amdgpu_pm4_command_buffer_profile_operations(command_buffer,
                                                            &operation_count);
  if (operation_count == 0) return iree_ok_status();
  if (IREE_UNLIKELY(!operations)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 command buffer has no retained profile operation records");
  }

  iree_host_size_t selected_storage_size = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(0, &selected_storage_size,
                         IREE_STRUCT_FIELD(operation_count, uint32_t, NULL)));
  uint32_t* selected_ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(scratch_arena, selected_storage_size,
                                           (void**)&selected_ordinals));

  const bool profile_all_dispatches =
      iree_hal_amdgpu_host_queue_should_profile_all_pm4_command_buffer_dispatches(
          queue, command_buffer_id);
  uint32_t selected_count = 0;
  for (uint32_t operation_ordinal = 0; operation_ordinal < operation_count;
       ++operation_ordinal) {
    const iree_hal_profile_command_operation_record_t* operation =
        &operations[operation_ordinal];
    if (operation->type != IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH) {
      continue;
    }
    if (profile_all_dispatches ||
        iree_hal_amdgpu_host_queue_should_profile_pm4_command_buffer_dispatch(
            queue, command_buffer_id, operation)) {
      selected_ordinals[selected_count++] = operation_ordinal;
    }
  }

  out_selection->operations = operations;
  out_selection->operation_count = operation_count;
  out_selection->selected_ordinals = selected_count ? selected_ordinals : NULL;
  out_selection->selected_count = selected_count;
  return iree_ok_status();
}

static void iree_hal_amdgpu_host_queue_initialize_pm4_dispatch_event(
    iree_hal_amdgpu_profile_dispatch_event_t* event,
    const iree_hal_profile_command_operation_record_t* operation) {
  const uint64_t event_id = event->event_id;
  memset(event, 0, sizeof(*event));
  event->record_length = sizeof(*event);
  event->flags = IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
  event->event_id = event_id;
  event->command_buffer_id = operation->command_buffer_id;
  event->executable_id = operation->executable_id;
  event->command_index = operation->command_index;
  event->export_ordinal = operation->export_ordinal;
  memcpy(event->workgroup_count, operation->workgroup_count,
         sizeof(event->workgroup_count));
  memcpy(event->workgroup_size, operation->workgroup_size,
         sizeof(event->workgroup_size));
}

static iree_status_t
iree_hal_amdgpu_host_queue_submit_profiled_pm4_command_buffer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t execute_flags,
    iree_hal_resource_set_t** inout_binding_resource_set,
    const iree_hal_amdgpu_host_queue_pm4_profile_dispatch_selection_t*
        selection,
    iree_arena_allocator_t* scratch_arena, bool* out_ready) {
  const iree_hal_amdgpu_pm4_command_buffer_profile_plan_t* profile_plan =
      iree_hal_amdgpu_pm4_command_buffer_profile_plan(command_buffer);
  if (IREE_UNLIKELY(!profile_plan->program.dwords ||
                    profile_plan->program.dword_count == 0 ||
                    !profile_plan->entries || profile_plan->entry_count == 0 ||
                    !profile_plan->target_base || !profile_plan->dummy_ticks)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 command buffer was not recorded with dispatch timestamp "
        "profiling materialized");
  }
  if (IREE_UNLIKELY(selection->operation_count !=
                    profile_plan->dispatch_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "PM4 profile operation count %u does not match profile dispatch "
        "count %u",
        selection->operation_count, profile_plan->dispatch_count);
  }
  if (IREE_UNLIKELY(profile_plan->binding_count <
                        command_buffer->binding_count ||
                    profile_plan->timestamp_binding_base !=
                        command_buffer->binding_count)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 profile binding layout is inconsistent");
  }
  if (command_buffer->binding_count == 0) {
    if (IREE_UNLIKELY(binding_table.count != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "static PM4 command buffer cannot execute with a binding table");
    }
  } else if (!*inout_binding_resource_set) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_host_queue_create_binding_table_resource_set(
            queue, command_buffer, binding_table, execute_flags,
            inout_binding_resource_set));
  }

  uint64_t* binding_ptrs = NULL;
  iree_host_size_t binding_ptr_bytes = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      0, &binding_ptr_bytes,
      IREE_STRUCT_FIELD(profile_plan->binding_count, uint64_t, NULL));
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate(scratch_arena, binding_ptr_bytes,
                                 (void**)&binding_ptrs);
  }
  if (iree_status_is_ok(status)) {
    memset(binding_ptrs, 0, binding_ptr_bytes);
    if (command_buffer->binding_count != 0) {
      status = iree_hal_amdgpu_host_queue_resolve_command_buffer_binding_ptrs(
          command_buffer, binding_table, binding_ptrs);
    }
  }
  if (iree_status_is_ok(status)) {
    const uint64_t dummy_start_tick =
        (uint64_t)(uintptr_t)&profile_plan->dummy_ticks->start_tick;
    const uint64_t dummy_end_tick =
        (uint64_t)(uintptr_t)&profile_plan->dummy_ticks->end_tick;
    for (uint32_t dispatch_ordinal = 0;
         dispatch_ordinal < profile_plan->dispatch_count; ++dispatch_ordinal) {
      const uint32_t timestamp_binding =
          profile_plan->timestamp_binding_base + 2u * dispatch_ordinal;
      binding_ptrs[timestamp_binding + 0] = dummy_start_tick;
      binding_ptrs[timestamp_binding + 1] = dummy_end_tick;
    }
  }

  iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_reserve_profile_dispatch_events(
        queue, selection->selected_count, &profile_events);
  }
  if (iree_status_is_ok(status)) {
    for (uint32_t selected_ordinal = 0;
         selected_ordinal < selection->selected_count; ++selected_ordinal) {
      const uint32_t operation_ordinal =
          selection->selected_ordinals[selected_ordinal];
      const iree_hal_profile_command_operation_record_t* operation =
          &selection->operations[operation_ordinal];
      iree_hal_amdgpu_profile_dispatch_event_t* event =
          iree_hal_amdgpu_host_queue_profile_dispatch_event_at(
              queue, profile_events.first_event_position + selected_ordinal);
      iree_hal_amdgpu_host_queue_initialize_pm4_dispatch_event(event,
                                                               operation);
      iree_hal_amdgpu_timestamp_range_t* event_ticks =
          iree_hal_amdgpu_profile_dispatch_event_ticks(event);
      const uint32_t timestamp_binding =
          profile_plan->timestamp_binding_base + 2u * operation_ordinal;
      binding_ptrs[timestamp_binding + 0] =
          (uint64_t)(uintptr_t)&event_ticks->start_tick;
      binding_ptrs[timestamp_binding + 1] =
          (uint64_t)(uintptr_t)&event_ticks->end_tick;
    }
  }

  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_pm4_command_buffer_profile_event_info(
          command_buffer, &profile_plan->program);
  const iree_hal_amdgpu_host_queue_profile_event_info_t*
      profile_event_info_ptr = (queue->profiling.queue_events_enabled ||
                                queue->profiling.queue_device_events_enabled)
                                   ? &profile_event_info
                                   : NULL;
  iree_hal_resource_t* command_buffer_resource =
      (iree_hal_resource_t*)command_buffer;
  hsa_signal_t publication_signal = iree_hsa_signal_null();
  bool submit_called = false;
  if (iree_status_is_ok(status)) {
    publication_signal =
        iree_hal_amdgpu_pm4_command_buffer_acquire_publication_reference(
            command_buffer);
    const iree_hal_amdgpu_reclaim_action_t publication_retire_action =
        iree_hal_amdgpu_host_queue_make_pm4_publication_retire_action(
            command_buffer, publication_signal);
    uint64_t submission_id = 0;
    submit_called = true;
    status = iree_hal_amdgpu_host_queue_submit_pm4_ib_with_binding_table_fixup(
        queue, resolution, signal_semaphore_list,
        &queue->transfer_context->kernels
             ->iree_hal_amdgpu_device_dispatch_patch_pm4_bindings,
        profile_plan->entries, profile_plan->entry_count,
        profile_plan->target_base, binding_ptrs, profile_plan->binding_count,
        profile_plan->program.dwords, profile_plan->program.dword_count,
        publication_signal, publication_retire_action, &command_buffer_resource,
        /*operation_resource_count=*/1, inout_binding_resource_set,
        profile_events, profile_event_info_ptr,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, out_ready,
        &submission_id);
    if (!iree_status_is_ok(status) || !*out_ready) {
      iree_hal_amdgpu_host_queue_cancel_pm4_publication_reference(
          command_buffer, publication_signal);
    } else {
      profile_event_info.submission_id = submission_id;
      iree_hal_amdgpu_host_queue_record_profile_queue_event(
          queue, resolution, signal_semaphore_list, &profile_event_info);
    }
  }
  if (!iree_status_is_ok(status)) {
    if (!submit_called) {
      iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                                profile_events);
    }
    iree_hal_resource_set_free(*inout_binding_resource_set);
    *inout_binding_resource_set = NULL;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_pm4_command_buffer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t execute_flags,
    iree_hal_resource_set_t** inout_binding_resource_set, bool* out_ready) {
  const iree_host_size_t command_buffer_device_ordinal =
      iree_hal_amdgpu_pm4_command_buffer_device_ordinal(command_buffer);
  if (IREE_UNLIKELY(command_buffer_device_ordinal != queue->device_ordinal)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 command buffer recorded for physical device %" PRIhsz
        " cannot execute on physical device %" PRIhsz,
        command_buffer_device_ordinal, queue->device_ordinal);
  }

  const iree_hal_amdgpu_pm4_program_t* program =
      iree_hal_amdgpu_pm4_command_buffer_program(command_buffer);
  if (IREE_UNLIKELY(!program->dwords || program->dword_count == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "PM4 command buffer has not been finalized");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_verify_pm4_command_buffer_profiling_supported(
          queue));

  if (queue->profiling.dispatch_profiling_enabled) {
    iree_arena_allocator_t scratch_arena;
    iree_arena_initialize(queue->block_pool, &scratch_arena);
    iree_hal_amdgpu_host_queue_pm4_profile_dispatch_selection_t selection = {0};
    iree_status_t status =
        iree_hal_amdgpu_host_queue_select_pm4_command_buffer_profile_dispatches(
            queue, command_buffer, &scratch_arena, &selection);
    if (iree_status_is_ok(status) && selection.selected_count != 0) {
      status = iree_hal_amdgpu_host_queue_submit_profiled_pm4_command_buffer(
          queue, resolution, signal_semaphore_list, command_buffer,
          binding_table, execute_flags, inout_binding_resource_set, &selection,
          &scratch_arena, out_ready);
    }
    iree_arena_deinitialize(&scratch_arena);
    if (!iree_status_is_ok(status) || selection.selected_count != 0) {
      return status;
    }
  }

  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_pm4_command_buffer_profile_event_info(
          command_buffer, program);
  const iree_hal_amdgpu_host_queue_profile_event_info_t*
      profile_event_info_ptr = (queue->profiling.queue_events_enabled ||
                                queue->profiling.queue_device_events_enabled)
                                   ? &profile_event_info
                                   : NULL;

  iree_hal_resource_t* command_buffer_resource =
      (iree_hal_resource_t*)command_buffer;
  if (command_buffer->binding_count == 0) {
    if (IREE_UNLIKELY(binding_table.count != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "static PM4 command buffer cannot execute with a binding table");
    }
    const hsa_signal_t publication_signal =
        iree_hal_amdgpu_pm4_command_buffer_acquire_publication_reference(
            command_buffer);
    const iree_hal_amdgpu_reclaim_action_t publication_retire_action =
        iree_hal_amdgpu_host_queue_make_pm4_publication_retire_action(
            command_buffer, publication_signal);
    uint64_t submission_id = 0;
    iree_status_t status = iree_hal_amdgpu_host_queue_submit_pm4_ib(
        queue, resolution, signal_semaphore_list, program->dwords,
        program->dword_count, publication_signal, publication_retire_action,
        &command_buffer_resource, /*operation_resource_count=*/1,
        profile_event_info_ptr,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, out_ready,
        &submission_id);
    if (!iree_status_is_ok(status) || !*out_ready) {
      iree_hal_amdgpu_host_queue_cancel_pm4_publication_reference(
          command_buffer, publication_signal);
    } else {
      profile_event_info.submission_id = submission_id;
      iree_hal_amdgpu_host_queue_record_profile_queue_event(
          queue, resolution, signal_semaphore_list, &profile_event_info);
    }
    return status;
  }

  if (!*inout_binding_resource_set) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_host_queue_create_binding_table_resource_set(
            queue, command_buffer, binding_table, execute_flags,
            inout_binding_resource_set));
  }

  const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t* fixup_plan =
      iree_hal_amdgpu_pm4_command_buffer_fixup_plan(command_buffer);
  if (IREE_UNLIKELY(fixup_plan->entry_count == 0 || !fixup_plan->entries ||
                    !fixup_plan->target_base)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "dynamic PM4 command buffer has no finalized binding fixup plan");
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(queue->block_pool, &scratch_arena);
  uint64_t* binding_ptrs = NULL;
  iree_host_size_t binding_ptr_bytes = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      0, &binding_ptr_bytes,
      IREE_STRUCT_FIELD(command_buffer->binding_count, uint64_t, NULL));
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate(&scratch_arena, binding_ptr_bytes,
                                 (void**)&binding_ptrs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_resolve_command_buffer_binding_ptrs(
        command_buffer, binding_table, binding_ptrs);
  }
  if (iree_status_is_ok(status)) {
    const hsa_signal_t publication_signal =
        iree_hal_amdgpu_pm4_command_buffer_acquire_publication_reference(
            command_buffer);
    const iree_hal_amdgpu_reclaim_action_t publication_retire_action =
        iree_hal_amdgpu_host_queue_make_pm4_publication_retire_action(
            command_buffer, publication_signal);
    uint64_t submission_id = 0;
    status = iree_hal_amdgpu_host_queue_submit_pm4_ib_with_binding_table_fixup(
        queue, resolution, signal_semaphore_list,
        &queue->transfer_context->kernels
             ->iree_hal_amdgpu_device_dispatch_patch_pm4_bindings,
        fixup_plan->entries, fixup_plan->entry_count, fixup_plan->target_base,
        binding_ptrs, command_buffer->binding_count, program->dwords,
        program->dword_count, publication_signal, publication_retire_action,
        &command_buffer_resource, /*operation_resource_count=*/1,
        inout_binding_resource_set,
        (iree_hal_amdgpu_profile_dispatch_event_reservation_t){0},
        profile_event_info_ptr,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, out_ready,
        &submission_id);
    if (!iree_status_is_ok(status) || !*out_ready) {
      iree_hal_amdgpu_host_queue_cancel_pm4_publication_reference(
          command_buffer, publication_signal);
    } else {
      profile_event_info.submission_id = submission_id;
      iree_hal_amdgpu_host_queue_record_profile_queue_event(
          queue, resolution, signal_semaphore_list, &profile_event_info);
    }
  }
  iree_arena_deinitialize(&scratch_arena);
  if (!iree_status_is_ok(status)) {
    iree_hal_resource_set_free(*inout_binding_resource_set);
    *inout_binding_resource_set = NULL;
  }
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_submit_command_buffer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t execute_flags,
    iree_hal_resource_set_t** inout_binding_resource_set, bool* out_ready) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(resolution);
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;

  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_execute_flags(execute_flags));
  if (IREE_UNLIKELY(!command_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command buffer is required");
  }
  if (iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer)) {
    return iree_hal_amdgpu_host_queue_submit_pm4_command_buffer(
        queue, resolution, signal_semaphore_list, command_buffer, binding_table,
        execute_flags, inout_binding_resource_set, out_ready);
  }
  if (IREE_UNLIKELY(!iree_hal_amdgpu_aql_command_buffer_isa(command_buffer))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command buffer is not an AMDGPU AQL command "
                            "buffer");
  }
  const iree_host_size_t command_buffer_device_ordinal =
      iree_hal_amdgpu_aql_command_buffer_device_ordinal(command_buffer);
  if (IREE_UNLIKELY(command_buffer_device_ordinal != queue->device_ordinal)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command buffer recorded for physical device %" PRIhsz
        " cannot execute on physical device %" PRIhsz,
        command_buffer_device_ordinal, queue->device_ordinal);
  }

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  if (IREE_UNLIKELY(!program->first_block)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer has not been finalized");
  }

  const bool requires_replay =
      program->max_block_aql_packet_count == 0 || program->block_count != 1;
  if (requires_replay && program->max_block_aql_packet_count == 0) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_aql_program_validate_metadata_only(program));
  }
  if (requires_replay) {
    iree_status_t status =
        iree_hal_amdgpu_command_buffer_replay_start_under_lock(
            queue, resolution, signal_semaphore_list, command_buffer,
            binding_table, execute_flags, inout_binding_resource_set);
    if (iree_status_is_ok(status)) *out_ready = true;
    return status;
  }
  if (!*inout_binding_resource_set) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_host_queue_create_binding_table_resource_set(
            queue, command_buffer, binding_table, execute_flags,
            inout_binding_resource_set));
  }
  iree_hal_resource_t* command_buffer_resource =
      (iree_hal_resource_t*)command_buffer;
  bool ready = false;
  iree_status_t status = iree_hal_amdgpu_host_queue_submit_command_buffer_block(
      queue, resolution, signal_semaphore_list, command_buffer, binding_table,
      /*binding_ptrs=*/NULL, program->first_block, inout_binding_resource_set,
      (iree_hal_amdgpu_reclaim_action_t){0}, &command_buffer_resource,
      /*operation_resource_count=*/1,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, &ready);
  if (!iree_status_is_ok(status)) {
    iree_hal_resource_set_free(*inout_binding_resource_set);
    *inout_binding_resource_set = NULL;
  } else {
    *out_ready = ready;
  }
  return status;
}
