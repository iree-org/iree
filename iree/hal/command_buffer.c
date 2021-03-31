// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/command_buffer.h"

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"

#define _VTABLE_DISPATCH(command_buffer, method_name) \
  IREE_HAL_VTABLE_DISPATCH(command_buffer, iree_hal_command_buffer, method_name)

IREE_HAL_API_RETAIN_RELEASE(command_buffer);

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, create_command_buffer)(
          device, mode, command_categories, queue_affinity, out_command_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_hal_command_category_t IREE_API_CALL
iree_hal_command_buffer_allowed_categories(
    const iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  return _VTABLE_DISPATCH(command_buffer, allowed_categories)(command_buffer);
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_begin(iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(command_buffer, begin)(command_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_end(iree_hal_command_buffer_t* command_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, end)(command_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, execution_barrier)(
      command_buffer, source_stage_mask, target_stage_mask, flags,
      memory_barrier_count, memory_barriers, buffer_barrier_count,
      buffer_barriers);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_signal_event(
    iree_hal_command_buffer_t* command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, signal_event)(
      command_buffer, event, source_stage_mask);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_reset_event(
    iree_hal_command_buffer_t* command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, reset_event)(
      command_buffer, event, source_stage_mask);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_wait_events(
    iree_hal_command_buffer_t* command_buffer, iree_host_size_t event_count,
    const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(!event_count || events);
  IREE_ASSERT_ARGUMENT(!memory_barrier_count || memory_barriers);
  IREE_ASSERT_ARGUMENT(!buffer_barrier_count || buffer_barriers);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, wait_events)(
      command_buffer, event_count, events, source_stage_mask, target_stage_mask,
      memory_barrier_count, memory_barriers, buffer_barrier_count,
      buffer_barriers);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(command_buffer, discard_buffer)(command_buffer, buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, fill_buffer)(
      command_buffer, target_buffer, target_offset, length, pattern,
      pattern_length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_update_buffer(iree_hal_command_buffer_t* command_buffer,
                                      const void* source_buffer,
                                      iree_host_size_t source_offset,
                                      iree_hal_buffer_t* target_buffer,
                                      iree_device_size_t target_offset,
                                      iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, update_buffer)(
      command_buffer, source_buffer, source_offset, target_buffer,
      target_offset, length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, copy_buffer)(
      command_buffer, source_buffer, source_offset, target_buffer,
      target_offset, length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_push_constants(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable_layout);
  IREE_ASSERT_ARGUMENT(values);
  if (IREE_UNLIKELY(values_length == 0)) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, push_constants)(
      command_buffer, executable_layout, offset, values, values_length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable_layout);
  IREE_ASSERT_ARGUMENT(!binding_count || bindings);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, push_descriptor_set)(
      command_buffer, executable_layout, set, binding_count, bindings);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable_layout);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(!dynamic_offset_count || dynamic_offsets);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, bind_descriptor_set)(
      command_buffer, executable_layout, set, descriptor_set,
      dynamic_offset_count, dynamic_offsets);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_dispatch(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, dispatch)(
      command_buffer, executable, entry_point, workgroup_x, workgroup_y,
      workgroup_z);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(workgroups_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(command_buffer, dispatch_indirect)(
      command_buffer, executable, entry_point, workgroups_buffer,
      workgroups_offset);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
