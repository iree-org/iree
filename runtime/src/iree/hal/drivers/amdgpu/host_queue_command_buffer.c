// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"

#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/aql_program_validation.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_block.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_replay.h"
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
