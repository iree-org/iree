// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/queue_emulation.h"

//===----------------------------------------------------------------------===//
// Emulated Queue Operations
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_device_queue_emulated_fill(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_ASSERT_ARGUMENT(pattern);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  // If we are starting execution immediately then we can reduce latency by
  // allowing inline command buffer execution.
  iree_hal_command_buffer_mode_t command_buffer_mode =
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT;
  if (wait_semaphore_list.count == 0) {
    command_buffer_mode |= IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION;
  }

  iree_hal_transfer_command_t command = {
      .type = IREE_HAL_TRANSFER_COMMAND_TYPE_FILL,
      .fill =
          {
              .target_buffer = target_buffer,
              .target_offset = target_offset,
              .length = length,
              .pattern = pattern,
              .pattern_length = pattern_length,
          },
  };

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_create_transfer_command_buffer(device, command_buffer_mode,
                                                  queue_affinity, 1, &command,
                                                  &command_buffer));

  iree_status_t status = iree_hal_device_queue_execute(
      device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE);

  iree_hal_command_buffer_release(command_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_emulated_update(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  // If we are starting execution immediately then we can reduce latency by
  // allowing inline command buffer execution.
  iree_hal_command_buffer_mode_t command_buffer_mode =
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT;
  if (wait_semaphore_list.count == 0) {
    command_buffer_mode |= IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION;
  }

  if (length > (iree_device_size_t)(IREE_HOST_SIZE_MAX - source_offset)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "queue update source range overflows host address space "
        "(source_offset=%" PRIhsz ", length=%" PRIdsz ")",
        source_offset, length);
  }
  if (length > IREE_DEVICE_SIZE_MAX - target_offset) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "queue update target range overflows device address space "
        "(target_offset=%" PRIdsz ", length=%" PRIdsz ")",
        target_offset, length);
  }

  const iree_device_size_t max_update_length =
      IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE;
  iree_device_size_t transfer_count_device =
      length == 0 ? 0 : iree_device_size_ceil_div(length, max_update_length);
  if (transfer_count_device > (iree_device_size_t)IREE_HOST_SIZE_MAX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "queue update requires too many transfer chunks");
  }
  iree_host_size_t transfer_count = (iree_host_size_t)transfer_count_device;

  iree_hal_transfer_command_t inline_command;
  iree_hal_transfer_command_t* transfer_commands =
      transfer_count <= 1 ? &inline_command : NULL;
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(device);
  iree_status_t status = iree_ok_status();
  if (transfer_count > 1) {
    iree_host_size_t transfer_commands_size = 0;
    if (!iree_host_size_checked_mul(transfer_count, sizeof(*transfer_commands),
                                    &transfer_commands_size)) {
      status =
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "queue update transfer command list is too large");
    } else {
      status = iree_allocator_malloc(host_allocator, transfer_commands_size,
                                     (void**)&transfer_commands);
    }
  }

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < transfer_count; ++i) {
      iree_device_size_t chunk_offset =
          (iree_device_size_t)i * max_update_length;
      iree_device_size_t chunk_length =
          iree_min(max_update_length, length - chunk_offset);
      transfer_commands[i] = (iree_hal_transfer_command_t){
          .type = IREE_HAL_TRANSFER_COMMAND_TYPE_UPDATE,
          .update =
              {
                  .source_buffer = source_buffer,
                  .source_offset =
                      source_offset + (iree_host_size_t)chunk_offset,
                  .target_buffer = target_buffer,
                  .target_offset = target_offset + chunk_offset,
                  .length = chunk_length,
              },
      };
    }
  }

  iree_hal_command_buffer_t* command_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_transfer_command_buffer(
        device, command_buffer_mode, queue_affinity, transfer_count,
        transfer_count == 0 ? NULL : transfer_commands, &command_buffer);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        command_buffer, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_EXECUTE_FLAG_NONE);
  }

  iree_hal_command_buffer_release(command_buffer);
  if (transfer_commands != &inline_command) {
    iree_allocator_free(host_allocator, transfer_commands);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_emulated_copy(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  // If we are starting execution immediately then we can reduce latency by
  // allowing inline command buffer execution.
  iree_hal_command_buffer_mode_t command_buffer_mode =
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT;
  if (wait_semaphore_list.count == 0) {
    command_buffer_mode |= IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION;
  }

  iree_hal_transfer_command_t command = {
      .type = IREE_HAL_TRANSFER_COMMAND_TYPE_COPY,
      .copy =
          {
              .source_buffer = source_buffer,
              .source_offset = source_offset,
              .target_buffer = target_buffer,
              .target_offset = target_offset,
              .length = length,
          },
  };

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_create_transfer_command_buffer(device, command_buffer_mode,
                                                  queue_affinity, 1, &command,
                                                  &command_buffer));

  iree_status_t status = iree_hal_device_queue_execute(
      device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE);

  iree_hal_command_buffer_release(command_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_emulated_dispatch(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  // If we are starting execution immediately then we can reduce latency by
  // allowing inline command buffer execution.
  iree_hal_command_buffer_mode_t command_buffer_mode =
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT;
  if (wait_semaphore_list.count == 0) {
    command_buffer_mode |= IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION;
  }

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_create(
              device, command_buffer_mode, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
              queue_affinity, /*binding_capacity=*/0, &command_buffer));

  iree_status_t status = iree_hal_command_buffer_begin(command_buffer);

  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_dispatch(command_buffer, executable,
                                              export_ordinal, config, constants,
                                              bindings, flags);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        command_buffer, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_EXECUTE_FLAG_NONE);
  }

  iree_hal_command_buffer_release(command_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
