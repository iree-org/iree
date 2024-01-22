// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/device.h"

#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

//===----------------------------------------------------------------------===//
// iree_hal_device_t
//===----------------------------------------------------------------------===//

#define _VTABLE_DISPATCH(device, method_name) \
  IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, method_name)

IREE_HAL_API_RETAIN_RELEASE(device);

IREE_API_EXPORT iree_string_view_t
iree_hal_device_id(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, id)(device);
}

IREE_API_EXPORT iree_allocator_t
iree_hal_device_host_allocator(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, host_allocator)(device);
}

IREE_API_EXPORT iree_hal_allocator_t* iree_hal_device_allocator(
    iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, device_allocator)(device);
}

IREE_API_EXPORT void iree_hal_device_replace_allocator(
    iree_hal_device_t* device, iree_hal_allocator_t* new_allocator) {
  IREE_ASSERT_ARGUMENT(device);
  _VTABLE_DISPATCH(device, replace_device_allocator)(device, new_allocator);
}

IREE_API_EXPORT void iree_hal_device_replace_channel_provider(
    iree_hal_device_t* device, iree_hal_channel_provider_t* new_provider) {
  IREE_ASSERT_ARGUMENT(device);
  _VTABLE_DISPATCH(device, replace_channel_provider)(device, new_provider);
}

IREE_API_EXPORT
iree_status_t iree_hal_device_trim(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, trim)(device);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_query_i64(
    iree_hal_device_t* device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_value);

  if (iree_string_view_equal(category,
                             iree_make_cstring_view("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(iree_hal_device_id(device), key) ? 1 : 0;
    return iree_ok_status();
  }

  return _VTABLE_DISPATCH(device, query_i64)(device, category, key, out_value);
}

IREE_API_EXPORT iree_hal_semaphore_compatibility_t
iree_hal_device_query_semaphore_compatibility(iree_hal_device_t* device,
                                              iree_hal_semaphore_t* semaphore) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(semaphore);
  return _VTABLE_DISPATCH(device, query_semaphore_compatibility)(device,
                                                                 semaphore);
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_alloca(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(
      !wait_semaphore_list.count ||
      (wait_semaphore_list.semaphores && wait_semaphore_list.payload_values));
  IREE_ASSERT_ARGUMENT(!signal_semaphore_list.count ||
                       (signal_semaphore_list.semaphores &&
                        signal_semaphore_list.payload_values));
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, queue_alloca)(
      device, queue_affinity, wait_semaphore_list, signal_semaphore_list, pool,
      params, allocation_size, out_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_dealloca(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(
      !wait_semaphore_list.count ||
      (wait_semaphore_list.semaphores && wait_semaphore_list.payload_values));
  IREE_ASSERT_ARGUMENT(!signal_semaphore_list.count ||
                       (signal_semaphore_list.semaphores &&
                        signal_semaphore_list.payload_values));
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, queue_dealloca)(
      device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_fill(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
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

  iree_status_t status =
      iree_hal_device_queue_execute(device, queue_affinity, wait_semaphore_list,
                                    signal_semaphore_list, 1, &command_buffer);

  iree_hal_command_buffer_release(command_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_copy(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
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

  iree_status_t status =
      iree_hal_device_queue_execute(device, queue_affinity, wait_semaphore_list,
                                    signal_semaphore_list, 1, &command_buffer);

  iree_hal_command_buffer_release(command_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_read(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(
      !wait_semaphore_list.count ||
      (wait_semaphore_list.semaphores && wait_semaphore_list.payload_values));
  IREE_ASSERT_ARGUMENT(!signal_semaphore_list.count ||
                       (signal_semaphore_list.semaphores &&
                        signal_semaphore_list.payload_values));
  IREE_ASSERT_ARGUMENT(source_file);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, queue_read)(
      device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_write(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(
      !wait_semaphore_list.count ||
      (wait_semaphore_list.semaphores && wait_semaphore_list.payload_values));
  IREE_ASSERT_ARGUMENT(!signal_semaphore_list.count ||
                       (signal_semaphore_list.semaphores &&
                        signal_semaphore_list.payload_values));
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_file);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, queue_write)(
      device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_execute(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(
      !wait_semaphore_list.count ||
      (wait_semaphore_list.semaphores && wait_semaphore_list.payload_values));
  IREE_ASSERT_ARGUMENT(!signal_semaphore_list.count ||
                       (signal_semaphore_list.semaphores &&
                        signal_semaphore_list.payload_values));
  IREE_ASSERT_ARGUMENT(!command_buffer_count || command_buffers);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): move into devices instead? then a synchronous/inline device
  // could assert the waits are resolved instead of blanket failing on an
  // already-resolved semaphore. This would make using stream-ordered
  // allocations easier.
  for (iree_host_size_t i = 0; i < command_buffer_count; ++i) {
    if (wait_semaphore_list.count > 0 &&
        iree_all_bits_set(
            iree_hal_command_buffer_mode(command_buffers[i]),
            IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
      // Inline command buffers are not allowed to wait (as they could have
      // already been executed!). This is a requirement of the API so we
      // validate it across all backends even if they don't support inline
      // execution and ignore it.
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "inline command buffer submitted with a wait; inline command "
          "buffers must be ready to execute immediately");
    }
  }

  iree_status_t status = _VTABLE_DISPATCH(device, queue_execute)(
      device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      command_buffer_count, command_buffers);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_barrier(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      iree_hal_device_queue_execute(device, queue_affinity, wait_semaphore_list,
                                    signal_semaphore_list, 0, NULL);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_flush(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(device, queue_flush)(device, queue_affinity);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_wait_semaphores(
    iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(device);
  if (semaphore_list.count == 0) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, wait_semaphores)(
      device, wait_mode, semaphore_list, timeout);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_profiling_begin(
    iree_hal_device_t* device,
    const iree_hal_device_profiling_options_t* options) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(options);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(device, profiling_begin)(device, options);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_device_profiling_flush(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, profiling_flush)(device);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_device_profiling_end(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, profiling_end)(device);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_device_list_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_device_list_allocate(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    iree_hal_device_list_t** out_list) {
  IREE_ASSERT_ARGUMENT(out_list);
  *out_list = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_device_list_t* list = NULL;
  iree_host_size_t total_size =
      sizeof(*list) + capacity * sizeof(list->devices[0]);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&list));
  list->host_allocator = host_allocator;
  list->capacity = capacity;
  list->count = 0;
  *out_list = list;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_device_list_free(iree_hal_device_list_t* list) {
  if (!list) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = list->host_allocator;
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    iree_hal_device_release(list->devices[i]);
  }
  iree_allocator_free(host_allocator, list);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_device_list_push_back(
    iree_hal_device_list_t* list, iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  if (list->count + 1 <= list->capacity) {
    iree_hal_device_retain(device);
    list->devices[list->count++] = device;
  } else {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "list capacity %" PRIhsz
                              " reached; no more devices can be added",
                              list->capacity);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_hal_device_t* iree_hal_device_list_at(
    const iree_hal_device_list_t* list, iree_host_size_t i) {
  IREE_ASSERT_ARGUMENT(list);
  return i < list->count ? list->devices[i] : NULL;
}
