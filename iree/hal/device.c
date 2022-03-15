// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/device.h"

#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

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

IREE_API_EXPORT
iree_status_t iree_hal_device_trim(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, trim)(device);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_query_i32(
    iree_hal_device_t* device, iree_string_view_t category,
    iree_string_view_t key, int32_t* out_value) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_value);

  if (iree_string_view_equal(category,
                             iree_make_cstring_view("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(iree_hal_device_id(device), key) ? 1 : 0;
    return iree_ok_status();
  }

  return _VTABLE_DISPATCH(device, query_i32)(device, category, key, out_value);
}

IREE_API_EXPORT iree_status_t iree_hal_device_transfer_range(
    iree_hal_device_t* device, iree_hal_transfer_buffer_t source,
    iree_device_size_t source_offset, iree_hal_transfer_buffer_t target,
    iree_device_size_t target_offset, iree_device_size_t data_length,
    iree_hal_transfer_buffer_flags_t flags, iree_timeout_t timeout) {
  if (data_length == 0) {
    return iree_ok_status();  // No-op.
  }

  // host->host is not allowed. We may want to support this one day to allow for
  // parallelized copies and such, however the validation code differs quite a
  // bit and it'd be better to have this as part of a task system API.
  bool is_source_host = source.device_buffer == NULL;
  bool is_target_host = target.device_buffer == NULL;
  if (is_source_host && is_target_host) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "cannot perform host->host transfers via this API, use memcpy/memmove");
  }

  // Check for overlap - like memcpy we require that the two ranges don't have
  // any overlap as we may use memcpy. This only matters if the buffers are
  // both device buffers - host and device should never alias: behavior is
  // undefined if a user tries to pass a mapped device pointer as if it was a
  // host pointer.
  if (!is_source_host && !is_target_host &&
      iree_hal_buffer_test_overlap(source.device_buffer, source_offset,
                                   data_length, target.device_buffer,
                                   target_offset, data_length) !=
          IREE_HAL_BUFFER_OVERLAP_DISJOINT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges must not overlap within the same buffer");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, is_source_host ? "h2d" : (is_target_host ? "d2h" : "d2d"));
  IREE_TRACE_ZONE_APPEND_VALUE(z0, data_length);

  // Defer to the backing implementation.
  iree_status_t status = _VTABLE_DISPATCH(device, transfer_range)(
      device, source, source_offset, target, target_offset, data_length, flags,
      timeout);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_transfer_and_wait(
    iree_hal_device_t* device, iree_hal_semaphore_t* wait_semaphore,
    uint64_t wait_value, iree_host_size_t transfer_count,
    const iree_hal_transfer_command_t* transfer_commands,
    iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(!transfer_count || transfer_commands);
  IREE_TRACE_ZONE_BEGIN(z0);

  // We only want to allow inline execution if we have not been instructed to
  // wait on a semaphore and it hasn't yet been signaled.
  iree_hal_command_buffer_mode_t mode = IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT;
  if (wait_semaphore) {
    uint64_t current_value = 0ull;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_semaphore_query(wait_semaphore, &current_value));
    if (current_value >= wait_value) {
      mode |= IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION;
    }
  } else {
    mode |= IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION;
  }

  // Create a command buffer performing all of the transfer operations.
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_create_transfer_command_buffer(
              device, mode, IREE_HAL_QUEUE_AFFINITY_ANY, transfer_count,
              transfer_commands, &command_buffer));

  // Perform a full submit-and-wait. On devices with multiple queues this can
  // run out-of-order/overlapped with other work and return earlier than device
  // idle.
  iree_hal_semaphore_t* fence_semaphore = NULL;
  iree_status_t status =
      iree_hal_semaphore_create(device, 0ull, &fence_semaphore);
  uint64_t signal_value = 1ull;
  if (iree_status_is_ok(status)) {
    iree_hal_submission_batch_t batch = {
        .wait_semaphores =
            {
                .count = wait_semaphore != NULL ? 1 : 0,
                .semaphores = &wait_semaphore,
                .payload_values = &wait_value,
            },
        .command_buffer_count = 1,
        .command_buffers = &command_buffer,
        .signal_semaphores =
            {
                .count = 1,
                .semaphores = &fence_semaphore,
                .payload_values = &signal_value,
            },
    };
    status = iree_hal_device_submit_and_wait(
        device, IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
        1, &batch, fence_semaphore, signal_value, timeout);
  }

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(fence_semaphore);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Validates that the submission is well-formed.
static iree_status_t iree_hal_device_validate_submission(
    iree_host_size_t batch_count, const iree_hal_submission_batch_t* batches) {
  for (iree_host_size_t i = 0; i < batch_count; ++i) {
    for (iree_host_size_t j = 0; j < batches[i].command_buffer_count; ++j) {
      if (batches[i].wait_semaphores.count > 0 &&
          iree_all_bits_set(
              iree_hal_command_buffer_mode(batches[i].command_buffers[j]),
              IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
        // Inline command buffers are not allowed to wait (as they could have
        // already been executed!). This is a requirement of the API so we
        // validate it across all backends even if they don't support inline
        // execution and ignore it.
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "inline command buffer submitted with a wait; inline command "
            "buffers must be ready to execute immediately");
      }
    }
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_submit(
    iree_hal_device_t* device, iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(!batch_count || batches);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_validate_submission(batch_count, batches));
  iree_status_t status = _VTABLE_DISPATCH(device, queue_submit)(
      device, command_categories, queue_affinity, batch_count, batches);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_submit_and_wait(
    iree_hal_device_t* device, iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
    iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(!batch_count || batches);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_validate_submission(batch_count, batches));
  iree_status_t status = _VTABLE_DISPATCH(device, submit_and_wait)(
      device, command_categories, queue_affinity, batch_count, batches,
      wait_semaphore, wait_value, timeout);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_wait_semaphores(
    iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(device);
  if (!semaphore_list || semaphore_list->count == 0) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, wait_semaphores)(
      device, wait_mode, semaphore_list, timeout);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_device_wait_idle(iree_hal_device_t* device, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, wait_idle)(device, timeout);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
