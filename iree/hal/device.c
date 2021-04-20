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

#include "iree/hal/device.h"

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"

#define _VTABLE_DISPATCH(device, method_name) \
  IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, method_name)

IREE_HAL_API_RETAIN_RELEASE(device);

IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_hal_device_id(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, id)(device);
}

IREE_API_EXPORT iree_allocator_t IREE_API_CALL
iree_hal_device_host_allocator(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, host_allocator)(device);
}

IREE_API_EXPORT iree_hal_allocator_t* IREE_API_CALL
iree_hal_device_allocator(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, device_allocator)(device);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_device_query_i32(
    iree_hal_device_t* device, iree_string_view_t key, int32_t* out_value) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, query_i32)(device, key, out_value);
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

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_device_queue_submit(
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

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_device_submit_and_wait(
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

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_device_wait_semaphores(
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
