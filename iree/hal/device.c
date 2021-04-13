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

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_device_queue_submit(
    iree_hal_device_t* device, iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(!batch_count || batches);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, queue_submit)(
      device, command_categories, queue_affinity, batch_count, batches);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_device_wait_semaphores_with_deadline(
    iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns) {
  IREE_ASSERT_ARGUMENT(device);
  if (!semaphore_list || semaphore_list->count == 0) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(device, wait_semaphores_with_deadline)(
          device, wait_mode, semaphore_list, deadline_ns);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_device_wait_semaphores_with_timeout(
    iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list,
    iree_duration_t timeout_ns) {
  IREE_ASSERT_ARGUMENT(device);
  if (!semaphore_list || semaphore_list->count == 0) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, wait_semaphores_with_timeout)(
      device, wait_mode, semaphore_list, timeout_ns);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_wait_idle_with_deadline(
    iree_hal_device_t* device, iree_time_t deadline_ns) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(device, wait_idle_with_deadline)(device, deadline_ns);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_wait_idle_with_timeout(
    iree_hal_device_t* device, iree_duration_t timeout_ns) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(device, wait_idle_with_timeout)(device, timeout_ns);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
