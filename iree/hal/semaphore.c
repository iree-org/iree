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

#include "iree/hal/semaphore.h"

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"

#define _VTABLE_DISPATCH(semaphore, method_name) \
  IREE_HAL_VTABLE_DISPATCH(semaphore, iree_hal_semaphore, method_name)

IREE_HAL_API_RETAIN_RELEASE(semaphore);

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_create(iree_hal_device_t* device, uint64_t initial_value,
                          iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, create_semaphore)(
          device, initial_value, out_semaphore);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_query(iree_hal_semaphore_t* semaphore, uint64_t* out_value) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(semaphore, query)(semaphore, out_value);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_signal(iree_hal_semaphore_t* semaphore, uint64_t new_value) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(semaphore, signal)(semaphore, new_value);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void IREE_API_CALL
iree_hal_semaphore_fail(iree_hal_semaphore_t* semaphore, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  _VTABLE_DISPATCH(semaphore, fail)(semaphore, status);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_semaphore_wait(
    iree_hal_semaphore_t* semaphore, uint64_t value, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(semaphore, wait)(semaphore, value, timeout);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
