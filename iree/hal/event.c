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

#include "iree/hal/event.h"

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"

#define _VTABLE_DISPATCH(event, method_name) \
  IREE_HAL_VTABLE_DISPATCH(event, iree_hal_event, method_name)

IREE_HAL_API_RETAIN_RELEASE(event);

IREE_API_EXPORT iree_status_t
iree_hal_event_create(iree_hal_device_t* device, iree_hal_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = IREE_HAL_VTABLE_DISPATCH(
      device, iree_hal_device, create_event)(device, out_event);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
