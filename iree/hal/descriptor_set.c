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

#include "iree/hal/descriptor_set.h"

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"

#define _VTABLE_DISPATCH(descriptor_set, method_name) \
  IREE_HAL_VTABLE_DISPATCH(descriptor_set, iree_hal_descriptor_set, method_name)

IREE_HAL_API_RETAIN_RELEASE(descriptor_set);

IREE_API_EXPORT iree_status_t iree_hal_descriptor_set_create(
    iree_hal_device_t* device, iree_hal_descriptor_set_layout_t* set_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_hal_descriptor_set_t** out_descriptor_set) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(set_layout);
  IREE_ASSERT_ARGUMENT(!binding_count || bindings);
  IREE_ASSERT_ARGUMENT(out_descriptor_set);
  *out_descriptor_set = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, create_descriptor_set)(
          device, set_layout, binding_count, bindings, out_descriptor_set);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
