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

#include "iree/hal/descriptor_set_layout.h"

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"

#define _VTABLE_DISPATCH(descriptor_set_layout, method_name) \
  IREE_HAL_VTABLE_DISPATCH(descriptor_set_layout,            \
                           iree_hal_descriptor_set_layout, method_name)

IREE_HAL_API_RETAIN_RELEASE(descriptor_set_layout);

IREE_API_EXPORT iree_status_t iree_hal_descriptor_set_layout_create(
    iree_hal_device_t* device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(!binding_count || bindings);
  IREE_ASSERT_ARGUMENT(out_descriptor_set_layout);
  *out_descriptor_set_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device,
                                                  create_descriptor_set_layout)(
      device, usage_type, binding_count, bindings, out_descriptor_set_layout);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
