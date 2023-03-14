// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/pipeline_layout.h"

#include <stddef.h>

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"
#include "iree/hal/resource.h"

//===----------------------------------------------------------------------===//
// iree_hal_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

#define _VTABLE_DISPATCH(descriptor_set_layout, method_name) \
  IREE_HAL_VTABLE_DISPATCH(descriptor_set_layout,            \
                           iree_hal_descriptor_set_layout, method_name)

IREE_HAL_API_RETAIN_RELEASE(descriptor_set_layout);

IREE_API_EXPORT iree_status_t iree_hal_descriptor_set_layout_create(
    iree_hal_device_t* device, iree_hal_descriptor_set_layout_flags_t flags,
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
      device, flags, binding_count, bindings, out_descriptor_set_layout);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#undef _VTABLE_DISPATCH

//===----------------------------------------------------------------------===//
// iree_hal_pipeline_layout_t
//===----------------------------------------------------------------------===//

#define _VTABLE_DISPATCH(pipeline_layout, method_name)                \
  IREE_HAL_VTABLE_DISPATCH(pipeline_layout, iree_hal_pipeline_layout, \
                           method_name)

IREE_HAL_API_RETAIN_RELEASE(pipeline_layout);

IREE_API_EXPORT iree_status_t iree_hal_pipeline_layout_create(
    iree_hal_device_t* device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(!set_layout_count || set_layouts);
  IREE_ASSERT_ARGUMENT(out_pipeline_layout);
  *out_pipeline_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, create_pipeline_layout)(
          device, push_constants, set_layout_count, set_layouts,
          out_pipeline_layout);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#undef _VTABLE_DISPATCH
