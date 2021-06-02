// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/event.h"

#include <stddef.h>

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"
#include "iree/hal/resource.h"

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
