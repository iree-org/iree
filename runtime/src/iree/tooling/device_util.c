// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/device_util.h"

#include "iree/base/internal/call_once.h"
#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/init.h"

static iree_once_flag iree_hal_driver_registry_init_flag = IREE_ONCE_FLAG_INIT;
static void iree_hal_driver_registry_init_from_flags(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_CHECK_OK(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));
  IREE_TRACE_ZONE_END(z0);
}
iree_hal_driver_registry_t* iree_hal_available_driver_registry(void) {
  iree_call_once(&iree_hal_driver_registry_init_flag,
                 iree_hal_driver_registry_init_from_flags);
  return iree_hal_driver_registry_default();
}
