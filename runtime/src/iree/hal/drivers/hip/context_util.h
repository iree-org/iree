// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_CONTEXT_UTIL_H_
#define IREE_HAL_DRIVERS_HIP_CONTEXT_UTIL_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/status_util.h"

static inline iree_status_t iree_hal_hip_set_context(
    const iree_hal_hip_dynamic_symbols_t* syms, hipCtx_t hip_context) {
  if (!hip_context) {
    return iree_ok_status();
  }
  IREE_TRACE({
    hipCtx_t current_context = NULL;
    IREE_HIP_RETURN_IF_ERROR(syms, hipCtxGetCurrent(&current_context),
                             "hipCtxGetCurrent");
    if (current_context != hip_context) {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_hip_set_context_switch");
      iree_status_t status =
          IREE_HIP_RESULT_TO_STATUS(syms, hipCtxSetCurrent(hip_context));
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  });
  return IREE_HIP_RESULT_TO_STATUS(syms, hipCtxSetCurrent(hip_context));
}

#endif  // IREE_HAL_DRIVERS_HIP_CONTEXT_UTIL_H_
