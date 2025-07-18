// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_ERROR_CALLBACK_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_ERROR_CALLBACK_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_error_callback_t
//===----------------------------------------------------------------------===//

// Handles an asynchronous error from a component.
// May be called from driver threads and must not call back into the originating
// component or driver APIs. Ownership of |status| is transferred to the callee
// and must be freed if not retained for later use.
typedef void(IREE_API_PTR* iree_hal_amdgpu_error_callback_fn_t)(
    void* user_data, iree_status_t status);

// A callback for handling errors from a component.
//
// WARNING: this may be called from arbitrary driver threads and any non-const
// calls back into either the originating component or underlying driver are
// disallowed. Implementations should stash the status in a thread-safe manner
// and schedule their own callbacks to propagate the errors higher up the stack.
typedef struct iree_hal_amdgpu_error_callback_t {
  iree_hal_amdgpu_error_callback_fn_t fn;
  void* user_data;
} iree_hal_amdgpu_error_callback_t;

// Returns an error callback that does nothing.
// Not intended for use outside of testing/hacking.
static inline iree_hal_amdgpu_error_callback_t
iree_hal_amdgpu_error_callback_null(void) {
  iree_hal_amdgpu_error_callback_t callback = {
      /*.fn=*/NULL,
      /*.user_data=*/NULL,
  };
  return callback;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_ERROR_CALLBACK_H_
