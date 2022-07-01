// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LEVEL_ZERO_NOP_EXECUTABLE_CACHE_H_
#define IREE_HAL_LEVEL_ZERO_NOP_EXECUTABLE_CACHE_H_

#include "experimental/level_zero/context_wrapper.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a no-op executable cache that does not cache at all.
// This is useful to isolate pipeline caching behavior and verify compilation
// behavior.
iree_status_t iree_hal_level_zero_nop_executable_cache_create(
    iree_hal_level_zero_context_wrapper_t* context,
    iree_string_view_t identifier, ze_device_handle_t level_zero_device,
    iree_hal_executable_cache_t** out_executable_cache);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LEVEL_ZERO_NOP_EXECUTABLE_CACHE_H_
