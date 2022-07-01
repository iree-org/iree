// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LEVEL_ZERO_ALLOCATOR_H_
#define IREE_HAL_LEVEL_ZERO_ALLOCATOR_H_

#include "experimental/level_zero/context_wrapper.h"
#include "experimental/level_zero/status_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Create a Level Zero allocator.
iree_status_t iree_hal_level_zero_allocator_create(
    iree_hal_device_t* base_device, ze_device_handle_t level_zero_device,
    iree_hal_level_zero_context_wrapper_t* context,
    iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LEVEL_ZERO_ALLOCATOR_H_
