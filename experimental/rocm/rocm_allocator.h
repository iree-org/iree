// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ROCM_ALLOCATOR_H_
#define IREE_HAL_ROCM_ALLOCATOR_H_

#include "experimental/rocm/context_wrapper.h"
#include "experimental/rocm/status_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Create a ROCM allocator.
iree_status_t iree_hal_rocm_allocator_create(
    iree_hal_device_t* base_device, iree_hal_rocm_context_wrapper_t* context,
    iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_ALLOCATOR_H_
