// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_DEBUG_ALLOCATOR_H_
#define IREE_HAL_UTILS_DEBUG_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A HAL buffer allocator scribbles debug patterns into all allocated
// buffers. This can be used to identify incorrect use of memory. Note that this
// has severe performance implications and should only be used when diagnosing
// memory correctness issues.
typedef struct iree_hal_debug_allocator_t iree_hal_debug_allocator_t;

// Creates a debug allocator intercepting all |device_allocator| allocations.
// If needed |device| will be used for scheduling work.
iree_status_t iree_hal_debug_allocator_create(
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_DEBUG_ALLOCATOR_H_
