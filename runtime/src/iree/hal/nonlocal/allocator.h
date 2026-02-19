// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_NONLOCAL_ALLOCATOR_H_
#define IREE_HAL_NONLOCAL_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a NL memory allocator.
// |device| and |stream| will be used for management operations.
// |pools| provides memory pools that may be shared across multiple allocators
// and the pointer must remain valid for the lifetime of the allocator. Pools
// may not be supported on all devices and can be NULL.
iree_status_t iree_hal_nl_allocator_create(
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_NONLOCAL_ALLOCATOR_H_
