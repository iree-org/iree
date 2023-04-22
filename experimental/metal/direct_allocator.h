// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_METAL_DIRECT_ALLOCATOR_H_
#define IREE_EXPERIMENTAL_METAL_DIRECT_ALLOCATOR_H_

#import <Metal/Metal.h>

#include "experimental/metal/api.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Create a straightforward Metal allocator from the given |base_device| that
// performs allocations separately without caching or suballocation.
//
// |out_allocator| must be released by the caller (see
// iree_hal_allocator_release).
iree_status_t iree_hal_metal_allocator_create(
    iree_hal_device_t* base_device, id<MTLDevice> device,
    iree_hal_metal_resource_hazard_tracking_mode_t resource_tracking_mode,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

// Returns the underyling HAL device associated with the given |allocator|.
const iree_hal_device_t* iree_hal_metal_allocator_device(
    const iree_hal_allocator_t* allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_METAL_DIRECT_ALLOCATOR_H_
