// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_METAL_DIRECT_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_METAL_DIRECT_ALLOCATOR_H_

#import <Metal/Metal.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a straightforward Metal allocator from the given |device| that
// performs allocations separately without caching or suballocation.
//
// On macOS, we additionally need the command queue to encode commands to make
// buffer contents visible to the CPU for managed storage type.
//
// |out_allocator| must be released by the caller (see
// iree_hal_allocator_release).
iree_status_t iree_hal_metal_allocator_create(
    id<MTLDevice> device,
#if defined(IREE_PLATFORM_MACOS)
    id<MTLCommandQueue> queue,
#endif  // IREE_PLATFORM_MACOS
    iree_hal_metal_resource_hazard_tracking_mode_t resource_tracking_mode,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

#if defined(IREE_PLATFORM_MACOS)
// Returns the underyling MetalCommandQueue associated with the given
// |allocator|.
id<MTLCommandQueue> iree_hal_metal_allocator_command_queue(
    const iree_hal_allocator_t* allocator);
#endif  // IREE_PLATFORM_MACOS

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_METAL_DIRECT_ALLOCATOR_H_
