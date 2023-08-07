// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_METAL_METAL_NOP_EXECUTABLE_CACHE_H_
#define IREE_HAL_DRIVERS_METAL_METAL_NOP_EXECUTABLE_CACHE_H_

#import <Metal/Metal.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a no-op executable cache that does not cache at all.
// This is useful to isolate pipeline caching behavior and verify compilation
// behavior.
//
// |out_executable_cache| must be released by the caller (see
// iree_hal_executable_cache_release).
iree_status_t iree_hal_metal_nop_executable_cache_create(
    id<MTLDevice> device, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_METAL_METAL_NOP_EXECUTABLE_CACHE_H_
