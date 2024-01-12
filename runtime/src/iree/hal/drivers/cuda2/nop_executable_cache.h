// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA2_NOP_EXECUTABLE_CACHE_H_
#define IREE_HAL_DRIVERS_CUDA2_NOP_EXECUTABLE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda2/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda2/cuda_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a no-op executable cache that does not cache at all.
// This is useful to isolate pipeline caching behavior and verify compilation
// behavior.
iree_status_t iree_hal_cuda2_nop_executable_cache_create(
    iree_string_view_t identifier,
    const iree_hal_cuda2_dynamic_symbols_t* symbols, CUdevice device,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA2_NOP_EXECUTABLE_CACHE_H_
