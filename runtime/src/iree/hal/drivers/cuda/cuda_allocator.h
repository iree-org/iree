// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_CUDA_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_CUDA_CUDA_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda/memory_pools.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a CUDA memory allocator.
// |device| and |stream| will be used for management operations.
// |pools| provides memory pools that may be shared across multiple allocators
// and the pointer must remain valid for the lifetime of the allocator. Pools
// may not be supported on all devices and can be NULL.
iree_status_t iree_hal_cuda_allocator_create(
    const iree_hal_cuda_dynamic_symbols_t* cuda_symbols, CUdevice device,
    CUstream stream, iree_hal_cuda_memory_pools_t* pools,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_CUDA_ALLOCATOR_H_
