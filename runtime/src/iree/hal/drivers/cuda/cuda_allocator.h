// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_CUDA_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/context_wrapper.h"
#include "iree/hal/drivers/cuda/memory_pools.h"
#include "iree/hal/drivers/cuda/status_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a CUDA memory allocator.
// |device| and |stream| will be used for management operations.
// |pools| provides memory pools that may be shared across multiple allocators
// and the pointer must remain valid for the lifetime of the allocator.
iree_status_t iree_hal_cuda_allocator_create(
    iree_hal_device_t* base_device, iree_hal_cuda_context_wrapper_t* context,
    CUdevice device, CUstream stream, iree_hal_cuda_memory_pools_t* pools,
    iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_ALLOCATOR_H_
