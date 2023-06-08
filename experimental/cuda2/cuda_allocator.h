// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_CUDA2_CUDA_ALLOCATOR_H_
#define EXPERIMENTAL_CUDA2_CUDA_ALLOCATOR_H_

#include "experimental/cuda2/cuda_dynamic_symbols.h"
#include "experimental/cuda2/memory_pools.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a CUDA memory allocator.
// |device| and |stream| will be used for management operations.
// |pools| provides memory pools that may be shared across multiple allocators
// and the pointer must remain valid for the lifetime of the allocator.
iree_status_t iree_hal_cuda2_allocator_create(
    iree_hal_device_t* base_device,
    const iree_hal_cuda2_dynamic_symbols_t* cuda_symbols, CUdevice device,
    CUstream stream, iree_hal_cuda2_memory_pools_t* pools,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_CUDA2_CUDA_ALLOCATOR_H_
