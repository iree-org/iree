// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CUDA_ALLOCATOR_H_
#define IREE_HAL_CUDA_ALLOCATOR_H_

#include "iree/hal/api.h"
#include "iree/hal/cuda/context_wrapper.h"
#include "iree/hal/cuda/status_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Create a cuda allocator.
iree_status_t iree_hal_cuda_allocator_create(
    iree_hal_cuda_context_wrapper_t* context,
    iree_hal_allocator_t** out_allocator);

// Free an allocation represent by the given device or host pointer.
void iree_hal_cuda_allocator_free(iree_hal_allocator_t* allocator,
                                  CUdeviceptr device_ptr, void* host_ptr,
                                  iree_hal_memory_type_t memory_type);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CUDA_ALLOCATOR_H_
