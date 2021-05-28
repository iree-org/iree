// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CUDA_EXECUTABLE_LAYOUT_H_
#define IREE_HAL_CUDA_EXECUTABLE_LAYOUT_H_

#include "iree/hal/api.h"
#include "iree/hal/cuda/context_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates the kernel arguments.
iree_status_t iree_hal_cuda_executable_layout_create(
    iree_hal_cuda_context_wrapper_t* context, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_host_size_t push_constant_count,
    iree_hal_executable_layout_t** out_executable_layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CUDA_EXECUTABLE_LAYOUT_H_
