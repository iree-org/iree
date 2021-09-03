// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ROCM_EXECUTABLE_LAYOUT_H_
#define IREE_HAL_ROCM_EXECUTABLE_LAYOUT_H_

#include "experimental/rocm/context_wrapper.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates the kernel arguments.
iree_status_t iree_hal_rocm_executable_layout_create(
    iree_hal_rocm_context_wrapper_t* context, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_host_size_t push_constant_count,
    iree_hal_executable_layout_t** out_executable_layout);

// Return the base binding index for the given set.
iree_host_size_t iree_hal_rocm_base_binding_index(
    iree_hal_executable_layout_t* executable_layout, uint32_t set);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_EXECUTABLE_LAYOUT_H_
