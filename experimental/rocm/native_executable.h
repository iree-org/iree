// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ROCM_NATIVE_EXECUTABLE_H_
#define IREE_HAL_ROCM_NATIVE_EXECUTABLE_H_

#include <stdint.h>

#include "experimental/rocm/context_wrapper.h"
#include "experimental/rocm/rocm_headers.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an executable from a HSACO module. The module may contain several
// kernels that can be extracted along with the associated block size.
iree_status_t iree_hal_rocm_native_executable_create(
    iree_hal_rocm_context_wrapper_t* context,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable);

hipFunction_t iree_hal_rocm_native_executable_for_entry_point(
    iree_hal_executable_t* executable, int32_t entry_point);

// Return the block size of the given |entry_point| within the executable.
iree_status_t iree_hal_rocm_native_executable_block_size(
    iree_hal_executable_t* executable, int32_t entry_point, uint32_t* x,
    uint32_t* y, uint32_t* z);

/// Return the layout associated with the entry point.
iree_hal_pipeline_layout_t* iree_hal_rocm_executable_get_layout(
    iree_hal_executable_t* executable, int32_t entry_point);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_NATIVE_EXECUTABLE_H_
