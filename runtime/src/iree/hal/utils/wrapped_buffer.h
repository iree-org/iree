// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_WRAPPED_BUFFER_H_
#define IREE_HAL_UTILS_WRAPPED_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_wrapped_buffer_t iree_hal_wrapped_buffer_t;

bool iree_hal_wrapped_buffer_isa(iree_hal_buffer_t* base_value);

iree_status_t iree_hal_wrapped_buffer_make_buffer(
    iree_hal_buffer_t* wrapped_buffer, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer);

void iree_hal_wrapped_buffer_get_wrapped_buffer(iree_hal_buffer_t* base_buffer,
                                                iree_hal_buffer_t** out_buffer);

void iree_hal_wrapped_buffer_set_wrapped_buffer(
    iree_hal_buffer_t* base_buffer, iree_hal_buffer_t* wrapped_buffer);

iree_device_size_t iree_hal_wrapped_buffer_allocation_size(
    const iree_hal_buffer_t* base_buffer);

void iree_hal_wrapped_buffer_get_buffer_params(
    iree_hal_buffer_t* base_buffer, iree_hal_buffer_params_t* out_params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_WRAPPED_BUFFER_H_
