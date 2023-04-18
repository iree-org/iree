// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_CUDA_DEVICE_H_
#define IREE_HAL_DRIVERS_CUDA_CUDA_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/api.h"
#include "iree/hal/drivers/cuda/dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a device that owns and manages its own CUcontext.
iree_status_t iree_hal_cuda_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_cuda_device_params_t* params,
    iree_hal_cuda_dynamic_symbols_t* syms, CUdevice device,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

// Returns a CUDA context bound to the given `base_device` if it is a HAL CUDA
// device. Returns error if `base_device` is not a HAL CUDA device.
iree_status_t iree_hal_cuda_device_get_context(iree_hal_device_t* base_device,
                                               CUcontext* out_context);

// Returns the dynamic symbol table from the device's context.
iree_hal_cuda_dynamic_symbols_t* iree_hal_cuda_get_dynamic_symbols(
    iree_hal_device_t* base_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_CUDA_DEVICE_H_
