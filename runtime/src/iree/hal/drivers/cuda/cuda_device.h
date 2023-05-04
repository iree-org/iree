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

// Returns true if |device| is a CUDA device.
bool iree_hal_cuda_device_isa(iree_hal_device_t* device);

// Returns a CUDA context bound to the given |device| if it is a CUDA device
// and otherwise returns NULL.
//
// WARNING: this API is unsafe and unstable. HAL devices may have any number of
// contexts and the context may be in use on other threads.
CUcontext iree_hal_cuda_device_context(iree_hal_device_t* device);

// Returns the dynamic symbol table from the |device| if it is a CUDA device
// and otherwise returns NULL.
//
// WARNING: the symbols are only valid for as long as the device is. Hosting
// libraries and applications should prefer to either link against CUDA
// themselves or maintain their own dynamic linking support: the IREE runtime
// only provides the symbols required by the HAL driver and not the entirety of
// the API.
iree_hal_cuda_dynamic_symbols_t* iree_hal_cuda_device_dynamic_symbols(
    iree_hal_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_CUDA_DEVICE_H_
