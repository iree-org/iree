// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ROCM_ROCM_DEVICE_H_
#define IREE_HAL_ROCM_ROCM_DEVICE_H_

#include "experimental/rocm/api.h"
#include "experimental/rocm/dynamic_symbols.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a device that owns and manages its own hipContext.
iree_status_t iree_hal_rocm_device_create(iree_hal_driver_t* driver,
                                          iree_string_view_t identifier,
                                          const iree_hal_rocm_device_params_t* params,
                                          iree_hal_rocm_dynamic_symbols_t* syms,
                                          hipDevice_t device,
                                          iree_allocator_t host_allocator,
                                          iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_ROCM_DEVICE_H_
