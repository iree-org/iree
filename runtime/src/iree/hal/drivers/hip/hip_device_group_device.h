// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_DEVICE_GROUP_DEVICE_H_
#define IREE_HAL_DRIVERS_HIP_DEVICE_GROUP_DEVICE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/rccl_dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a device group from a set of hip devices that manage their own
// hipCtxs.
iree_status_t iree_hal_hip_device_group_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hip_device_params_t* params,
    const iree_hal_hip_dynamic_symbols_t* symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    uint32_t device_count, hipDevice_t* devices,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_DEVICE_GROUP_DEVICE_H_
