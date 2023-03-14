// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_LOCAL_SYNC_SYNC_DRIVER_H_
#define IREE_HAL_DRIVERS_LOCAL_SYNC_SYNC_DRIVER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/executable_loader.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a new synchronous local CPU driver that creates devices that perform
// execution inline on threads issuing submissions. |loaders| is the set of
// executable loaders that are available for loading in each device context.
iree_status_t iree_hal_sync_driver_create(
    iree_string_view_t identifier,
    const iree_hal_sync_device_params_t* default_params,
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_SYNC_SYNC_DRIVER_H_
