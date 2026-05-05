// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_LOGICAL_DEVICE_H_
#define IREE_HAL_DRIVERS_VULKAN_LOGICAL_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_logical_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_logical_device_t
    iree_hal_vulkan_logical_device_t;

// Creates a driver-owned Vulkan HAL device by physical-device id.
iree_status_t iree_hal_vulkan_logical_device_create_by_id(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_hal_device_id_t device_id, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

// Creates a driver-owned Vulkan HAL device by physical-device path.
iree_status_t iree_hal_vulkan_logical_device_create_by_path(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_LOGICAL_DEVICE_H_
