// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_VULKAN_DEVICE_H_
#define IREE_HAL_DRIVERS_VULKAN_VULKAN_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/debug_reporter.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/extensibility_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a device that owns and manages its own VkDevice.
//
// The |driver| will be retained for as long as the device is live such that if
// the driver owns the |instance| provided it is ensured to be valid. |driver|
// may be NULL if there is no parent driver to retain (such as when wrapping
// existing VkInstances provided by the application).
iree_status_t iree_hal_vulkan_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    iree_hal_vulkan_features_t requested_features,
    const iree_hal_vulkan_device_options_t* options,
    iree_hal_vulkan_syms_t* instance_syms, VkInstance instance,
    VkPhysicalDevice physical_device, iree_allocator_t host_allocator,
    iree_hal_vulkan_debug_reporter_t* debug_reporter,
    iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_VULKAN_DEVICE_H_
