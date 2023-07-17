// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_NATIVE_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_VULKAN_NATIVE_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a native Vulkan API-based allocator that directly allocates memory
// from the underlying implementation with no pooling or suballocation.
iree_status_t iree_hal_vulkan_native_allocator_create(
    const iree_hal_vulkan_device_options_t* options, VkInstance instance,
    VkPhysicalDevice physical_device,
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_hal_device_t* device, iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_NATIVE_ALLOCATOR_H_
