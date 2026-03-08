// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_NATIVE_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_VULKAN_NATIVE_SEMAPHORE_H_

#include <stdint.h>

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a timeline semaphore implemented using the native VkSemaphore type.
// |proactor| is borrowed from the device's proactor pool and must outlive the
// semaphore.
iree_status_t iree_hal_vulkan_native_semaphore_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_async_proactor_t* proactor, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore);

// Returns true if |semaphore| is a Vulkan native semaphore.
bool iree_hal_vulkan_native_semaphore_isa(iree_hal_semaphore_t* semaphore);

// Returns the Vulkan timeline semaphore handle.
VkSemaphore iree_hal_vulkan_native_semaphore_handle(
    iree_hal_semaphore_t* semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_NATIVE_SEMAPHORE_H_
