// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_NATIVE_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_VULKAN_NATIVE_SEMAPHORE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a timeline semaphore implemented using the native VkSemaphore type.
iree_status_t iree_hal_vulkan_native_semaphore_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore);

// Returns true if |semaphore| is a Vulkan native semaphore.
bool iree_hal_vulkan_native_semaphore_isa(iree_hal_semaphore_t* semaphore);

// Returns the Vulkan timeline semaphore handle.
VkSemaphore iree_hal_vulkan_native_semaphore_handle(
    iree_hal_semaphore_t* semaphore);

// Performs a multi-wait on one or more semaphores.
// By default this is an all-wait but |wait_flags| may contain
// VK_SEMAPHORE_WAIT_ANY_BIT to change to an any-wait.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the wait does not complete before
// |deadline_ns| elapses.
iree_status_t iree_hal_vulkan_native_semaphore_multi_wait(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout,
    VkSemaphoreWaitFlags wait_flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_NATIVE_SEMAPHORE_H_
