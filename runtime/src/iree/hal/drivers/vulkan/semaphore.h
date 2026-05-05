// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_VULKAN_SEMAPHORE_H_

#include <stdint.h>

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a Vulkan HAL semaphore backed by a native timeline VkSemaphore.
iree_status_t iree_hal_vulkan_semaphore_create(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_async_proactor_t* proactor, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

// Returns true if |semaphore| is a Vulkan timeline semaphore.
bool iree_hal_vulkan_semaphore_isa(iree_hal_semaphore_t* semaphore);

// Returns the native Vulkan timeline semaphore handle.
iree_status_t iree_hal_vulkan_semaphore_handle(iree_hal_semaphore_t* semaphore,
                                               VkSemaphore* out_handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_SEMAPHORE_H_
