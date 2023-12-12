// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_DEBUG_REPORTER_H_
#define IREE_HAL_DRIVERS_VULKAN_DEBUG_REPORTER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"

// A debug reporter that works with the VK_EXT_debug_utils extension.
// One reporter should be created per VkInstance to receive callbacks from the
// API and route them to our logging systems.
//
// Since creating a reporter requires a VkInstance it's not possible to report
// on messages during instance creation. To work around this it's possible to
// pass a *CreateInfo struct to vkCreateInstance as part of the
// VkInstanceCreateInfo::pNext chain. The callback will only be used this way
// during the creation call after which users can create the real
// instance-specific reporter.
typedef struct iree_hal_vulkan_debug_reporter_t
    iree_hal_vulkan_debug_reporter_t;

iree_status_t iree_hal_vulkan_debug_reporter_allocate(
    VkInstance instance, iree::hal::vulkan::DynamicSymbols* syms,
    int32_t min_verbosity, bool check_errors,
    const VkAllocationCallbacks* allocation_callbacks,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_debug_reporter_t** out_reporter);

void iree_hal_vulkan_debug_reporter_free(
    iree_hal_vulkan_debug_reporter_t* reporter);

// Returns true if the reporter encountered an error.
// iree_hal_vulkan_debug_reporter_consume_status can be used once to get the
// full status describing the failure and subsequent calls will return the
// status code.
bool iree_hal_vulkan_debug_reporter_has_error(
    iree_hal_vulkan_debug_reporter_t* reporter);

// Returns the error to the caller (transfering ownership).
// The reporter will remain in a failed state with the status code.
iree_status_t iree_hal_vulkan_debug_reporter_consume_status(
    iree_hal_vulkan_debug_reporter_t* reporter);

#endif  // IREE_HAL_DRIVERS_VULKAN_DEBUG_REPORTER_H_
