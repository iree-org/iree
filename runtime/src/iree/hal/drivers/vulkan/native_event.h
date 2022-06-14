// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_NATIVE_EVENT_H_
#define IREE_HAL_DRIVERS_VULKAN_NATIVE_EVENT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a native Vulkan VkEvent object.
iree_status_t iree_hal_vulkan_native_event_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_hal_event_t** out_event);

// Returns Vulkan event handle.
VkEvent iree_hal_vulkan_native_event_handle(const iree_hal_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_NATIVE_EVENT_H_
