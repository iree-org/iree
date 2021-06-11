// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_H_
#define IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a native Vulkan VkDescriptorSet object.
iree_status_t iree_hal_vulkan_native_descriptor_set_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device, VkDescriptorSet handle,
    iree_hal_descriptor_set_t** out_descriptor_set);

// Returns the native Vulkan VkDescriptorSet handle.
VkDescriptorSet iree_hal_vulkan_native_descriptor_set_handle(
    iree_hal_descriptor_set_t* base_descriptor_set);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_H_
