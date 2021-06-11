// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_LAYOUT_H_
#define IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a native Vulkan VkDescriptorSetLayout object.
iree_status_t iree_hal_vulkan_native_descriptor_set_layout_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

// Returns the native Vulkan VkDescriptorSetLayout handle.
VkDescriptorSetLayout iree_hal_vulkan_native_descriptor_set_layout_handle(
    iree_hal_descriptor_set_layout_t* base_descriptor_set_layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_LAYOUT_H_
