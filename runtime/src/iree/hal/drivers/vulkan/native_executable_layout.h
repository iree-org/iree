// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_NATIVE_EXECUTABLE_LAYOUT_H_
#define IREE_HAL_DRIVERS_VULKAN_NATIVE_EXECUTABLE_LAYOUT_H_

// clang-format off: must be included before all other headers.
#include "iree/hal/drivers/vulkan/vulkan_headers.h"
// clang-format on

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a VkPipelineLayout-based executable layout composed of one or more
// descriptor set layouts.
iree_status_t iree_hal_vulkan_native_executable_layout_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_host_size_t push_constant_count, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_hal_executable_layout_t** out_executable_layout);

// Returns the native VkPipelineLayout handle for the executable layout.
VkPipelineLayout iree_hal_vulkan_native_executable_layout_handle(
    iree_hal_executable_layout_t* executable_layout);

// Returns the total number of descriptor sets within the layout.
iree_host_size_t iree_hal_vulkan_native_executable_layout_set_count(
    iree_hal_executable_layout_t* executable_layout);

// Returns the descriptor set layout with the given |set_index|.
iree_hal_descriptor_set_layout_t* iree_hal_vulkan_native_executable_layout_set(
    iree_hal_executable_layout_t* executable_layout,
    iree_host_size_t set_index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_NATIVE_EXECUTABLE_LAYOUT_H_
