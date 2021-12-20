// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_VULKAN_DIRECT_COMMAND_BUFFER_H_
#define IREE_HAL_VULKAN_DIRECT_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/vulkan/builtin_executables.h"
#include "iree/hal/vulkan/descriptor_pool_cache.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/tracing.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a command buffer that directly records into a VkCommandBuffer.
iree_status_t iree_hal_vulkan_direct_command_buffer_allocate(
    iree_hal_device_t* device,
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree::hal::vulkan::VkCommandPoolHandle* command_pool,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_vulkan_tracing_context_t* tracing_context,
    iree::hal::vulkan::DescriptorPoolCache* descriptor_pool_cache,
    iree::hal::vulkan::BuiltinExecutables* builtin_executables,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns the native Vulkan VkCommandBuffer handle.
VkCommandBuffer iree_hal_vulkan_direct_command_buffer_handle(
    iree_hal_command_buffer_t* command_buffer);

// Returns true if |command_buffer| is a Vulkan command buffer.
bool iree_hal_vulkan_direct_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_DIRECT_COMMAND_BUFFER_H_
