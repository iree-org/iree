// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_VULKAN_DIRECT_COMMAND_BUFFER_H_
#define IREE_HAL_VULKAN_DIRECT_COMMAND_BUFFER_H_

#include "iree/hal/api.h"
#include "iree/hal/vulkan/descriptor_pool_cache.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/tracing.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a command buffer that directly records into a VkCommandBuffer.
iree_status_t iree_hal_vulkan_direct_command_buffer_allocate(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree::hal::vulkan::VkCommandPoolHandle* command_pool,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_vulkan_tracing_context_t* tracing_context,
    iree::hal::vulkan::DescriptorPoolCache* descriptor_pool_cache,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns the native Vulkan VkCommandBuffer handle.
VkCommandBuffer iree_hal_vulkan_direct_command_buffer_handle(
    iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_DIRECT_COMMAND_BUFFER_H_
