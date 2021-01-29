// Copyright 2020 Google LLC
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

#ifndef IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_H_
#define IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_H_

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
