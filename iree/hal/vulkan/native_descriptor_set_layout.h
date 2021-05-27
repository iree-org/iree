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

#ifndef IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_LAYOUT_H_
#define IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_LAYOUT_H_

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
