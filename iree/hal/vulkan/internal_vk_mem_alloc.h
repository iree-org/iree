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

#ifndef IREE_HAL_VULKAN_INTERNAL_VK_MEM_ALLOC_H_
#define IREE_HAL_VULKAN_INTERNAL_VK_MEM_ALLOC_H_

// Force all Vulkan calls to go through an indirect pVulkanFunctions interface.
// https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/configuration.html
#define VMA_STATIC_VULKAN_FUNCTIONS 0

// Allow VMA to query for dynamic functions we may not have provided.
// TODO(benvanik): see if we can remove this for more predictable failures; we
// want our code to be printing out nice symbol-not-found errors, not VMA
// abort()ing.
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include <vk_mem_alloc.h>

#endif  // IREE_HAL_VULKAN_INTERNAL_VK_MEM_ALLOC_H_
