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

#ifndef IREE_DIALECT_VULKAN_VULKANATTRIBUTES
#define IREE_DIALECT_VULKAN_VULKANATTRIBUTES

include "iree/compiler/Dialect/Vulkan/IR/VulkanBase.td"

// TODO(antiagainst): consider auto-generating this file (or part of it) from
// vk.xml:
// https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/master/xml/vk.xml

// Core Vulkan 1.0 physical device properties.
//
// This corresponds to the `VkPhysicalDeviceProperties` structure:
// https://renderdoc.org/vkspec_chunked/chap4.html#VkPhysicalDeviceProperties.
def VK_Core10PropertiesAttr : StructAttr<"Core10PropertiesAttr", VK_Dialect, [
  StructFieldAttr<"maxComputeWorkGroupInvocations", I32Attr>,
  StructFieldAttr<"maxComputeWorkGroupSize", I32ElementsAttr>
]>;

#endif  // IREE_DIALECT_VULKAN_VULKANATTRIBUTES
