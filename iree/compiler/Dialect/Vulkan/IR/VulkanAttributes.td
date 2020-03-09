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

// Dictionary attribute containing various Vulkan capability bits. This is
// aggregated from various Vulkan properties, limits, features from the spec.
//
// Note that we are using UnitAttr for booleans to allow omitting to mean false.
// TODO(antiagainst): support DefaultValuedAttr in StrucctAttr to allow
// specifying defaults for non-boolean fields.
def VK_CapabilitiesAttr : StructAttr<"CapabilitiesAttr", VK_Dialect, [
  // Core Vulkan 1.0 physical device properties.
  //
  // This corresponds to the `VkPhysicalDeviceProperties` structure:
  // https://renderdoc.org/vkspec_chunked/chap4.html#VkPhysicalDeviceProperties.
  StructFieldAttr<"maxComputeWorkGroupInvocations", I32Attr>,
  StructFieldAttr<"maxComputeWorkGroupSize", I32ElementsAttr>,

  // Core Vulkan 1.0 physical device features.
  //
  // This corresponds to the `VkPhysicalDeviceFeatures` structure:
  // https://renderdoc.org/vkspec_chunked/chap35.html#VkPhysicalDeviceFeatures
  StructFieldAttr<"shaderFloat64", UnitAttr>,
  StructFieldAttr<"shaderInt64", UnitAttr>,
  StructFieldAttr<"shaderInt16", UnitAttr>,

  // VK_KHR_shader_float16_int8 features.
  //
  // This corresponds to the `VkPhysicalDeviceShaderFloat16Int8Features`
  // structure:
  // https://renderdoc.org/vkspec_chunked/chap35.html#VkPhysicalDeviceShaderFloat16Int8Features
  StructFieldAttr<"shaderFloat16", UnitAttr>,
  StructFieldAttr<"shaderInt8", UnitAttr>
]>;

#endif  // IREE_DIALECT_VULKAN_VULKANATTRIBUTES
