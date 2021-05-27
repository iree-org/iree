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

// Attribute that can be used to specify the configuration of the
// cooperative matrix multiply instructions supported by the target
// device. This corresponds to `VkCooperativeMatrixPropertiesNV` structure:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkCooperativeMatrixPropertiesNV.html
def VK_CooperativeMatrixPropertiesNVAttr :
  StructAttr<"CooperativeMatrixPropertiesNVAttr", VK_Dialect, [
    StructFieldAttr<"mSize", I32Attr>,
    StructFieldAttr<"nSize", I32Attr>,
    StructFieldAttr<"kSize", I32Attr>,
    StructFieldAttr<"aType", TypeAttr>,
    StructFieldAttr<"bType", TypeAttr>,
    StructFieldAttr<"cType", TypeAttr>,
    StructFieldAttr<"resultType", TypeAttr>,
    StructFieldAttr<"scope", VK_ScopeNV_Attr>
]>;

def VK_CooperativeMatrixPropertiesNVArrayAttr :
  TypedArrayAttrBase<VK_CooperativeMatrixPropertiesNVAttr,
                     "CooperativeMatrixPropertiesNV array attribute">;

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
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceProperties.html
  StructFieldAttr<"maxComputeSharedMemorySize", I32Attr>,
  StructFieldAttr<"maxComputeWorkGroupInvocations", I32Attr>,
  StructFieldAttr<"maxComputeWorkGroupSize", I32ElementsAttr>,

  // Core Vulkan 1.0 physical device features.
  //
  // This corresponds to the `VkPhysicalDeviceFeatures` structure:
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceFeatures.html
  StructFieldAttr<"shaderFloat64", UnitAttr>,
  StructFieldAttr<"shaderInt16", UnitAttr>,
  StructFieldAttr<"shaderInt64", UnitAttr>,

  // Core Vulkan 1.1 physical device subgroup properties.
  //
  // This corresponds to the `VkPhysicalDeviceSubgroupProperties` structure:
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceSubgroupProperties.html

  // TODO(antiagainst): StructAttr does not actually support attribute kinds
  // that are derived from IntegerAttr well. So the nice parsing/printing for
  // VK_SubgroupFeatureAttr does not really kick in here. We need to enhance
  // upstream MLIR.
  StructFieldAttr<"subgroupFeatures", VK_SubgroupFeatureAttr>,
  StructFieldAttr<"subgroupSize", I32Attr>,

  // VK_KHR_16bit_storage features.
  //
  // This corresponds to the `VkPhysicalDevice16BitStorageFeatures` structure:
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDevice16BitStorageFeatures.html
  StructFieldAttr<"storageBuffer16BitAccess", UnitAttr>,
  StructFieldAttr<"storagePushConstant16", UnitAttr>,
  StructFieldAttr<"uniformAndStorageBuffer16BitAccess", UnitAttr>,

  // VK_KHR_8bit_storage features.
  //
  // This corresponds to the `VkPhysicalDevice8BitStorageFeatures` structure:
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDevice8BitStorageFeatures.html
  StructFieldAttr<"storageBuffer8BitAccess", UnitAttr>,
  StructFieldAttr<"storagePushConstant8", UnitAttr>,
  StructFieldAttr<"uniformAndStorageBuffer8BitAccess", UnitAttr>,

  // VK_KHR_shader_float16_int8 features.
  //
  // This corresponds to the `VkPhysicalDeviceShaderFloat16Int8Features`
  // structure:
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceShaderFloat16Int8Features.html
  StructFieldAttr<"shaderFloat16", UnitAttr>,
  StructFieldAttr<"shaderInt8", UnitAttr>,

  // VK_KHR_variable_pointers features.
  // This corresponds to the `VkPhysicalDeviceVariablePointersFeatures`
  // structure:
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceVariablePointersFeatures.html
  StructFieldAttr<"variablePointersStorageBuffer", UnitAttr>,
  StructFieldAttr<"variablePointers", UnitAttr>,

  // VkCooperativeMatrixPropertiesNV features.
  // This corresponds to `VkCoooperativeMatrixPropertiesNV` structure:
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkCooperativeMatrixPropertiesNV.html
  StructFieldAttr<
      "cooperativeMatrixPropertiesNV",
      DefaultValuedAttr<VK_CooperativeMatrixPropertiesNVArrayAttr, "{}">>
]>;

#endif  // IREE_DIALECT_VULKAN_VULKANATTRIBUTES
