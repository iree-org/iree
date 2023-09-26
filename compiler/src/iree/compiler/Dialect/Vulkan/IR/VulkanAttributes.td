// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VULKAN_VULKANATTRIBUTES
#define IREE_DIALECT_VULKAN_VULKANATTRIBUTES

include "iree/compiler/Dialect/Vulkan/IR/VulkanBase.td"

class VK_Attr<string attrName, string attrMnemonic>
    : AttrDef<VK_Dialect, attrName> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

// Attribute that can be used to specify the configuration of the
// cooperative matrix multiply instructions supported by the target
// device. This corresponds to `VkCooperativeMatrixPropertiesKHR` structure:
// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkCooperativeMatrixPropertiesKHR.html
def VK_CooperativeMatrixPropertiesKHRAttr :
  VK_Attr<"CooperativeMatrixPropertiesKHR", "coop_matrix_props"> {
  let parameters = (ins
    "uint32_t":$mSize,
    "uint32_t":$nSize,
    "uint32_t":$kSize,
    "::mlir::Type":$aType,
    "::mlir::Type":$bType,
    "::mlir::Type":$cType,
    "::mlir::Type":$resultType,
    "bool":$accSat,
    "::mlir::iree_compiler::IREE::Vulkan::ScopeKHRAttr":$scope
  );
}

// TODO(antiagainst): consider auto-generating this file (or part of it) from
// vk.xml:
// https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/master/xml/vk.xml

// Dictionary attribute containing various Vulkan capability bits. This is
// aggregated from various Vulkan properties, limits, features from the spec.
//
// Note that we are using UnitAttr for booleans to allow omitting to mean false.
// TODO(antiagainst): support DefaultValuedAttr in StrucctAttr to allow
// specifying defaults for non-boolean fields.
def VK_CapabilitiesAttr : VK_Attr<"Capabilities", "caps"> {
  let parameters = (ins
    // Core Vulkan 1.0 physical device properties.
    //
    // This corresponds to the `VkPhysicalDeviceProperties` structure:
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceProperties.html
    "int":$maxComputeSharedMemorySize,
    "int":$maxComputeWorkGroupInvocations,
    "::mlir::DenseIntElementsAttr":$maxComputeWorkGroupSize,

    // Core Vulkan 1.0 physical device features.
    //
    // This corresponds to the `VkPhysicalDeviceFeatures` structure:
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceFeatures.html
    OptionalParameter<"::mlir::UnitAttr">:$shaderFloat64,
    OptionalParameter<"::mlir::UnitAttr">:$shaderInt16,
    OptionalParameter<"::mlir::UnitAttr">:$shaderInt64,

    // Core Vulkan 1.1 physical device subgroup properties.
    //
    // This corresponds to the `VkPhysicalDeviceSubgroupProperties` structure:
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceSubgroupProperties.html

    // TODO(antiagainst): StructAttr does not actually support attribute kinds
    // that are derived from IntegerAttr well. So the nice parsing/printing for
    // VK_SubgroupFeatureAttr does not really kick in here. We need to enhance
    // upstream MLIR.
    "::mlir::iree_compiler::IREE::Vulkan::SubgroupFeatureAttr":$subgroupFeatures,
    "int":$subgroupSize,

    // VK_EXT_subgroup_size_control features.
    //
    // This corresponds to the `VkPhysicalDeviceSubgroupSizeControlProperties` structure:
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceSubgroupSizeControlPropertiesEXT.html
    OptionalParameter<"::std::optional<int>">:$minSubgroupSize,
    OptionalParameter<"::std::optional<int>">:$maxSubgroupSize,

    // VK_KHR_16bit_storage features.
    //
    // This corresponds to the `VkPhysicalDevice16BitStorageFeatures` structure:
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDevice16BitStorageFeatures.html
    OptionalParameter<"::mlir::UnitAttr">:$storageBuffer16BitAccess,
    OptionalParameter<"::mlir::UnitAttr">:$storagePushConstant16,
    OptionalParameter<"::mlir::UnitAttr">:$uniformAndStorageBuffer16BitAccess,

    // VK_KHR_8bit_storage features.
    //
    // This corresponds to the `VkPhysicalDevice8BitStorageFeatures` structure:
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDevice8BitStorageFeatures.html
    OptionalParameter<"::mlir::UnitAttr">:$storageBuffer8BitAccess,
    OptionalParameter<"::mlir::UnitAttr">:$storagePushConstant8,
    OptionalParameter<"::mlir::UnitAttr">:$uniformAndStorageBuffer8BitAccess,

    // VK_KHR_shader_float16_int8 features.
    //
    // This corresponds to the `VkPhysicalDeviceShaderFloat16Int8Features`
    // structure:
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceShaderFloat16Int8Features.html
    OptionalParameter<"::mlir::UnitAttr">:$shaderFloat16,
    OptionalParameter<"::mlir::UnitAttr">:$shaderInt8,

    // VK_KHR_shader_integer_dot_product features.
    //
    // This corresponds to the `VkPhysicalDeviceShaderIntegerDotProductFeatures`
    // structure:
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR.html
    OptionalParameter<"::mlir::UnitAttr">:$shaderIntegerDotProduct,

    // VK_KHR_variable_pointers features.
    // This corresponds to the `VkPhysicalDeviceVariablePointersFeatures`
    // structure:
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceVariablePointersFeatures.html
    OptionalParameter<"::mlir::UnitAttr">:$variablePointersStorageBuffer,
    OptionalParameter<"::mlir::UnitAttr">:$variablePointers,

    // VkCooperativeMatrixPropertiesKHR features.
    // This corresponds to `VkCoooperativeMatrixPropertiesKHR` structure:
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_cooperative_matrix.html
    DefaultValuedParameter<"ArrayAttr", "nullptr">:$cooperativeMatrixPropertiesKHR
  );
}

#endif  // IREE_DIALECT_VULKAN_VULKANATTRIBUTES
