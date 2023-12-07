// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvironment.h"

#include "iree/compiler/Dialect/Vulkan/Utils/TargetTriple.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler::IREE::Vulkan {

namespace {

/// Gets the corresponding SPIR-V version for the ggiven Vulkan target
/// environment.
spirv::Version convertVersion(Vulkan::TargetEnvAttr vkTargetEnv) {
  // Special extension to enable SPIR-V 1.4.
  const bool has14Ext = (llvm::is_contained(vkTargetEnv.getExtensions(),
                                            Extension::VK_KHR_spirv_1_4));

  switch (vkTargetEnv.getVersion()) {
  case Version::V_1_0:
    // Vulkan 1.0 only supports SPIR-V 1.0 by default.
    return has14Ext ? spirv::Version::V_1_4 : spirv::Version::V_1_0;
  case Version::V_1_1:
    // Vulkan 1.1 supports up to SPIR-V 1.3 by default.
    return has14Ext ? spirv::Version::V_1_4 : spirv::Version::V_1_3;
  case Version::V_1_2:
    // Vulkan 1.1 supports up to SPIR-V 1.5 by default.
    return spirv::Version::V_1_5;
  case Version::V_1_3:
    // Vulkan 1.1 supports up to SPIR-V 1.6 by default.
    return spirv::Version::V_1_6;
  }
  return spirv::Version::V_1_0;
}

/// Gets the corresponding SPIR-V extensions for the given Vulkan target
/// environment.
void convertExtensions(Vulkan::TargetEnvAttr vkTargetEnv,
                       SmallVectorImpl<spirv::Extension> &extensions) {
  extensions.clear();

  for (Extension ext : vkTargetEnv.getExtensions()) {
    switch (ext) {
    case Extension::VK_KHR_16bit_storage:
      extensions.push_back(spirv::Extension::SPV_KHR_16bit_storage);
      break;
    case Extension::VK_KHR_8bit_storage:
      extensions.push_back(spirv::Extension::SPV_KHR_8bit_storage);
      break;
    case Extension::VK_KHR_shader_float16_int8:
      // This extension allows using certain SPIR-V capabilities.
      break;
    case Extension::VK_KHR_shader_integer_dot_product:
      extensions.push_back(spirv::Extension::SPV_KHR_integer_dot_product);
      break;
    case Extension::VK_KHR_spirv_1_4:
      // This extension only affects SPIR-V version.
      break;
    case Extension::VK_KHR_storage_buffer_storage_class:
      extensions.push_back(
          spirv::Extension::SPV_KHR_storage_buffer_storage_class);
      break;
    case Extension::VK_KHR_variable_pointers:
      extensions.push_back(spirv::Extension::SPV_KHR_variable_pointers);
      break;
    case Extension::VK_EXT_subgroup_size_control:
      // This extension allows specifying min/max subgroup size.
      break;
    case Extension::VK_KHR_cooperative_matrix:
      extensions.push_back(spirv::Extension::SPV_KHR_cooperative_matrix);
      break;
    }
  }
}

/// Gets the corresponding SPIR-V capabilities for the given Vulkan target
/// environment.
void convertCapabilities(Vulkan::TargetEnvAttr vkTargetEnv,
                         SmallVectorImpl<spirv::Capability> &capabilities) {
  // Add unconditionally supported capabilities.
  // Note that "Table 54. List of SPIR-V Capabilities and enabling features or
  // extensions" in the Vulkan spec contains the full list. Right now omit those
  // implicitly declared or not useful for us.
  capabilities.assign({spirv::Capability::Shader});

  auto vkCapabilities = vkTargetEnv.getCapabilitiesAttr();

#define MAP_PRIMITIVE_TYPE(type)                                               \
  if (vkCapabilities.getShader##type())                                        \
  capabilities.push_back(spirv::Capability::type)

  MAP_PRIMITIVE_TYPE(Float64);
  MAP_PRIMITIVE_TYPE(Float16);
  MAP_PRIMITIVE_TYPE(Int64);
  MAP_PRIMITIVE_TYPE(Int16);
  MAP_PRIMITIVE_TYPE(Int8);
#undef MAP_PRIMITIVE_TYPE

#define MAP_8_16_BIT_STORAGE(vkFeature, spvCap)                                \
  if (vkCapabilities.vkFeature())                                              \
  capabilities.push_back(spirv::Capability::spvCap)

  MAP_8_16_BIT_STORAGE(getStorageBuffer16BitAccess, StorageBuffer16BitAccess);
  MAP_8_16_BIT_STORAGE(getUniformAndStorageBuffer16BitAccess, StorageUniform16);
  MAP_8_16_BIT_STORAGE(getStoragePushConstant16, StoragePushConstant16);
  MAP_8_16_BIT_STORAGE(getStorageBuffer8BitAccess, StorageBuffer8BitAccess);
  MAP_8_16_BIT_STORAGE(getUniformAndStorageBuffer8BitAccess,
                       UniformAndStorageBuffer8BitAccess);
  MAP_8_16_BIT_STORAGE(getStoragePushConstant8, StoragePushConstant8);
#undef MAP_8_16_BIT_STORAGE

  auto subgroupFeatures = vkCapabilities.getSubgroupFeatures().getValue();

#define MAP_SUBGROUP_FEATURE(featureBit)                                       \
  if ((subgroupFeatures & SubgroupFeature::featureBit) ==                      \
      SubgroupFeature::featureBit)                                             \
  capabilities.push_back(spirv::Capability::GroupNonUniform##featureBit)

  if ((subgroupFeatures & SubgroupFeature::Basic) == SubgroupFeature::Basic) {
    capabilities.push_back(spirv::Capability::GroupNonUniform);
  }
  MAP_SUBGROUP_FEATURE(Vote);
  MAP_SUBGROUP_FEATURE(Arithmetic);
  MAP_SUBGROUP_FEATURE(Ballot);
  MAP_SUBGROUP_FEATURE(Shuffle);
  MAP_SUBGROUP_FEATURE(ShuffleRelative);
  MAP_SUBGROUP_FEATURE(Clustered);
  MAP_SUBGROUP_FEATURE(Quad);
  MAP_SUBGROUP_FEATURE(PartitionedNV);
#undef MAP_SUBGROUP_FEATURE

  if (vkCapabilities.getVariablePointers()) {
    capabilities.push_back(spirv::Capability::VariablePointers);
  }
  if (vkCapabilities.getVariablePointersStorageBuffer()) {
    capabilities.push_back(spirv::Capability::VariablePointersStorageBuffer);
  }
  if (vkCapabilities.getShaderIntegerDotProduct()) {
    capabilities.push_back(spirv::Capability::DotProduct);
    capabilities.push_back(spirv::Capability::DotProductInputAll);
    capabilities.push_back(spirv::Capability::DotProductInput4x8BitPacked);
    if (vkCapabilities.getShaderInt8()) {
      capabilities.push_back(spirv::Capability::DotProductInput4x8Bit);
    }
  }
  if (ArrayAttr attr = vkCapabilities.getCooperativeMatrixPropertiesKHR()) {
    if (!attr.empty()) {
      capabilities.push_back(spirv::Capability::CooperativeMatrixKHR);
    }
  }
}

/// Gets the corresponding SPIR-V resource limits for the given Vulkan target
/// environment.
spirv::ResourceLimitsAttr
convertResourceLimits(Vulkan::TargetEnvAttr vkTargetEnv) {
  MLIRContext *context = vkTargetEnv.getContext();
  Builder builder(context);
  auto vkCapabilities = vkTargetEnv.getCapabilitiesAttr();
  SmallVector<Attribute, 1> khrCoopAttrs;
  if (ArrayAttr attr = vkCapabilities.getCooperativeMatrixPropertiesKHR()) {
    for (auto props :
         attr.getAsRange<Vulkan::CooperativeMatrixPropertiesKHRAttr>()) {
      auto scope = static_cast<spirv::Scope>(props.getScope().getValue());
      khrCoopAttrs.push_back(spirv::CooperativeMatrixPropertiesKHRAttr::get(
          context, props.getMSize(), props.getNSize(), props.getKSize(),
          props.getAType(), props.getBType(), props.getCType(),
          props.getResultType(), props.getAccSat(),
          spirv::ScopeAttr::get(context, scope)));
    }
  }
  auto sizeValues =
      vkCapabilities.getMaxComputeWorkGroupSize().getValues<int32_t>();
  SmallVector<int64_t> sizes;
  sizes.insert(sizes.end(), sizeValues.begin(), sizeValues.end());
  return spirv::ResourceLimitsAttr::get(
      context, vkCapabilities.getMaxComputeSharedMemorySize(),
      vkCapabilities.getMaxComputeWorkGroupInvocations(),
      builder.getI64ArrayAttr(sizes), vkCapabilities.getSubgroupSize(),
      vkCapabilities.getMinSubgroupSize(), vkCapabilities.getMaxSubgroupSize(),
      ArrayAttr::get(context, khrCoopAttrs), ArrayAttr{});
}

} // namespace

Vulkan::TargetEnvAttr getTargetEnvForTriple(MLIRContext *context,
                                            llvm::StringRef triple) {
  return TargetTriple::get(triple.data()).getTargetEnv(context);
}

spirv::TargetEnvAttr convertTargetEnv(Vulkan::TargetEnvAttr vkTargetEnv) {
  auto spvVersion = convertVersion(vkTargetEnv);

  SmallVector<spirv::Extension> spvExtensions;
  convertExtensions(vkTargetEnv, spvExtensions);

  SmallVector<spirv::Capability, 8> spvCapabilities;
  convertCapabilities(vkTargetEnv, spvCapabilities);

  auto spvLimits = convertResourceLimits(vkTargetEnv);

  auto triple = spirv::VerCapExtAttr::get(
      spvVersion, spvCapabilities, spvExtensions, vkTargetEnv.getContext());
  return spirv::TargetEnvAttr::get(
      triple, spvLimits, spirv::ClientAPI::Vulkan, vkTargetEnv.getVendorID(),
      vkTargetEnv.getDeviceType(), vkTargetEnv.getDeviceID());
}

} // namespace mlir::iree_compiler::IREE::Vulkan
