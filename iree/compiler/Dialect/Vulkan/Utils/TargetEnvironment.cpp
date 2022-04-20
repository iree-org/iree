// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvironment.h"

#include "iree/compiler/Dialect/Vulkan/Utils/TargetTriple.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Vulkan {

namespace {
/// Gets the corresponding SPIR-V version for the ggiven Vulkan target
/// environment.
spirv::Version convertVersion(Vulkan::TargetEnvAttr vkTargetEnv) {
  // Vulkan 1.2 supports up to SPIR-V 1.5 by default.
  if (vkTargetEnv.getVersion() == Version::V_1_2) return spirv::Version::V_1_5;

  // Special extension to enable SPIR-V 1.4.
  if (llvm::is_contained(vkTargetEnv.getExtensions(),
                         Extension::VK_KHR_spirv_1_4))
    return spirv::Version::V_1_4;

  switch (vkTargetEnv.getVersion()) {
    case Version::V_1_0:
      // Vulkan 1.0 only supports SPIR-V 1.0 by default.
      return spirv::Version::V_1_0;
    case Version::V_1_1:
      // Vulkan 1.1 supports up to SPIR-V 1.3 by default.
      return spirv::Version::V_1_3;
    default:
      break;
  }
  assert(false && "unhandled Vulkan version!");
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
      case Extension::VK_NV_cooperative_matrix:
        extensions.push_back(spirv::Extension::SPV_NV_cooperative_matrix);
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

#define MAP_PRIMITIVE_TYPE(type)     \
  if (vkCapabilities.shader##type()) \
  capabilities.push_back(spirv::Capability::type)

  MAP_PRIMITIVE_TYPE(Float64);
  MAP_PRIMITIVE_TYPE(Float16);
  MAP_PRIMITIVE_TYPE(Int64);
  MAP_PRIMITIVE_TYPE(Int16);
  MAP_PRIMITIVE_TYPE(Int8);
#undef MAP_PRIMITIVE_TYPE

#define MAP_8_16_BIT_STORAGE(vkFeature, spvCap) \
  if (vkCapabilities.vkFeature())               \
  capabilities.push_back(spirv::Capability::spvCap)

  MAP_8_16_BIT_STORAGE(storageBuffer16BitAccess, StorageBuffer16BitAccess);
  MAP_8_16_BIT_STORAGE(uniformAndStorageBuffer16BitAccess, StorageUniform16);
  MAP_8_16_BIT_STORAGE(storagePushConstant16, StoragePushConstant16);
  MAP_8_16_BIT_STORAGE(storageBuffer8BitAccess, StorageBuffer8BitAccess);
  MAP_8_16_BIT_STORAGE(uniformAndStorageBuffer8BitAccess,
                       UniformAndStorageBuffer8BitAccess);
  MAP_8_16_BIT_STORAGE(storagePushConstant8, StoragePushConstant8);
#undef MAP_8_16_BIT_STORAGE

  auto subgroupFeatures = vkCapabilities.subgroupFeatures().getValue();

#define MAP_SUBGROUP_FEATURE(featureBit)                  \
  if ((subgroupFeatures & SubgroupFeature::featureBit) == \
      SubgroupFeature::featureBit)                        \
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

  if (vkCapabilities.variablePointers()) {
    capabilities.push_back(spirv::Capability::VariablePointers);
  }
  if (vkCapabilities.variablePointersStorageBuffer()) {
    capabilities.push_back(spirv::Capability::VariablePointersStorageBuffer);
  }
  if (ArrayAttr attr = vkCapabilities.cooperativeMatrixPropertiesNV()) {
    if (!attr.empty()) {
      capabilities.push_back(spirv::Capability::CooperativeMatrixNV);
    }
  }
}

/// Gets the corresponding SPIR-V resource limits for the given Vulkan target
/// environment.
spirv::ResourceLimitsAttr convertResourceLimits(
    Vulkan::TargetEnvAttr vkTargetEnv) {
  MLIRContext *context = vkTargetEnv.getContext();
  auto vkCapabilities = vkTargetEnv.getCapabilitiesAttr();
  SmallVector<Attribute, 1> spvAttrs;
  if (ArrayAttr attr = vkCapabilities.cooperativeMatrixPropertiesNV()) {
    for (auto cooperativeMatrixPropertiesNV :
         attr.getAsRange<Vulkan::CooperativeMatrixPropertiesNVAttr>()) {
      spvAttrs.push_back(spirv::CooperativeMatrixPropertiesNVAttr::get(
          cooperativeMatrixPropertiesNV.mSize(),
          cooperativeMatrixPropertiesNV.nSize(),
          cooperativeMatrixPropertiesNV.kSize(),
          cooperativeMatrixPropertiesNV.aType(),
          cooperativeMatrixPropertiesNV.bType(),
          cooperativeMatrixPropertiesNV.cType(),
          cooperativeMatrixPropertiesNV.resultType(),
          cooperativeMatrixPropertiesNV.scope().cast<spirv::ScopeAttr>(),
          context));
    }
  }
  return spirv::ResourceLimitsAttr::get(
      vkCapabilities.maxComputeSharedMemorySize(),
      vkCapabilities.maxComputeWorkGroupInvocations(),
      vkCapabilities.maxComputeWorkGroupSize(), vkCapabilities.subgroupSize(),
      ArrayAttr::get(context, spvAttrs), context);
}
}  // anonymous namespace

Vulkan::TargetEnvAttr getTargetEnvForTriple(MLIRContext *context,
                                            llvm::StringRef triple) {
  return TargetTriple::get(triple.data()).getTargetEnv(context);
}

spirv::TargetEnvAttr convertTargetEnv(Vulkan::TargetEnvAttr vkTargetEnv) {
  auto spvVersion = convertVersion(vkTargetEnv);

  SmallVector<spirv::Extension, 4> spvExtensions;
  convertExtensions(vkTargetEnv, spvExtensions);

  SmallVector<spirv::Capability, 8> spvCapabilities;
  convertCapabilities(vkTargetEnv, spvCapabilities);

  auto spvLimits = convertResourceLimits(vkTargetEnv);

  auto triple = spirv::VerCapExtAttr::get(
      spvVersion, spvCapabilities, spvExtensions, vkTargetEnv.getContext());
  return spirv::TargetEnvAttr::get(triple, vkTargetEnv.getVendorID(),
                                   vkTargetEnv.getDeviceType(),
                                   vkTargetEnv.getDeviceID(), spvLimits);
}

}  // namespace Vulkan
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
