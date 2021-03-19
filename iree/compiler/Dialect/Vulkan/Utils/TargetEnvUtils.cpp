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

#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvUtils.h"

#include "iree/compiler/Dialect/Vulkan/IR/VulkanTypes.h"
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
  llvm_unreachable("unhandled Vulkan version!");
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

// TODO(antiagainst): The following is good to get us started but it is
// certainly not a scalable way to describe GPU targets. We need more proper
// data structures and such.
const char *getTargetEnvForTriple(llvm::StringRef triple) {
  if (triple == "qualcomm-adreno640-unknown-android10") {
    // Example profile: https://vulkan.gpuinfo.org/displayreport.php?id=7175
    return R"(#vk.target_env<
      v1.1, r(87), [
        VK_KHR_storage_buffer_storage_class, VK_KHR_variable_pointers
      ], Qualcomm:IntegratedGPU, {
        maxComputeSharedMemorySize = 32768: i32,
        maxComputeWorkGroupInvocations = 1024: i32,
        maxComputeWorkGroupSize = dense<[1024, 1024, 64]>: vector<3xi32>,
        shaderInt16,
        subgroupFeatures = 3: i32,
        subgroupSize = 64: i32,
        variablePointersStorageBuffer, variablePointers
    }>)";
  }

  if (triple == "valhall-g77-unknown-android10") {
    // Example profile: https://vulkan.gpuinfo.org/displayreport.php?id=8046
    return R"(#vk.target_env<
      v1.1, r(108), [
        VK_KHR_16bit_storage, VK_KHR_8bit_storage, VK_KHR_shader_float16_int8,
        VK_KHR_storage_buffer_storage_class, VK_KHR_variable_pointers
      ], ARM:IntegratedGPU, {
        maxComputeSharedMemorySize = 32768: i32,
        maxComputeWorkGroupInvocations = 512: i32,
        maxComputeWorkGroupSize = dense<[512, 512, 512]>: vector<3xi32>,
        shaderInt16,
        subgroupFeatures = 1: i32,
        subgroupSize = 16: i32,
        storageBuffer16BitAccess, storagePushConstant16,
        uniformAndStorageBuffer16BitAccess,
        storageBuffer8BitAccess, uniformAndStorageBuffer8BitAccess,
        storagePushConstant8,
        shaderFloat16, shaderInt8,
        variablePointersStorageBuffer, variablePointers
    }>)";
  }

  if (triple == "swiftshader-unknown-unknown") {
    // Example profile: https://vulkan.gpuinfo.org/displayreport.php?id=9095
    return R"(#vk.target_env<
      v1.1, r(0), [VK_KHR_storage_buffer_storage_class], SwiftShader:CPU, {
        maxComputeSharedMemorySize = 16384: i32,
        maxComputeWorkGroupInvocations = 128: i32,
        maxComputeWorkGroupSize = dense<[128, 128, 64]>: vector<3xi32>,
        subgroupFeatures = 63: i32,
        subgroupSize = 4: i32
    }>)";
  }

  if (triple == "turing-t4-unknown-linux") {
    return R"(#vk.target_env<
     v1.2, r(133), [
        VK_KHR_16bit_storage, VK_KHR_8bit_storage, VK_KHR_shader_float16_int8,
        VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class,
        VK_KHR_variable_pointers, VK_NV_cooperative_matrix], NVIDIA:DiscreteGPU, {
       maxComputeSharedMemorySize = 49152: i32,
       maxComputeWorkGroupInvocations = 1024: i32,
       maxComputeWorkGroupSize = dense<[2147483647, 65535, 65535]> : vector<3xi32>,
       shaderFloat64, shaderInt16, shaderInt64,
       subgroupFeatures = 63: i32, subgroupSize = 32: i32,
       storageBuffer16BitAccess, storagePushConstant16,
       uniformAndStorageBuffer16BitAccess,
       storageBuffer8BitAccess, storagePushConstant8,
       uniformAndStorageBuffer8BitAccess,
       shaderFloat16, shaderInt8,
       variablePointersStorageBuffer, variablePointers,
       cooperativeMatrixPropertiesNV = [{
         mSize = 8: i32, nSize = 8: i32, kSize = 32: i32, aType = i8,
         bType = i8, cType = i32, resultType = i32, scope = 3: i32
       }, {
         mSize = 16: i32, nSize = 16: i32, kSize = 16: i32, aType = f16,
         bType = f16, cType = f16, resultType = f16, scope = 3: i32
       }, {
         mSize = 16: i32, nSize = 16: i32, kSize = 16: i32, aType = f16,
         bType = f16, cType = f32, resultType = f32, scope = 3: i32
       }]
    }>)";
  }
  return nullptr;
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
