// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Vulkan/Utils/TargetTriple.h"

#include <array>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Vulkan {

namespace {

/// Returns the GPU vendor for the given target `triple`.
spirv::Vendor getVendor(const TargetTriple &triple) {
  switch (triple.getArch()) {
    case TargetTripleArch::Unknown:
      return spirv::Vendor::Unknown;
    case TargetTripleArch::AMD_RDNAv1:
    case TargetTripleArch::AMD_RDNAv2:
      return spirv::Vendor::AMD;
    case TargetTripleArch::ARM_Valhall:
      return spirv::Vendor::ARM;
    case TargetTripleArch::NV_Turing:
    case TargetTripleArch::NV_Ampere:
      return spirv::Vendor::NVIDIA;
    case TargetTripleArch::QC_Adreno:
      return spirv::Vendor::Qualcomm;
    case TargetTripleArch::CPU:
      switch (triple.getProduct()) {
        case TargetTripleProduct::SwiftShader:
          return spirv::Vendor::SwiftShader;
        default:
          return spirv::Vendor::Unknown;
      }
    default:
      assert(false && "unhandled vendor");
      return spirv::Vendor::Unknown;
  }
}

/// Returns the GPU device type for the given target `triple`.
spirv::DeviceType getDeviceType(const TargetTriple &triple) {
  switch (triple.getArch()) {
    case TargetTripleArch::Unknown:
      return spirv::DeviceType::Unknown;
    case TargetTripleArch::CPU:
      return spirv::DeviceType::CPU;
    case TargetTripleArch::AMD_RDNAv1:
    case TargetTripleArch::AMD_RDNAv2:
    case TargetTripleArch::NV_Turing:
    case TargetTripleArch::NV_Ampere:
      return spirv::DeviceType::DiscreteGPU;
    case TargetTripleArch::ARM_Valhall:
    case TargetTripleArch::QC_Adreno:
      return spirv::DeviceType::IntegratedGPU;
    default:
      assert(false && "unhandled device type");
      return spirv::DeviceType::Unknown;
  }
}

/// Returns the Vulkan version for the given target `triple`.
Vulkan::Version getVersion(const TargetTriple &triple) {
  // Android 11 stays at Vulkan 1.1.
  if (triple.getOS() == TargetTripleOS::Android11) {
    return Version::V_1_1;
  }

  // SwiftShader stays at Vulkan 1.1.
  if (triple.getProduct() == TargetTripleProduct::SwiftShader) {
    return Version::V_1_1;
  }

  return Version::V_1_2;
}

/// Writes the Vulkan extensions supported by the given `triple` into
/// `extensions`.
///
/// Note that this is an "approximation": Android compatibility will provide
/// some minimal guarantee but still different Android devices can have
/// different set of extensions, depending on the Android and GPU driver
/// version. The GPU triple is a handy way to specify the target but we cannot
/// encode all the information in the triple.
void getExtensions(const TargetTriple &triple,
                   llvm::SmallVectorImpl<Vulkan::Extension> &extensions) {
  // Mobile GPUs need to take Android version into consideration.
  switch (triple.getArch()) {
    case TargetTripleArch::ARM_Valhall: {
      // Example: https://vulkan.gpuinfo.org/displayreport.php?id=10312
      const std::array<Extension, 6> list = {
          Extension::VK_KHR_16bit_storage,
          Extension::VK_KHR_8bit_storage,
          Extension::VK_KHR_shader_float16_int8,
          Extension::VK_KHR_spirv_1_4,
          Extension::VK_KHR_storage_buffer_storage_class,
          Extension::VK_KHR_variable_pointers,
      };
      return extensions.append(list.begin(), list.end());
    }
    case TargetTripleArch::QC_Adreno: {
      // Example: https://vulkan.gpuinfo.org/displayreport.php?id=10983
      const std::array<Extension, 5> list = {
          Extension::VK_KHR_16bit_storage,
          Extension::VK_KHR_shader_float16_int8,
          Extension::VK_KHR_spirv_1_4,
          Extension::VK_KHR_storage_buffer_storage_class,
          Extension::VK_KHR_variable_pointers,
      };
      return extensions.append(list.begin(), list.end());
    }
    default:
      break;
  }

  // SwiftShader is very limited regarding functionalities.
  if (getVendor(triple) == spirv::Vendor::SwiftShader) {
    extensions.push_back(Extension::VK_KHR_storage_buffer_storage_class);
    return;
  }

  // Desktop GPUs typically support all extensions we care.
  const std::array<Extension, 6> desktop = {
      Extension::VK_KHR_16bit_storage,
      Extension::VK_KHR_8bit_storage,
      Extension::VK_KHR_shader_float16_int8,
      Extension::VK_KHR_spirv_1_4,
      Extension::VK_KHR_storage_buffer_storage_class,
      Extension::VK_KHR_variable_pointers};

  extensions.append(desktop.begin(), desktop.end());
  if (getVendor(triple) == spirv::Vendor::NVIDIA) {
    extensions.push_back(Extension::VK_NV_cooperative_matrix);
  }
}

/// Returns the Vulkan features/limits/capabilities supported by the given
/// `triple`.
///
/// Note that this is an "approximation": Android compatibility will provide
/// some minimal guarantee but still different Android devices can have
/// different set of extensions, depending on the Android and GPU driver
/// version. The GPU triple is a handy way to specify the target but we cannot
/// encode all the information in the triple.
CapabilitiesAttr getCapabilities(const TargetTriple &triple,
                                 MLIRContext *context) {
  // Default to Vulkan required limits.
  int maxComputeSharedMemorySize = 16384;
  int maxComputeWorkGroupInvocations = 128;
  std::array<int, 3> maxComputeWorkGroupSize = {128, 128, 64};

  int subgroupSize = 32;
  SubgroupFeature subgroupFeatures = SubgroupFeature::Basic;

  bool shaderFloat16 = false, shaderFloat64 = false;
  bool shaderInt8 = false, shaderInt16 = false, shaderInt64 = false;

  bool storageBuffer16BitAccess = false, storagePushConstant16 = false;
  bool uniformAndStorageBuffer16BitAccess = false;
  bool storageBuffer8BitAccess = false, storagePushConstant8 = false;
  bool uniformAndStorageBuffer8BitAccess = false;

  bool variablePointers = false, variablePointersStorageBuffer = false;

  SmallVector<Attribute, 4> coopmatCases;

  Builder builder(context);

  switch (triple.getArch()) {
    case TargetTripleArch::Unknown:
      break;
    case TargetTripleArch::AMD_RDNAv1:
    case TargetTripleArch::AMD_RDNAv2:
      // Example: https://vulkan.gpuinfo.org/displayreport.php?id=10906
      maxComputeSharedMemorySize = 65536;
      maxComputeWorkGroupInvocations = 1024;
      maxComputeWorkGroupSize = {1024, 1024, 1024};

      subgroupSize = 64;
      subgroupFeatures = SubgroupFeature::Basic | SubgroupFeature::Vote |
                         SubgroupFeature::Arithmetic | SubgroupFeature::Ballot |
                         SubgroupFeature::Shuffle |
                         SubgroupFeature::ShuffleRelative |
                         SubgroupFeature::Clustered | SubgroupFeature::Quad;

      shaderFloat16 = shaderFloat64 = true;
      shaderInt8 = shaderInt16 = shaderInt64 = true;

      storageBuffer16BitAccess = storagePushConstant16 = true;
      uniformAndStorageBuffer16BitAccess = true;
      storageBuffer8BitAccess = true, storagePushConstant8 = true;
      uniformAndStorageBuffer8BitAccess = true;

      variablePointers = variablePointersStorageBuffer = true;
      break;
    case TargetTripleArch::ARM_Valhall:
      // Example: https://vulkan.gpuinfo.org/displayreport.php?id=10312
      maxComputeSharedMemorySize = 32768;
      maxComputeWorkGroupInvocations = 512;
      maxComputeWorkGroupSize = {512, 512, 512};

      subgroupSize = 16;
      subgroupFeatures = SubgroupFeature::Basic | SubgroupFeature::Vote |
                         SubgroupFeature::Arithmetic | SubgroupFeature::Ballot |
                         SubgroupFeature::Clustered | SubgroupFeature::Quad;

      shaderFloat16 = shaderInt8 = shaderInt16 = true;

      storageBuffer16BitAccess = storagePushConstant16 = true;
      uniformAndStorageBuffer16BitAccess = true;
      storageBuffer8BitAccess = true, storagePushConstant8 = true;
      uniformAndStorageBuffer8BitAccess = true;

      variablePointers = variablePointersStorageBuffer = true;
      break;
    case TargetTripleArch::CPU:
      if (triple.getProduct() == TargetTripleProduct::SwiftShader) {
        // Example: https://vulkan.gpuinfo.org/displayreport.php?id=11023
        maxComputeSharedMemorySize = 16384;

        subgroupSize = 4;
        subgroupFeatures = SubgroupFeature::Basic | SubgroupFeature::Vote |
                           SubgroupFeature::Arithmetic |
                           SubgroupFeature::Ballot | SubgroupFeature::Shuffle |
                           SubgroupFeature::ShuffleRelative;
      }
      break;
    case TargetTripleArch::NV_Turing:
    case TargetTripleArch::NV_Ampere: {
      // Example: https://vulkan.gpuinfo.org/displayreport.php?id=11252
      maxComputeSharedMemorySize = 49152;
      maxComputeWorkGroupInvocations = 1024;
      maxComputeWorkGroupSize = {1024, 1024, 64};

      subgroupSize = 32;
      subgroupFeatures = SubgroupFeature::Basic | SubgroupFeature::Vote |
                         SubgroupFeature::Arithmetic | SubgroupFeature::Ballot |
                         SubgroupFeature::Shuffle |
                         SubgroupFeature::ShuffleRelative |
                         SubgroupFeature::Clustered | SubgroupFeature::Quad;

      shaderFloat16 = shaderFloat64 = true;
      shaderInt8 = shaderInt16 = shaderInt64 = true;

      storageBuffer16BitAccess = storagePushConstant16 = true;
      uniformAndStorageBuffer16BitAccess = true;
      storageBuffer8BitAccess = true, storagePushConstant8 = true;
      uniformAndStorageBuffer8BitAccess = true;

      variablePointers = variablePointersStorageBuffer = true;

      auto i32v8 = builder.getI32IntegerAttr(8);
      auto i32v16 = builder.getI32IntegerAttr(16);
      auto i32v32 = builder.getI32IntegerAttr(32);
      auto i8t = TypeAttr::get(builder.getIntegerType(8));
      auto i32t = TypeAttr::get(builder.getIntegerType(32));
      auto f16t = TypeAttr::get(builder.getF16Type());
      auto f32t = TypeAttr::get(builder.getF32Type());
      auto scope = ScopeNVAttr::get(context, ScopeNV::Subgroup);

      coopmatCases.push_back(CooperativeMatrixPropertiesNVAttr::get(
          /*mSize=*/i32v8, /*nSize=*/i32v8, /*kSize=*/i32v32, /*aType=*/i8t,
          /*bType=*/i8t, /*cType=*/i32t, /*resultType=*/i32t, scope, context));
      coopmatCases.push_back(CooperativeMatrixPropertiesNVAttr::get(
          /*mSize=*/i32v16, /*nSize=*/i32v16, /*kSize=*/i32v16, /*aType=*/f16t,
          /*bType=*/f16t, /*cType=*/f16t, /*resultType=*/f16t, scope, context));
      coopmatCases.push_back(CooperativeMatrixPropertiesNVAttr::get(
          /*mSize=*/i32v16, /*nSize=*/i32v16, /*kSize=*/i32v16, /*aType=*/f16t,
          /*bType=*/f16t, /*cType=*/f32t, /*resultType=*/f32t, scope, context));
    } break;
    case TargetTripleArch::QC_Adreno:
      // Example: https://vulkan.gpuinfo.org/displayreport.php?id=10983
      maxComputeSharedMemorySize = 32768;
      maxComputeWorkGroupInvocations = 1024;
      maxComputeWorkGroupSize = {1024, 1024, 64};

      subgroupSize = 64;
      subgroupFeatures = SubgroupFeature::Basic | SubgroupFeature::Vote |
                         SubgroupFeature::Arithmetic | SubgroupFeature::Ballot |
                         SubgroupFeature::Shuffle |
                         SubgroupFeature::ShuffleRelative |
                         SubgroupFeature::Quad;

      shaderFloat16 = shaderInt8 = shaderInt16 = true;

      storageBuffer16BitAccess = true;
      variablePointers = variablePointersStorageBuffer = true;
      break;
  }

  auto getBoolAttr = [context](bool value) -> UnitAttr {
    return value ? UnitAttr::get(context) : UnitAttr();
  };

  return CapabilitiesAttr::get(
      builder.getI32IntegerAttr(maxComputeSharedMemorySize),
      builder.getI32IntegerAttr(maxComputeWorkGroupInvocations),
      builder.getI32VectorAttr(maxComputeWorkGroupSize),
      getBoolAttr(shaderFloat64), getBoolAttr(shaderInt16),
      getBoolAttr(shaderInt64),
      SubgroupFeatureAttr::get(context, subgroupFeatures),
      builder.getI32IntegerAttr(subgroupSize),
      getBoolAttr(storageBuffer16BitAccess), getBoolAttr(storagePushConstant16),
      getBoolAttr(uniformAndStorageBuffer16BitAccess),
      getBoolAttr(storageBuffer8BitAccess), getBoolAttr(storagePushConstant8),
      getBoolAttr(uniformAndStorageBuffer8BitAccess),
      getBoolAttr(shaderFloat16), getBoolAttr(shaderInt8),
      getBoolAttr(variablePointersStorageBuffer), getBoolAttr(variablePointers),
      builder.getArrayAttr(coopmatCases), context);
}
}  // namespace

TargetTriple TargetTriple::get(const char *triple) {
  llvm::SmallVector<llvm::StringRef, 3> fragments;
  llvm::SplitString(triple, fragments, "-");

  TargetTripleArch arch = TargetTripleArch::Unknown;
  if (auto symbol = symbolizeTargetTripleArch(fragments[0]))
    arch = symbol.getValue();

  TargetTripleProduct product = TargetTripleProduct::Unknown;
  if (auto symbol = symbolizeTargetTripleProduct(fragments[1]))
    product = symbol.getValue();

  TargetTripleOS os = TargetTripleOS::Unknown;
  if (auto symbol = symbolizeTargetTripleOS(fragments[2]))
    os = symbol.getValue();

  return TargetTriple(arch, product, os);
}

TargetTriple::TargetTriple(TargetTripleArch arch, TargetTripleProduct product,
                           TargetTripleOS os)
    : arch(arch), product(product), os(os) {}

std::string TargetTriple::getTriple() const {
  llvm::StringRef archStr = stringifyTargetTripleArch(arch);
  llvm::StringRef productStr = stringifyTargetTripleProduct(product);
  llvm::StringRef osStr = stringifyTargetTripleOS(os);
  return llvm::formatv("{0}-{1}-{2}", archStr, productStr, osStr);
}

TargetEnvAttr TargetTriple::getTargetEnv(MLIRContext *context) const {
  SmallVector<Vulkan::Extension> extensions;
  getExtensions(*this, extensions);
  return TargetEnvAttr::get(getVersion(*this), /*revision=*/0, extensions,
                            getVendor(*this), getDeviceType(*this),
                            spirv::TargetEnvAttr::kUnknownDeviceID,
                            getCapabilities(*this, context));
}

}  // namespace Vulkan
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
