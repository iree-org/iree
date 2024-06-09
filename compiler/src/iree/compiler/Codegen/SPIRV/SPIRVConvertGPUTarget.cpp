// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

namespace {

using IREE::GPU::ComputeBitwidths;
using IREE::GPU::DotProductOps;
using IREE::GPU::StorageBitwidths;
using IREE::GPU::SubgroupOps;

using spirv::Capability;
using spirv::ClientAPI;
using spirv::Extension;
using spirv::Vendor;
using spirv::Version;

//===----------------------------------------------------------------------===//
// Freeform features
//===----------------------------------------------------------------------===//

// Scans the given |features| list and pushes SPIR-V version specification of
// 'spirv:v1.x' format into |caps|.
std::optional<Version> deduceVersion(ArrayRef<StringRef> features) {
  for (StringRef feature : features) {
    if (feature.consume_front("spirv:v1.")) {
      return llvm::StringSwitch<std::optional<Version>>(feature)
          .Case("6", Version::V_1_6)
          .Case("5", Version::V_1_5)
          .Case("4", Version::V_1_4)
          .Case("3", Version::V_1_3)
          .Case("2", Version::V_1_2)
          .Case("1", Version::V_1_1)
          .Default(std::nullopt);
    }
  }
  return std::nullopt;
}

// Scans the given |features| list and pushes capability specification with
// 'cap:' prefix into |caps|.
std::optional<Version> processCapabilities(ArrayRef<StringRef> features,
                                           SetVector<Capability> &caps) {
  for (StringRef feature : features) {
    if (feature.consume_front("cap:")) {
      if (std::optional<Capability> cap = spirv::symbolizeCapability(feature))
        caps.insert(*cap);
    }
  }
  return std::nullopt;
}

// Scans the given |features| list and pushes extension specification with
// 'ext:' prefix into |exts|.
std::optional<Version> processExtensions(ArrayRef<StringRef> features,
                                         SetVector<Extension> &exts) {
  for (StringRef feature : features) {
    if (feature.consume_front("ext:")) {
      if (std::optional<Extension> ext = spirv::symbolizeExtension(feature))
        exts.insert(*ext);
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Client API and vendor
//===----------------------------------------------------------------------===//

ClientAPI deduceClientAPI(StringRef backend) {
  return llvm::StringSwitch<ClientAPI>(backend)
      .Case("vulkan", ClientAPI::Vulkan)
      .Case("metal", ClientAPI::Metal)
      .Case("webgpu", ClientAPI::WebGPU)
      .Case("opencl", ClientAPI::OpenCL)
      .Default(ClientAPI::Unknown);
}

Vendor deduceVendor(StringRef arch) {
  if (arch.starts_with("gfx") || arch.starts_with("rdna"))
    return Vendor::AMD;
  if (arch.starts_with("mali"))
    return Vendor::ARM;
  if (arch.starts_with("sm_"))
    return Vendor::NVIDIA;
  if (arch.starts_with("adreno"))
    return Vendor::Qualcomm;
  return Vendor::Unknown;
}

//===----------------------------------------------------------------------===//
// Workgroup-processor features and limits
//===----------------------------------------------------------------------===//

void addComputeFeatures(ComputeBitwidths compute, SetVector<Capability> &caps,
                        SetVector<Extension> &exts) {
  if (bitEnumContainsAny(compute, ComputeBitwidths::FP64))
    caps.insert(Capability::Float64);
  // FP32 does not need special capabilities or extensions.
  if (bitEnumContainsAny(compute, ComputeBitwidths::FP16))
    caps.insert(Capability::Float16);

  if (bitEnumContainsAny(compute, ComputeBitwidths::Int64))
    caps.insert(Capability::Int64);
  // Int32 does not need special capabilities or extensions.
  if (bitEnumContainsAny(compute, ComputeBitwidths::Int16))
    caps.insert(Capability::Int16);
  if (bitEnumContainsAny(compute, ComputeBitwidths::Int8))
    caps.insert(Capability::Int8);
}

void addStorageFeatures(StorageBitwidths storage, SetVector<Capability> &caps,
                        SetVector<Extension> &exts) {
  // 64bit does not need special capabilities or extensions.
  // 32bit does not need special capabilities or extensions.
  if (bitEnumContainsAny(storage, StorageBitwidths::B16)) {
    caps.insert(Capability::StorageBuffer16BitAccess);
    caps.insert(Capability::StorageUniform16);
    caps.insert(Capability::StoragePushConstant16);
    exts.insert(Extension::SPV_KHR_16bit_storage);
  }
  if (bitEnumContainsAny(storage, StorageBitwidths::B8)) {
    caps.insert(Capability::StorageBuffer8BitAccess);
    caps.insert(Capability::UniformAndStorageBuffer8BitAccess);
    caps.insert(Capability::StoragePushConstant8);
    exts.insert(Extension::SPV_KHR_8bit_storage);
  }
}

void addSubgroupFeatures(SubgroupOps subgroup, SetVector<Capability> &caps,
                         SetVector<Extension> &exts) {
  if (bitEnumContainsAny(subgroup, SubgroupOps::Shuffle)) {
    caps.insert(Capability::GroupNonUniformShuffle);
    caps.insert(Capability::GroupNonUniformShuffleRelative);
  }
  if (bitEnumContainsAny(subgroup, SubgroupOps::Arithmetic)) {
    caps.insert(Capability::GroupNonUniformArithmetic);
  }
}

void addDotProductFeatures(ComputeBitwidths compute, DotProductOps dotProduct,
                           SetVector<Capability> &caps,
                           SetVector<Extension> &exts) {
  if (bitEnumContainsAny(dotProduct, DotProductOps::DP4xI8ToI32)) {
    caps.insert(Capability::DotProduct);
    caps.insert(Capability::DotProductInput4x8BitPacked);
    if (bitEnumContainsAny(compute, ComputeBitwidths::Int8)) {
      caps.insert(Capability::DotProductInput4x8Bit);
    }
    exts.insert(Extension::SPV_KHR_integer_dot_product);
  }
}

void addMatrixFeatures(IREE::GPU::MMAOpsArrayAttr mmaOps,
                       SetVector<Capability> &caps, SetVector<Extension> &exts,
                       SetVector<Attribute> &coopMatAttrs) {
  if (!mmaOps.empty()) {
    caps.insert(Capability::CooperativeMatrixKHR);
    exts.insert(Extension::SPV_KHR_cooperative_matrix);
  }
}

spirv::ResourceLimitsAttr convertLimits(IREE::GPU::TargetWgpAttr wgp) {
  MLIRContext *context = wgp.getContext();
  Builder b(context);

  SmallVector<Attribute, 4> coopMatAttrs;
  for (IREE::GPU::MMAAttr mmaOp : wgp.getMma()) {
    auto [mSize, nSize, kSize] = mmaOp.getMNKShape();
    auto [aType, bType, cType] = mmaOp.getABCElementTypes();
    coopMatAttrs.push_back(spirv::CooperativeMatrixPropertiesKHRAttr::get(
        context, mSize, nSize, kSize, aType, bType, cType, cType,
        false /*saturatingAccumulation*/,
        spirv::ScopeAttr::get(context, spirv::Scope::Subgroup)));
  }

  ArrayRef<int> subgroupSizes = wgp.getSubgroupSizeChoices().asArrayRef();

  return spirv::ResourceLimitsAttr::get(
      context, wgp.getMaxWorkgroupMemoryBytes(),
      wgp.getMaxThreadCountPerWorkgroup(),
      b.getI32ArrayAttr(wgp.getMaxWorkgroupSizes().asArrayRef()),
      subgroupSizes.front(), *llvm::min_element(subgroupSizes),
      *llvm::max_element(subgroupSizes), ArrayAttr::get(context, coopMatAttrs),
      ArrayAttr{});
}

//===----------------------------------------------------------------------===//
// Target specification conversion
//===----------------------------------------------------------------------===//

FailureOr<spirv::TargetEnvAttr>
convertGPUTarget(IREE::HAL::ExecutableVariantOp variant) {
  IREE::HAL::ExecutableTargetAttr target = variant.getTarget();
  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(target);

  SmallVector<StringRef> features;
  llvm::SplitString(gpuTarget.getFeatures(), features, ",");

  SetVector<Capability> caps;
  SetVector<Extension> exts;
  SetVector<Attribute> coopMatAttrs;

  std::optional<Version> version = deduceVersion(features);
  if (!version) {
    return variant.emitError("cannot deduce spirv version from target "
                             "features; need to specify 'spirv1.x'");
  }
  processCapabilities(features, caps);
  processExtensions(features, exts);

  IREE::GPU::TargetWgpAttr wgp = gpuTarget.getWgp();
  ComputeBitwidths compute = wgp.getCompute().getValue();
  addComputeFeatures(compute, caps, exts);
  addStorageFeatures(wgp.getStorage().getValue(), caps, exts);
  addSubgroupFeatures(wgp.getSubgroup().getValue(), caps, exts);
  addDotProductFeatures(compute, wgp.getDot().getValue(), caps, exts);
  addMatrixFeatures(wgp.getMma(), caps, exts, coopMatAttrs);

  auto triple = spirv::VerCapExtAttr::get(
      *version, caps.getArrayRef(), exts.getArrayRef(), variant.getContext());
  return spirv::TargetEnvAttr::get(
      triple, convertLimits(wgp), deduceClientAPI(target.getBackend()),
      deduceVendor(gpuTarget.getArch()), spirv::DeviceType::Unknown,
      spirv::TargetEnvAttr::kUnknownDeviceID);
}

struct SPIRVConvertGPUTargetPass final
    : SPIRVConvertGPUTargetBase<SPIRVConvertGPUTargetPass> {
  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variant = getOperation();
    IREE::HAL::ExecutableTargetAttr target = variant.getTarget();

    FailureOr<spirv::TargetEnvAttr> spirvTarget = convertGPUTarget(variant);
    if (failed(spirvTarget))
      return signalPassFailure();

    Builder b(&getContext());
    auto attrs = llvm::to_vector(target.getConfiguration().getValue());
    attrs.emplace_back(b.getStringAttr(spirv::getTargetEnvAttrName()),
                       *spirvTarget);
    auto configAttr = b.getDictionaryAttr(attrs);

    auto halTarget = IREE::HAL::ExecutableTargetAttr::get(
        target.getContext(), target.getBackend(), target.getFormat(),
        configAttr);
    variant.setTargetAttr(halTarget);
  }
};

} // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVConvertGPUTargetPass() {
  return std::make_unique<SPIRVConvertGPUTargetPass>();
}

} // namespace mlir::iree_compiler
