// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-spirv-map-memref-storage-class"

namespace mlir::iree_compiler {

namespace {

std::optional<spirv::StorageClass>
mapHALDescriptorTypeForVulkan(Attribute attr) {
  if (auto dtAttr =
          llvm::dyn_cast_if_present<IREE::HAL::DescriptorTypeAttr>(attr)) {
    switch (dtAttr.getValue()) {
    case IREE::HAL::DescriptorType::UniformBuffer:
      return spirv::StorageClass::Uniform;
    case IREE::HAL::DescriptorType::StorageBuffer:
      return spirv::StorageClass::StorageBuffer;
    default:
      return std::nullopt;
    }
  }
  if (auto gpuAttr = llvm::dyn_cast_if_present<gpu::AddressSpaceAttr>(attr)) {
    switch (gpuAttr.getValue()) {
    case gpu::AddressSpace::Workgroup:
      return spirv::StorageClass::Workgroup;
    default:
      return std::nullopt;
    }
  };
  return spirv::mapMemorySpaceToVulkanStorageClass(attr);
}

std::optional<spirv::StorageClass>
mapHALDescriptorTypeForOpenCL(Attribute attr) {
  if (auto dtAttr =
          llvm::dyn_cast_if_present<IREE::HAL::DescriptorTypeAttr>(attr)) {
    switch (dtAttr.getValue()) {
    case IREE::HAL::DescriptorType::UniformBuffer:
      return spirv::StorageClass::Uniform;
    case IREE::HAL::DescriptorType::StorageBuffer:
      return spirv::StorageClass::CrossWorkgroup;
    }
  }
  if (auto gpuAttr = llvm::dyn_cast_if_present<gpu::AddressSpaceAttr>(attr)) {
    switch (gpuAttr.getValue()) {
    case gpu::AddressSpace::Workgroup:
      return spirv::StorageClass::Workgroup;
    default:
      return std::nullopt;
    }
  };
  return spirv::mapMemorySpaceToOpenCLStorageClass(attr);
}

struct SPIRVMapMemRefStorageClassPass final
    : public SPIRVMapMemRefStorageClassBase<SPIRVMapMemRefStorageClassPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    spirv::MemorySpaceToStorageClassMap memorySpaceMap;

    if (spirv::TargetEnvAttr attr = getSPIRVTargetEnvAttr(op)) {
      spirv::TargetEnv targetEnv(attr);
      if (targetEnv.allows(spirv::Capability::Shader)) {
        memorySpaceMap = mapHALDescriptorTypeForVulkan;
      } else if (targetEnv.allows(spirv::Capability::Kernel)) {
        memorySpaceMap = mapHALDescriptorTypeForOpenCL;
      }
    }
    if (!memorySpaceMap) {
      op->emitError("missing storage class map for unknown client API");
      return signalPassFailure();
    }

    auto target = spirv::getMemorySpaceToStorageClassTarget(*context);
    spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);

    RewritePatternSet patterns(context);
    spirv::populateMemorySpaceToStorageClassPatterns(converter, patterns);

    if (failed(applyFullConversion(op, *target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createSPIRVMapMemRefStorageClassPass() {
  return std::make_unique<SPIRVMapMemRefStorageClassPass>();
}

} // namespace mlir::iree_compiler
