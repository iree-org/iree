// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- EraseHALDescriptorTypeFromMemRef -----------------------------------===//
// Patterns and pass to erase #hal.descriptor_type from MemRef memory space.
// The purpose of these utilities is just to make transitioning easier--right
// now converting to LLVM still has lots of underlying assumption over numeric
// memory spaces, and some pattern does not support memory space other than 0.
//===----------------------------------------------------------------------===//

#include <memory>

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-codegen-erase-hal-descriptor-type"

namespace mlir::iree_compiler {

namespace {

/// Returns true if the given `type` is considered as legal.
static bool isLegalType(Type type) {
  if (auto memRefType = llvm::dyn_cast<BaseMemRefType>(type)) {
    Attribute spaceAttr = memRefType.getMemorySpace();
    return !spaceAttr || !llvm::isa<IREE::HAL::DescriptorTypeAttr>(spaceAttr);
  }
  return true;
}

struct EraseHALDescriptorTypeFromMemRefPass final
    : public EraseHALDescriptorTypeFromMemRefBase<
          EraseHALDescriptorTypeFromMemRefPass> {
  void runOnOperation() override {
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [](BaseMemRefType memRefType) -> std::optional<BaseMemRefType> {
          if (isLegalType(memRefType))
            return std::nullopt;

          // Erase the #hal.descriptor_type memory space.
          if (auto rankedType = llvm::dyn_cast<MemRefType>(memRefType)) {
            return MemRefType::get(memRefType.getShape(),
                                   memRefType.getElementType(),
                                   rankedType.getLayout());
          }
          return UnrankedMemRefType::get(memRefType.getElementType(),
                                         /*memorySpace=*/0);
        });

    Operation *op = getOperation();

    replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/false,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
  }
};

struct ConvertHALDescriptorTypeToGPUAddressSpacePass final
    : public ConvertHALDescriptorTypeToGPUAddressSpaceBase<
          ConvertHALDescriptorTypeToGPUAddressSpacePass> {
  void runOnOperation() override {
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [](BaseMemRefType memRefType) -> std::optional<BaseMemRefType> {
          if (isLegalType(memRefType))
            return std::nullopt;

          Attribute globalSpace = gpu::AddressSpaceAttr::get(
              memRefType.getContext(), gpu::AddressSpace::Global);

          // Erase the #hal.descriptor_type memory space.
          if (auto rankedType = llvm::dyn_cast<MemRefType>(memRefType)) {
            return MemRefType::get(memRefType.getShape(),
                                   memRefType.getElementType(),
                                   rankedType.getLayout(), globalSpace);
          }
          return UnrankedMemRefType::get(memRefType.getElementType(),
                                         /*memorySpace=*/globalSpace);
        });

    Operation *op = getOperation();

    replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/false,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
  }
};

} // namespace

std::unique_ptr<Pass> createEraseHALDescriptorTypeFromMemRefPass() {
  return std::make_unique<EraseHALDescriptorTypeFromMemRefPass>();
}

std::unique_ptr<Pass> createConvertHALDescriptorTypeToGPUAddressSpacePass() {
  return std::make_unique<ConvertHALDescriptorTypeToGPUAddressSpacePass>();
}

} // namespace mlir::iree_compiler
