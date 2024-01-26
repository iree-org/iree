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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-codegen-erase-hal-descriptor-type"

namespace mlir::iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// Conversion Passes
//===----------------------------------------------------------------------===//

struct EraseHALDescriptorTypeFromMemRefPass final
    : public EraseHALDescriptorTypeFromMemRefBase<
          EraseHALDescriptorTypeFromMemRefPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    auto replaceMemorySpace =
        [](Attribute memorySpace) -> std::optional<Attribute> {
      if (isa<IREE::HAL::DescriptorTypeAttr>(memorySpace)) {
        return Attribute();
      }
      return std::nullopt;
    };

    if (failed(replaceMemRefMemorySpace(op, replaceMemorySpace)))
      return signalPassFailure();
  }
};

struct ConvertHALDescriptorTypeToGPUAddressSpacePass final
    : public ConvertHALDescriptorTypeToGPUAddressSpaceBase<
          ConvertHALDescriptorTypeToGPUAddressSpacePass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    auto replaceMemorySpace =
        [](Attribute memorySpace) -> std::optional<Attribute> {
      if (isa<IREE::HAL::DescriptorTypeAttr>(memorySpace)) {
        return gpu::AddressSpaceAttr::get(memorySpace.getContext(),
                                          gpu::AddressSpace::Global);
      }
      return std::nullopt;
    };

    if (failed(replaceMemRefMemorySpace(op, replaceMemorySpace)))
      return signalPassFailure();
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
