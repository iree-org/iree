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

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

struct MemRefTypeConverter final : public TypeConverter {
  MemRefTypeConverter() {
    // Pass through for all other types.
    addConversion([](Type type) { return type; });

    addConversion([](BaseMemRefType memRefType) -> std::optional<Type> {
      // Expect #hal.descriptor_type memory spaces.
      Attribute spaceAttr = memRefType.getMemorySpace();
      if (!spaceAttr)
        return std::nullopt;
      auto dtAttr = llvm::dyn_cast<IREE::HAL::DescriptorTypeAttr>(spaceAttr);
      if (!dtAttr)
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
  }
};

struct GPUMemRefTypeConverter final : public TypeConverter {
  GPUMemRefTypeConverter() {
    // Pass through for all other types.
    addConversion([](Type type) { return type; });

    addConversion([](BaseMemRefType memRefType) -> std::optional<Type> {
      // Expect #hal.descriptor_type memory spaces.
      Attribute spaceAttr = memRefType.getMemorySpace();
      if (!spaceAttr)
        return std::nullopt;
      auto dtAttr = llvm::dyn_cast<IREE::HAL::DescriptorTypeAttr>(spaceAttr);
      if (!dtAttr)
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
  }
};

//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

/// Returns true if the given `type` is considered as legal.
static bool isLegalType(Type type) {
  if (auto memRefType = llvm::dyn_cast<BaseMemRefType>(type)) {
    Attribute spaceAttr = memRefType.getMemorySpace();
    return !spaceAttr || !llvm::isa<IREE::HAL::DescriptorTypeAttr>(spaceAttr);
  }
  return true;
}

/// Returns true if the given `op` is considered as legal.
static bool isLegalOp(Operation *op) {
  if (!llvm::all_of(op->getOperandTypes(), isLegalType) ||
      !llvm::all_of(op->getResultTypes(), isLegalType)) {
    return false;
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      if (!llvm::all_of(block.getArgumentTypes(), isLegalType)) {
        return false;
      }
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Conversion Pattern
//===----------------------------------------------------------------------===//

struct EraseMemorySpacePattern final : public ConversionPattern {
  EraseMemorySpacePattern(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

LogicalResult EraseMemorySpacePattern::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  const TypeConverter &typeConverter = *getTypeConverter();
  llvm::SmallVector<Type> newResults;
  if (failed(typeConverter.convertTypes(op->getResultTypes(), newResults))) {
    op->emitError("Can't convert results");
  }

  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       newResults, op->getAttrs(), op->getSuccessors());

  for (Region &region : op->getRegions()) {
    Region *newRegion = state.addRegion();
    rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
    if (failed(rewriter.convertRegionTypes(newRegion, typeConverter))) {
      return op->emitError("Cant'convert region types");
    }
  }

  Operation *newOp = rewriter.create(state);
  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

static LogicalResult eraseHALDescriptorTypeFromMemRef(Operation *op) {
  MLIRContext *context = op->getContext();
  ConversionTarget target(*context);
  target.markUnknownOpDynamicallyLegal(isLegalOp);

  MemRefTypeConverter typeConverter;
  RewritePatternSet patterns(context);
  patterns.add<EraseMemorySpacePattern>(context, typeConverter);

  return applyFullConversion(op, target, std::move(patterns));
}

static LogicalResult convertHALDescriptorTypeToGPUAddressSpace(Operation *op) {
  MLIRContext *context = op->getContext();
  ConversionTarget target(*context);
  target.markUnknownOpDynamicallyLegal(isLegalOp);

  GPUMemRefTypeConverter typeConverter;
  RewritePatternSet patterns(context);
  patterns.add<EraseMemorySpacePattern>(context, typeConverter);

  return applyFullConversion(op, target, std::move(patterns));
}

//===----------------------------------------------------------------------===//
// Conversion Passes
//===----------------------------------------------------------------------===//

struct EraseHALDescriptorTypeFromMemRefPass final
    : public EraseHALDescriptorTypeFromMemRefBase<
          EraseHALDescriptorTypeFromMemRefPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    if (failed(eraseHALDescriptorTypeFromMemRef(op)))
      return signalPassFailure();
  }
};

struct ConvertHALDescriptorTypeToGPUAddressSpacePass final
    : public ConvertHALDescriptorTypeToGPUAddressSpaceBase<
          ConvertHALDescriptorTypeToGPUAddressSpacePass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    if (failed(convertHALDescriptorTypeToGPUAddressSpace(op)))
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
