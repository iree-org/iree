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

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-codegen-erase-hal-descriptor-type"

namespace mlir {
namespace iree_compiler {
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
      if (!spaceAttr) return std::nullopt;
      auto dtAttr = spaceAttr.dyn_cast<IREE::HAL::DescriptorTypeAttr>();
      if (!dtAttr) return std::nullopt;

      // Erase the #hal.descriptor_type memory space.
      if (auto rankedType = memRefType.dyn_cast<MemRefType>()) {
        return MemRefType::get(memRefType.getShape(),
                               memRefType.getElementType(),
                               rankedType.getLayout());
      }
      return UnrankedMemRefType::get(memRefType.getElementType(),
                                     /*memorySpace=*/0);
    });
  }
};

//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

/// Returns true if the given `type` is considered as legal.
static bool isLegalType(Type type) {
  if (auto memRefType = type.dyn_cast<BaseMemRefType>()) {
    Attribute spaceAttr = memRefType.getMemorySpace();
    return !spaceAttr || !spaceAttr.isa<IREE::HAL::DescriptorTypeAttr>();
  }
  return true;
}

/// Returns true if the given `op` is considered as legal.
static bool isLegalOp(Operation *op) {
  return llvm::all_of(op->getOperandTypes(), isLegalType) &&
         llvm::all_of(op->getResultTypes(), isLegalType);
}

//===----------------------------------------------------------------------===//
// Conversion Pattern
//===----------------------------------------------------------------------===//

struct EraseMemorySpacePattern final : public ConversionPattern {
  EraseMemorySpacePattern(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

LogicalResult EraseMemorySpacePattern::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  llvm::SmallVector<Type, 4> newResults;
  (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);

  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       newResults, op->getAttrs(), op->getSuccessors());

  for (Region &region : op->getRegions()) {
    Region *newRegion = state.addRegion();
    rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
    TypeConverter::SignatureConversion result(newRegion->getNumArguments());
    (void)getTypeConverter()->convertSignatureArgs(
        newRegion->getArgumentTypes(), result);
    rewriter.applySignatureConversion(newRegion, result);
  }

  Operation *newOp = rewriter.create(state);
  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

struct EraseHALDescriptorTypeFromMemRefPass final
    : public EraseHALDescriptorTypeFromMemRefBase<
          EraseHALDescriptorTypeFromMemRefPass> {
  void runOnOperation() override {
    func::FuncOp op = getOperation();
    if (failed(eraseHALDescriptorTypeFromMemRef(op)))
      return signalPassFailure();
  }
};

}  // namespace

LogicalResult eraseHALDescriptorTypeFromMemRef(func::FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  ConversionTarget target(*context);
  target.markUnknownOpDynamicallyLegal(isLegalOp);

  MemRefTypeConverter typeConverter;
  RewritePatternSet patterns(context);
  patterns.add<EraseMemorySpacePattern>(context, typeConverter);

  return applyFullConversion(funcOp, target, std::move(patterns));
}

std::unique_ptr<OperationPass<func::FuncOp>>
createEraseHALDescriptorTypeFromMemRefPass() {
  return std::make_unique<EraseHALDescriptorTypeFromMemRefPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
