// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPROMOTEMATMULOPERANDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
/// Helper to insert copy with derived thread config.
Value promoteValue(OpBuilder &builder, Location loc, Value v,
                   bool useDirectLoad) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  SmallVector<OpFoldResult> mixedSizes = tensor::getMixedSizes(builder, loc, v);

  Value empty = builder.create<tensor::EmptyOp>(loc, mixedSizes,
                                                tensorType.getElementType());
  auto copy = builder.create<linalg::CopyOp>(loc, v, empty);

  if (useDirectLoad) {
    setLoweringConfig(
        copy, IREE::GPU::UseGlobalLoadDMAAttr::get(builder.getContext()));
  } else {
    setLoweringConfig(
        copy, IREE::GPU::DerivedThreadConfigAttr::get(builder.getContext()));
  }
  return copy.getResult(0);
}

/// Helper to promote results. If the target value is consumed only by a
/// `tensor.extract_slice`, this will promote the result of the slice instead.
void promoteResult(OpBuilder &builder, Operation *op, Value valToMakeShared) {
  IRRewriter rewriter(builder);
  Location loc = op->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfterValue(valToMakeShared);
  tensor::ExtractSliceOp extractSliceOp;
  SetVector<Operation *> opsToReplaceUseIn;
  Value valueToReplace = valToMakeShared;
  for (auto user : valToMakeShared.getUsers()) {
    extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (extractSliceOp) {
      // If the result is consumed by an extract_slice then we expect there to
      // be exactly one extract slice that is then consumed.
      // TODO (nirvedhmeshram) : This is fairly special case. Instead we should
      // just promote results before doing padding which introduces the extract
      // slice.
      if (!valToMakeShared.hasOneUse())
        return;
      valueToReplace = extractSliceOp.getResult();
      for (auto user : extractSliceOp->getUsers()) {
        opsToReplaceUseIn.insert(user);
      }
      break;
    }
    opsToReplaceUseIn.insert(user);
  }
  auto tensorType = cast<RankedTensorType>(valToMakeShared.getType());
  if (!tensorType) {
    return;
  }
  SmallVector<Value> dynamicSizes;
  for (auto [idx, size] : llvm::enumerate(tensorType.getShape())) {
    if (ShapedType::isDynamic(size)) {
      dynamicSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, valToMakeShared, idx));
    }
  }
  Attribute addressSpace = gpu::AddressSpaceAttr::get(
      rewriter.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  auto alloc = rewriter.create<bufferization::AllocTensorOp>(loc, tensorType,
                                                             dynamicSizes);
  alloc.setMemorySpaceAttr(addressSpace);
  auto copy =
      rewriter.create<linalg::CopyOp>(loc, valToMakeShared, alloc.getResult());

  Value replacement = copy.getResult(0);
  // If in extract slice is present we make it consume the new copy.
  if (extractSliceOp) {
    extractSliceOp.getSourceMutable().assign(replacement);
    replacement = valueToReplace;
  }

  rewriter.setInsertionPointAfterValue(replacement);
  replacement =
      promoteValue(rewriter, loc, replacement, /*useDirectLoad=*/false);
  valueToReplace.replaceUsesWithIf(replacement, [&](OpOperand &use) {
    return opsToReplaceUseIn.contains(use.getOwner());
  });
}

void promoteOperand(OpBuilder &builder, Operation *op, unsigned index,
                    IREE::GPU::PromotionAttr promotionAttr) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dpsOp)
    return;
  // We use the convention that if we are passing an index beyond the inputs
  // then we promote the result of the corresponding dps init.
  if (index >= dpsOp.getNumDpsInputs()) {
    index -= dpsOp.getNumDpsInputs();
    assert(index < op->getNumResults() &&
           "trying to promote out of bound result index");
    // TODO(qedawkins): Move result promotion to attribute interface.
    return promoteResult(builder, op, op->getResult(index));
  }
  OpOperand &operand = op->getOpOperand(index);

  Value replacement = promotionAttr.promoteOperand(builder, operand);
  op->setOperand(index, replacement);
}

struct GPUPromoteMatmulOperandsPass final
    : impl::GPUPromoteMatmulOperandsPassBase<GPUPromoteMatmulOperandsPass> {
  using GPUPromoteMatmulOperandsPassBase::GPUPromoteMatmulOperandsPassBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    OpBuilder builder(funcOp);
    WalkResult walkResult = funcOp.walk([&](Operation *op) {
      auto loweringConfig =
          getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
      if (!loweringConfig) {
        return WalkResult::advance();
      }

      std::optional<SmallVector<int64_t>> promotedOperands =
          getPromotedOperandList(loweringConfig);
      if (!promotedOperands) {
        return WalkResult::advance();
      }

      std::optional<ArrayRef<Attribute>> maybePromotionTypes =
          getPromotionTypesList(loweringConfig);
      if (maybePromotionTypes &&
          maybePromotionTypes->size() != promotedOperands->size()) {
        op->emitOpError(
            "promoted operand and promotion types lists size mismatch");
        return WalkResult::interrupt();
      }

      Attribute derived =
          IREE::GPU::DerivedThreadConfigAttr::get(op->getContext());
      ArrayRef<Attribute> promotionTypes =
          maybePromotionTypes.value_or(ArrayRef<Attribute>());

      builder.setInsertionPoint(op);
      for (auto [operand, maybePromotionType] :
           llvm::zip_longest(promotedOperands.value(), promotionTypes)) {
        auto promotionType = dyn_cast<IREE::GPU::PromotionAttr>(
            maybePromotionType.value_or(derived));
        if (!promotionType) {
          op->emitOpError(
              "promotion types does not implement promotion attr interface");
          return WalkResult::interrupt();
        }
        promoteOperand(builder, op, operand.value(), promotionType);
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
