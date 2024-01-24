// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

/// Check if a `t` is a `tensor` with zero extents.
static std::optional<RankedTensorType> isZeroExtent(Type t) {
  auto operandType = dyn_cast<RankedTensorType>(t);
  if (operandType &&
      llvm::any_of(operandType.getShape(), [](int64_t s) { return s == 0; })) {
    return operandType;
  }
  return std::nullopt;
}

/// Replace operands of the operation that have zero-extent tensors with
/// a `tensor.empty` op of the same type. This breaks dependencies between
/// different operations which can be handled subsequently.
struct ReplaceZeroExtentOperands : public RewritePattern {
  ReplaceZeroExtentOperands(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (isa<tensor::EmptyOp, tensor::DimOp>(op)) {
      return failure();
    }
    Location loc = op->getLoc();
    bool didUpdate = false;
    for (OpOperand &operand : op->getOpOperands()) {
      auto operandType = isZeroExtent(operand.get().getType());
      if (!operandType) {
        continue;
      }
      if (operand.get().getDefiningOp<tensor::EmptyOp>()) {
        continue;
      }
      Operation *owner = operand.getOwner();
      int operandNum = operand.getOperandNumber();
      auto shape = tensor::getMixedSizes(rewriter, loc, operand.get());
      auto emptyTensorOp = rewriter.create<tensor::EmptyOp>(
          loc, shape, operandType->getElementType());
      rewriter.modifyOpInPlace(
          owner, [&]() { owner->setOperand(operandNum, emptyTensorOp); });
      didUpdate = true;
    }
    return success(didUpdate);
  }
};

/// Forward the destination of a `tensor.insert_slice` to its uses
/// if the source is zero-extent.
struct FoldZeroExtentInserts : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!isZeroExtent(sliceOp.getSource().getType())) {
      return failure();
    }
    rewriter.replaceOp(sliceOp, sliceOp.getDest());
    return success();
  }
};

namespace {

struct RemoveZeroExtentTensorsPass
    : RemoveZeroExtentTensorsBase<RemoveZeroExtentTensorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

} // namespace

void RemoveZeroExtentTensorsPass::runOnOperation() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();
  SmallVector<Operation *> opWithZeroExtentTensorOperands;
  SmallVector<tensor::InsertSliceOp> insertSliceOps;

  RewritePatternSet patterns(context);
  patterns.insert<FoldZeroExtentInserts, ReplaceZeroExtentOperands>(context);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    funcOp->emitOpError("failed to run canonicalizations (proxy for DCE)");
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createRemoveZeroExtentTensorsPass() {
  return std::make_unique<RemoveZeroExtentTensorsPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
