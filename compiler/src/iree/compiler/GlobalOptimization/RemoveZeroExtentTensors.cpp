// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_REMOVEZEROEXTENTTENSORSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

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
struct ReplaceZeroExtentOperands : RewritePattern {
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
      auto emptyTensorOp = tensor::EmptyOp::create(
          rewriter, loc, shape, operandType->getElementType());
      rewriter.modifyOpInPlace(
          owner, [&]() { owner->setOperand(operandNum, emptyTensorOp); });
      didUpdate = true;
    }
    return success(didUpdate);
  }
};

/// Forward the destination of a `tensor.insert_slice` to its uses
/// if the source is zero-extent.
struct FoldZeroExtentInserts : OpRewritePattern<tensor::InsertSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!isZeroExtent(sliceOp.getSource().getType())) {
      return failure();
    }
    rewriter.replaceOp(sliceOp, sliceOp.getDest());
    return success();
  }
};

struct FoldZeroExtentReduction : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Must be a single-result reduction on tensors.
    if (genericOp.getNumReductionLoops() == 0) {
      return failure();
    }
    if (genericOp.getNumDpsInits() != 1) {
      return failure();
    }
    if (!genericOp.hasPureTensorSemantics()) {
      return failure();
    }

    auto resultType = dyn_cast<RankedTensorType>(genericOp.getResultTypes()[0]);
    if (!resultType) {
      return failure();
    }

    // Output must be non-empty; the zero-sized dim is on the (reduced) input.
    if (isZeroExtent(resultType)) {
      return failure();
    }

    // Require at least one zero-extent input operand.
    bool hasZeroExtentInput = false;
    for (Value in : genericOp.getDpsInputs()) {
      if (isZeroExtent(in.getType())) {
        hasZeroExtentInput = true;
        break;
      }
    }
    if (!hasZeroExtentInput) {
      return failure();
    }

    // With a zero-sized reduction dimension, no input element is combined,
    // so the result equals the init operand. Forward it directly; this
    // preserves the semantics of any initial accumulator value (identity or
    // not) without materializing a new fill.
    Value init = genericOp.getDpsInitOperand(0)->get();
    rewriter.replaceOp(genericOp, init);
    return success();
  }
};

namespace {

struct RemoveZeroExtentTensorsPass
    : impl::RemoveZeroExtentTensorsPassBase<RemoveZeroExtentTensorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override;
};

} // namespace

void RemoveZeroExtentTensorsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  patterns.insert<FoldZeroExtentReduction, FoldZeroExtentInserts,
                  ReplaceZeroExtentOperands>(context);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    funcOp->emitOpError("failed to run canonicalizations (proxy for DCE)");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::GlobalOptimization
