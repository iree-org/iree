// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_INITIALIZEEMPTYTENSORSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

/// Returns a zero value attribute based on the `elementType`.
/// Returns failure, when the type is not handled.
static FailureOr<TypedAttr> getZero(OpBuilder &builder, Location loc,
                                    Type elementType) {
  if (auto intType = llvm::dyn_cast<IntegerType>(elementType)) {
    return cast<TypedAttr>(builder.getIntegerAttr(intType, 0));
  }
  if (auto floatType = llvm::dyn_cast<FloatType>(elementType)) {
    return cast<TypedAttr>(builder.getFloatAttr(floatType, 0.0));
  }
  return failure();
}

/// Returns true for the `tensor.empty` op has to be converted to a
/// `flow.tensor.*` op.
static bool shouldBeConvertedToFlowTensorOp(tensor::EmptyOp emptyTensorOp) {
  return !(llvm::all_of(emptyTensorOp->getUsers(),
                        llvm::IsaPred<linalg::LinalgOp, LinalgExt::LinalgExtOp,
                                      linalg::PackOp, linalg::UnPackOp>) ||
           emptyTensorOp->getParentOfType<Flow::DispatchWorkgroupsOp>());
}

namespace {

/// Converts an tensor.empty() op to `flow.tensor.splat` op.
struct RewriteTensorEmptyToSplat : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::EmptyOp emptyTensorOp,
                                PatternRewriter &rewriter) const override {
    if (!shouldBeConvertedToFlowTensorOp(emptyTensorOp)) {
      return failure();
    }
    RankedTensorType resultType = emptyTensorOp.getType();
    Type elementType = resultType.getElementType();
    Location loc = emptyTensorOp.getLoc();
    FailureOr<TypedAttr> zero = getZero(rewriter, loc, elementType);
    if (failed(zero)) {
      return rewriter.notifyMatchFailure(
          emptyTensorOp, "unable to get zero value for element type");
    }
    Value value =
        rewriter.create<arith::ConstantOp>(loc, elementType, zero.value());
    rewriter.replaceOpWithNewOp<TensorSplatOp>(emptyTensorOp, resultType, value,
                                               emptyTensorOp.getDynamicSizes());
    return success();
  }
};

/// Converts an tensor.empty() op to `flow.tensor.empty` op.
struct RewriteTensorEmptyToEmpty : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::EmptyOp emptyTensorOp,
                                PatternRewriter &rewriter) const override {
    if (!shouldBeConvertedToFlowTensorOp(emptyTensorOp)) {
      return failure();
    }
    RankedTensorType resultType = emptyTensorOp.getType();
    rewriter.replaceOpWithNewOp<TensorEmptyOp>(emptyTensorOp, resultType,
                                               emptyTensorOp.getDynamicSizes());
    return success();
  }
};

/// Pass to invoke the pattern.
struct InitializeEmptyTensorsPass
    : public IREE::Flow::impl::InitializeEmptyTensorsPassBase<
          InitializeEmptyTensorsPass> {
  using IREE::Flow::impl::InitializeEmptyTensorsPassBase<
      InitializeEmptyTensorsPass>::InitializeEmptyTensorsPassBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    if (zeroFill) {
      patterns.insert<RewriteTensorEmptyToSplat>(context);
    } else {
      patterns.insert<RewriteTensorEmptyToEmpty>(context);
    }
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
