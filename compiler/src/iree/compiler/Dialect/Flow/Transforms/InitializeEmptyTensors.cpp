// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

/// Returns a zero value attribute based on the `elementType`.
/// Returns failure, when the type is not handled.
static FailureOr<Attribute> getZero(OpBuilder &builder, Location loc,
                                    Type elementType) {
  if (auto intType = elementType.dyn_cast<IntegerType>()) {
    return builder.getIntegerAttr(intType, 0);
  }
  if (auto floatType = elementType.dyn_cast<FloatType>()) {
    return builder.getFloatAttr(floatType, 0.0);
  }
  return failure();
}

namespace {

/// Converts an linalg.init_tensor op to `flow.tensor.splat` op.
struct RewriteInitTensorToSplat : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp emptyOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::all_of(emptyOp->getUsers(), [](Operation *user) -> bool {
          return isa<linalg::LinalgOp, LinalgExt::LinalgExtOp>(user);
        })) {
      return failure();
    }

    RankedTensorType resultType = emptyOp.getType();
    Type elementType = resultType.getElementType();
    Location loc = emptyOp.getLoc();
    FailureOr<Attribute> zero = getZero(rewriter, loc, elementType);
    if (failed(zero)) {
      return rewriter.notifyMatchFailure(
          emptyOp, "unable to get zero value for element type");
    }
    Value value =
        rewriter.create<arith::ConstantOp>(loc, elementType, zero.value());
    rewriter.replaceOpWithNewOp<TensorSplatOp>(emptyOp, resultType, value,
                                               emptyOp.getDynamicSizes());
    return success();
  }
};

/// Pass to invoke the pattern.
struct InitializeEmptyTensorsPass
    : public InitializeEmptyTensorsBase<InitializeEmptyTensorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, IREE::Flow::FlowDialect,
                    linalg::LinalgDialect>();
  }
  InitializeEmptyTensorsPass() = default;
  InitializeEmptyTensorsPass(const InitializeEmptyTensorsPass &) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<RewriteInitTensorToSplat>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createInitializeEmptyTensorsPass() {
  return std::make_unique<InitializeEmptyTensorsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
