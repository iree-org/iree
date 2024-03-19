// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-global-opt-fuse-winograd-padding"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

namespace {

/// Fuse iree_linalg_ext.winograd.input_transform with its producer
class FuseWinogradInputWithPadPattern final
    : public OpRewritePattern<IREE::LinalgExt::WinogradInputTransformOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(IREE::LinalgExt::WinogradInputTransformOp inputOp,
                  PatternRewriter &rewriter) const override {
    if (!inputOp->hasOneUse()) {
      return failure();
    }
    auto collapseShape = dyn_cast<tensor::CollapseShapeOp>(
        *inputOp->getResult(0).getUsers().begin());
    if (!collapseShape) {
      return failure();
    }
    if (!collapseShape->hasOneUse()) {
      return failure();
    }
    auto pad = dyn_cast<tensor::PadOp>(
        *collapseShape->getResult(0).getUsers().begin());
    if (!pad) {
      return failure();
    }

    SmallVector<Operation *> opsToFuse = {inputOp, collapseShape, pad};

    // Fail if matmul is already in a dispatch.
    for (Operation *op : opsToFuse) {
      if (!IREE::Flow::isNonNullAndOutsideDispatch(op)) {
        return failure();
      }
    }

    auto result = wrapConsecutiveOpsInDispatchRegion(rewriter, opsToFuse);
    if (failed(result)) {
      return failure();
    }

    return success();
  }
};

/// Fuse iree_linalg_ext.winograd.output_transform with its consumer
/// expand_shape and extract_slice ops.
class FuseWinogradOutputWithExtractSlicePattern final
    : public OpRewritePattern<IREE::LinalgExt::WinogradOutputTransformOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(IREE::LinalgExt::WinogradOutputTransformOp outputOp,
                  PatternRewriter &rewriter) const override {
    auto expandShape =
        outputOp.getInputs()[0].getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandShape) {
      return failure();
    }
    auto extractSlice =
        expandShape.getSrc().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractSlice) {
      return failure();
    }
    SmallVector<Operation *> opsToFuse = {extractSlice, expandShape, outputOp};

    // Fail if matmul is already in a dispatch.
    for (Operation *op : opsToFuse) {
      if (!IREE::Flow::isNonNullAndOutsideDispatch(op)) {
        return failure();
      }
    }

    auto result = wrapConsecutiveOpsInDispatchRegion(rewriter, opsToFuse);
    if (failed(result)) {
      return failure();
    }

    return success();
  }
};

struct FuseWinogradPaddingPass
    : public FuseWinogradPaddingBase<FuseWinogradPaddingPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect>();
  }

  void runOnOperation() override;
};

} // namespace

void FuseWinogradPaddingPass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  // patterns.insert<FuseWinogradInputWithPadPattern>(context);
  patterns.insert<FuseWinogradOutputWithExtractSlicePattern>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFuseWinogradPaddingPass() {
  return std::make_unique<FuseWinogradPaddingPass>();
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
