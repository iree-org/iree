// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CANONICALIZERPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

/// Folds a chain of `tensor.pad` ops with the same constant padding value.
///
/// Example:
///
/// ```mlir
///   %1 = tensor.pad %0 low[0, 1] high[0, 2] {
///       tensor.yield %val
///     } : tensor<1x2xf32> to tensor<2x5xf32>
///   %res = tensor.pad %1 low[0, 2] high[3, 0] {
///       tensor.yield %val
///     } : tensor<1x5xf32> to tensor<5x7xf32>
/// ```
///
/// folds into:
///
/// ```mlir
///   %res = tensor.pad %0 low[0, 3] high[3, 2] {
///       tensor.yield %val
///     } : tensor<1x2xf32> to tensor<5x7xf32>
/// ```
///
/// NOTE: This wasn't sent upstream as a canonicalization due to the use of
/// the Affine dialect.
struct FoldConsecutiveConstantPadding : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    if (padOp.getNofold()) {
      return failure();
    }
    auto producerPad = padOp.getSource().getDefiningOp<tensor::PadOp>();
    if (!producerPad || producerPad.getNofold()) {
      return rewriter.notifyMatchFailure(
          padOp, "producer is not a foldable tensor.pad op");
    }

    // Fail if the tensor::PadOps padding values do not match.
    Value consumerPadValue = padOp.getConstantPaddingValue();
    Value producerPadValue = producerPad.getConstantPaddingValue();
    if (!consumerPadValue || !producerPadValue ||
        consumerPadValue != producerPadValue) {
      return rewriter.notifyMatchFailure(
          padOp, "cannot fold PadOps with different padding values");
    }

    Location loc = padOp.getLoc();
    AffineExpr d0, d1;
    bindDims(rewriter.getContext(), d0, d1);

    // Combine the low/high paddings of the two tensor::PadOps.
    auto addPaddings = [&](ArrayRef<OpFoldResult> consumerPaddings,
                           ArrayRef<OpFoldResult> producerPaddings) {
      SmallVector<OpFoldResult> sumPaddings;
      for (auto [consumerIndex, producerIndex] :
           llvm::zip_equal(consumerPaddings, producerPaddings)) {
        sumPaddings.push_back(affine::makeComposedFoldedAffineApply(
            rewriter, loc, d0 + d1, {consumerIndex, producerIndex}));
      }
      return sumPaddings;
    };

    SmallVector<OpFoldResult> newHighPad =
        addPaddings(padOp.getMixedHighPad(), producerPad.getMixedHighPad());
    SmallVector<OpFoldResult> newLowPad =
        addPaddings(padOp.getMixedLowPad(), producerPad.getMixedLowPad());

    auto newPadOp = rewriter.create<tensor::PadOp>(
        padOp.getLoc(), padOp.getResultType(), producerPad.getSource(),
        newLowPad, newHighPad, padOp.getNofold(),
        getPrunedAttributeList(padOp, tensor::PadOp::getAttributeNames()));
    rewriter.inlineRegionBefore(padOp.getRegion(), newPadOp.getRegion(),
                                newPadOp.getRegion().begin());
    rewriter.replaceOp(padOp, newPadOp.getResult());
    return success();
  }
};

/// Canonicalize operations in nested regions.
struct CanonicalizerPass
    : public impl::CanonicalizerPassBase<CanonicalizerPass> {
  using IREE::Flow::impl::CanonicalizerPassBase<
      CanonicalizerPass>::CanonicalizerPassBase;
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Inherit the same config defaults from the upstream canonicalizer pass.
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Normal;

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    // Pull in some borderline/downstream canonicalizations for the Flow
    // compilation phase.
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(owningPatterns);
    owningPatterns.add<FoldConsecutiveConstantPadding>(context);

    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }
  void runOnOperation() override {
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    LogicalResult didConverge =
        applyPatternsAndFoldGreedily(getOperation(), *patterns, config);
    if (this->testConvergence && failed(didConverge)) {
      getOperation()->emitError("Canonicalizer failed to converge");
      return signalPassFailure();
    }
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
