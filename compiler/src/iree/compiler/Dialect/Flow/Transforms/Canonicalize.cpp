// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CANONICALIZEPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

static std::optional<SmallVector<OpFoldResult>> getDefiningMixedSizes(Value v) {
  if (auto empty = v.getDefiningOp<tensor::EmptyOp>()) {
    return empty.getMixedSizes();
  } else if (auto extract = v.getDefiningOp<tensor::ExtractSliceOp>()) {
    // TODO: Support rank reducing cases.
    if (extract.getSourceType().getRank() !=
        extract.getResultType().getRank()) {
      return {};
    }
    return extract.getMixedSizes();
  }
  return {};
}

struct FoldFullInsertSlice : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!insertSliceOp.hasUnitStride() || !insertSliceOp.hasZeroOffset()) {
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "non-unit stride or non-zero offset.");
    }

    RankedTensorType sourceType = insertSliceOp.getSourceType();
    RankedTensorType resultType = insertSliceOp.getResultType();
    if (sourceType != resultType) {
      return rewriter.notifyMatchFailure(
          insertSliceOp,
          "unimplemented: Cast-like or reshape-like insert ops.");
    }

    std::optional<SmallVector<OpFoldResult>> mixedSizes =
        getDefiningMixedSizes(insertSliceOp.getDest());
    if (!mixedSizes) {
      return rewriter.notifyMatchFailure(
          insertSliceOp, "Could not find producer with list of tensor sizes.");
    }

    for (auto [insertSize, destSize] :
         llvm::zip_equal(insertSliceOp.getMixedSizes(), mixedSizes.value())) {
      if (isa<Value>(insertSize) || isa<Value>(destSize)) {
        if (insertSize != destSize) {
          return rewriter.notifyMatchFailure(insertSliceOp,
                                             "dynamic size mismatch");
        }
        continue;
      }

      // `getMixedSizes` for different ops returns different attribute types
      // (`index` or `i64`) so we compare the values of the ints directly here.
      int64_t staticInsertSize = getConstantIntValue(insertSize).value();
      int64_t staticDestSize = getConstantIntValue(insertSize).value();
      if (staticInsertSize != staticDestSize) {
        return rewriter.notifyMatchFailure(insertSliceOp,
                                           "static size mismatch");
      }
    }

    rewriter.replaceOp(insertSliceOp, insertSliceOp.getSource());
    return success();
  }
};

/// Canonicalize operations in nested regions.
struct CanonicalizePass : public impl::CanonicalizePassBase<CanonicalizePass> {
  using IREE::Flow::impl::CanonicalizePassBase<
      CanonicalizePass>::CanonicalizePassBase;
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Inherit the same config defaults from the upstream canonicalizer pass.
    config.setUseTopDownTraversal().setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Normal);

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    // Pull in some borderline/downstream canonicalizations for the Flow
    // compilation phase.
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(owningPatterns);
    owningPatterns.add<FoldFullInsertSlice>(context);

    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }
  void runOnOperation() override {
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    config.enableConstantCSE(cseConstants);
    LogicalResult didConverge =
        applyPatternsGreedily(getOperation(), *patterns, config);
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
