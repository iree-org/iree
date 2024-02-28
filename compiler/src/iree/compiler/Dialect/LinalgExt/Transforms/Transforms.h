// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

//===----------------------------------------------------------------------===//
// Transformations exposed as patterns, moved from upstream MLIR as IREE still
// heavily relies on patterns that compose through filters.
// TODO: Deprecate all the code below.
//===----------------------------------------------------------------------===//
/// Wrap upstream linalg::splitReduction with a filter.
inline FailureOr<linalg::LinalgOp>
splitReduction(PatternRewriter &b, linalg::LinalgOp op,
               const linalg::ControlSplitReductionFn &controlSplitReductionFn,
               const LinalgTransformationFilter &filter,
               bool useAlloc = false) {
  if (failed(filter.checkAndNotify(b, op)) || !op.hasPureTensorSemantics() ||
      op.getNumReductionLoops() != 1 || op.getNumDpsInits() != 1 ||
      !op.hasOnlyProjectedPermutations())
    return b.notifyMatchFailure(op, "precondition not met");

  FailureOr<linalg::SplitReductionResult> res =
      linalg::splitReduction(b, op, controlSplitReductionFn, useAlloc);
  if (failed(res))
    return failure();

  filter.replaceLinalgTransformationFilter(b, res->splitLinalgOp);
  filter.replaceLinalgTransformationFilter(b, res->resultCombiningLinalgOp);

  return res->splitLinalgOp;
}

///
/// Linalg promotion patterns.
///
/// Apply the `promoteSubViews` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `promoteSubViews` for more details.
struct LinalgBasePromotionPattern : public RewritePattern {
  /// Entry point to match any LinalgOp
  /// OpInterface. MatchAnyOpTag-based constructor
  /// with a mandatory `filter`.
  LinalgBasePromotionPattern(
      MLIRContext *context, LinalgTransformationFilter f,
      linalg::LinalgPromotionOptions options = linalg::LinalgPromotionOptions(),
      PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        filter(std::move(f)), options(std::move(options)) {}
  /// Entry point to match a specific Linalg op.
  LinalgBasePromotionPattern(
      StringRef opName, MLIRContext *context,
      linalg::LinalgPromotionOptions options,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : RewritePattern(opName, benefit, context, {}), filter(std::move(f)),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, op)))
      return failure();
    if (failed(promoteSubviewsPrecondition(op, options)))
      return failure();

    // TODO: We cannot use root update here. This
    // pattern is creating other ops, so if the
    // promotion fails, those need to be cleaned
    // up, which doesnt seem to be happening here.
    // So to fail properly, we should be cloning
    // the op and deleting the previous op. This
    // needs more investigation.
    rewriter.startOpModification(op);
    std::optional<linalg::LinalgOp> promotedOp =
        promoteSubViews(rewriter, cast<linalg::LinalgOp>(op), options);
    if (!promotedOp) {
      rewriter.cancelOpModification(op);
      return op->emitError("subview promotion failed");
    }
    rewriter.finalizeOpModification(op);
    filter.replaceLinalgTransformationFilter(rewriter, op);
    return success();
  }

private:
  /// LinalgTransformMarker handles special
  /// attribute manipulations.
  LinalgTransformationFilter filter;
  /// Promotion options.
  linalg::LinalgPromotionOptions options;
};

template <typename OpTy>
struct LinalgPromotionPattern : public LinalgBasePromotionPattern {
  /// SFINAE: This constructor can only trigger for
  /// concrete ops that have a static
  /// `getOperationName` method.
  template <typename ConcreateOpTy = OpTy>
  LinalgPromotionPattern(
      MLIRContext *context, linalg::LinalgPromotionOptions options,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(OpTy::getOperationName(), context, options,
                                   f, benefit) {}
  /// This constructor is available to anyone.
  LinalgPromotionPattern(
      StringRef opName, MLIRContext *context,
      linalg::LinalgPromotionOptions options,
      LinalgTransformationFilter f = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(opName, context, options, f, benefit) {}
};

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
