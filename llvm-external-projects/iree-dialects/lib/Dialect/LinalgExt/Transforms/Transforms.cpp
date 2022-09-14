// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {
/// Linalg tiling pattern.
LinalgTilingPattern::LinalgTilingPattern(MLIRContext *context,
                                         linalg::LinalgTilingOptions options,
                                         linalg::LinalgTransformationFilter f,
                                         PatternBenefit benefit)
    : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
      filter(std::move(f)), options(std::move(options)) {}

LinalgTilingPattern::LinalgTilingPattern(StringRef opName, MLIRContext *context,
                                         linalg::LinalgTilingOptions options,
                                         linalg::LinalgTransformationFilter f,
                                         PatternBenefit benefit)
    : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
      filter(f.addOpNameFilter(opName)), options(std::move(options)) {}

FailureOr<linalg::TiledLinalgOp>
LinalgTilingPattern::returningMatchAndRewrite(linalg::LinalgOp op,
                                              PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, op)))
    return failure();

  FailureOr<linalg::TiledLinalgOp> res = tileLinalgOp(rewriter, op, options);
  if (failed(res))
    return failure();

  // Clear filter to stop recursive pattern application.
  // This must be done here to properly propagate to peeling branches.
  filter.replaceLinalgTransformationFilter(rewriter, res->op);

  // Peel the loops of the TiledLinalgOp.
  peelTiledLinalgOp(rewriter, *res, options.peeledLoops, options.loopType);

  if (res->tensorResults.empty())
    rewriter.eraseOp(op);
  else
    rewriter.replaceOp(op, res->tensorResults);

  return res;
}

LinalgVectorizationPattern::LinalgVectorizationPattern(
    MLIRContext *context, linalg::LinalgTransformationFilter f,
    linalg::LinalgVectorizationOptions options, PatternBenefit benefit)
    : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
      filter(std::move(f)) {}

LinalgVectorizationPattern::LinalgVectorizationPattern(
    StringRef opName, MLIRContext *context,
    linalg::LinalgVectorizationOptions options,
    linalg::LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
      filter(f.addOpNameFilter(opName)) {}

LogicalResult
LinalgVectorizationPattern::matchAndRewrite(linalg::LinalgOp linalgOp,
                                            PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, linalgOp)))
    return failure();
  return vectorize(rewriter, linalgOp);
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
