// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace scf {
class ForOp;
}
namespace linalg {
class LinalgOp;
}

namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

struct TilingResult {
  TileOp tileOp;
  Operation *tiledOp;
};

/// Pattern to tile a TilingInterface op using a TileOp.
struct LinalgExtTilingPattern
    : public OpInterfaceRewritePattern<TilingInterface> {
  LinalgExtTilingPattern(MLIRContext *context, linalg::LinalgTilingOptions opt)
      : OpInterfaceRewritePattern<TilingInterface>(context), options(opt) {}

  FailureOr<TilingResult>
  returningMatchAndRewrite(TilingInterface op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  linalg::LinalgTilingOptions options;
};

/// Pattern to swap a `TilingInterface` op -> `tensor::ExtractSliceOp`.
struct SwapTilingInterfaceOp : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  FailureOr<Operation *>
  returningMatchAndRewrite(tensor::ExtractSliceOp sliceOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(sliceOp, rewriter);
  }
};

/// Pattern to rewrite a TileOp to an scf::ForOp.
struct TileOpToSCFRewriter : public OpRewritePattern<TileOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<scf::ForOp>
  returningMatchAndRewrite(TileOp tileOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TileOp tileOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(tileOp, rewriter);
  }
};

/// Pattern to rewrite a TileOp to a InParallelOp.
struct TileOpToInParallelRewriter : public OpRewritePattern<TileOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<InParallelOp>
  returningMatchAndRewrite(TileOp tileOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TileOp tileOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(tileOp, rewriter);
  }
};

/// Pattern to rewrite a InParallelOp to the async dialect.
struct InParallelOpToAsyncRewriter : public OpRewritePattern<InParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<Operation *>
  returningMatchAndRewrite(InParallelOp inParallelOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(InParallelOp inParallelOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(inParallelOp, rewriter);
  }
};

/// Pattern to rewrite a InParallelOp to the HAL dialect.
struct InParallelOpToHALRewriter : public OpRewritePattern<InParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<SmallVector<Operation *>>
  returningMatchAndRewrite(InParallelOp inParallelOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(InParallelOp inParallelOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(inParallelOp, rewriter);
  }
};

/// Pattern to rewrite a InParallelOp to an scf::ForOp.
struct InParallelOpToScfForRewriter : public OpRewritePattern<InParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<scf::ForOp>
  returningMatchAndRewrite(InParallelOp inParallelOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(InParallelOp inParallelOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(inParallelOp, rewriter);
  }
};

/// Pattern to fuse a LinalgOp into a containing op.
struct LinalgExtFusionInContainingOpPattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  LinalgExtFusionInContainingOpPattern(MLIRContext *context,
                                       Operation *containingOp)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context),
        containingOp(containingOp) {}

  FailureOr<SmallVector<linalg::LinalgOp>>
  returningMatchAndRewrite(linalg::LinalgOp producerOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(linalg::LinalgOp producerOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(producerOp, rewriter);
  }

private:
  Operation *containingOp;
};

struct FusionResult {
  linalg::LinalgOp consumerOp;
  SmallVector<linalg::LinalgOp> fusedOps;
};

/// Pattern to fuse the producers of a LinalgOp.
struct LinalgExtFusionPattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  LinalgExtFusionPattern(MLIRContext *context, ArrayRef<int64_t> operandsToFuse)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context),
        operandsToFuse(operandsToFuse.begin(), operandsToFuse.end()) {}

  FailureOr<FusionResult>
  returningMatchAndRewrite(linalg::LinalgOp consumerOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(linalg::LinalgOp consumerOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(consumerOp, rewriter);
  }

private:
  SmallVector<int64_t> operandsToFuse;
};

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
