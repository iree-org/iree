//===- Transforms.h - LinalgExt transformations as patterns -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace scf {
class ForOp;
}

namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

/// Pattern to tile a TilingInterface op using a TileOp.
struct LinalgExtTilingPattern
    : public OpInterfaceRewritePattern<TilingInterface> {
  LinalgExtTilingPattern(MLIRContext *context, linalg::LinalgTilingOptions opt)
      : OpInterfaceRewritePattern<TilingInterface>(context), options(opt) {}

  FailureOr<Operation *> returningMatchAndRewrite(
      TilingInterface op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

 private:
  linalg::LinalgTilingOptions options;
};

/// Pattern to rewrite a TileOp to an scf::ForOp.
struct TileOpToSCFRewriter : public OpRewritePattern<TileOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<scf::ForOp> returningMatchAndRewrite(
      TileOp tileOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TileOp tileOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(tileOp, rewriter);
  }
};

/// Pattern to rewrite a TileOp to a InParallelOp.
struct TileOpToInParallelRewriter : public OpRewritePattern<TileOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<InParallelOp> returningMatchAndRewrite(
      TileOp tileOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TileOp tileOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(tileOp, rewriter);
  }
};

/// Pattern to rewrite a InParallelOp to the async dialect.
struct InParallelOpToAsyncRewriter : public OpRewritePattern<InParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<Operation *> returningMatchAndRewrite(
      InParallelOp inParallelOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(InParallelOp inParallelOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(inParallelOp, rewriter);
  }
};

/// Pattern to rewrite a InParallelOp to an scf::ForOp.
struct InParallelOpToScfForRewriter : public OpRewritePattern<InParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<scf::ForOp> returningMatchAndRewrite(
      InParallelOp inParallelOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(InParallelOp inParallelOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(inParallelOp, rewriter);
  }
};

}  // namespace LinalgExt
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
