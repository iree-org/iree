// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_

#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

/// Structure to represent the result of tiling operation.
struct TiledOp {
  /// Tiled op.
  Operation *op;
  /// Loops generated during tiling.
  SmallVector<Operation *> loops;
  /// Values that are replacements for the untiled operations.
  SmallVector<Value> results;
};

/// Main entry point for tiling LinalgExtOps using TiledOpInterface.
FailureOr<TiledOp> tileLinalgExtOp(OpBuilder &b, TiledOpInterface tilableOp,
                                   const linalg::LinalgTilingOptions &options);

/// Base rewrite pattern to tile and distribute operations that implement the
/// `TiledOpInterface`.
/// Base pattern for tiling TiledOpInterfaceOps.
struct TiledOpInterfaceBaseTilingPattern
    : public OpInterfaceRewritePattern<TiledOpInterface> {
  TiledOpInterfaceBaseTilingPattern(MLIRContext *context,
                                    linalg::LinalgTilingOptions options,
                                    linalg::LinalgTransformationFilter filter =
                                        linalg::LinalgTransformationFilter(),
                                    PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit),
        filter(filter),
        options(options) {}

  LogicalResult matchAndRewriteBase(TiledOpInterface tilableOp,
                                    PatternRewriter &rewriter,
                                    TiledOp &result) const;

 private:
  /// LinalgTransformMarker handles special attribute manipulations.
  linalg::LinalgTransformationFilter filter;
  /// Options to control tiling;
  linalg::LinalgTilingOptions options;
};

struct TiledOpInterfaceTilingPattern
    : public TiledOpInterfaceBaseTilingPattern {
  TiledOpInterfaceTilingPattern(MLIRContext *context,
                                linalg::LinalgTilingOptions options,
                                linalg::LinalgTransformationFilter filter =
                                    linalg::LinalgTransformationFilter(),
                                PatternBenefit benefit = 1)
      : TiledOpInterfaceBaseTilingPattern(context, options, filter, benefit) {}

  LogicalResult matchAndRewrite(TiledOpInterface tilableOp,
                                PatternRewriter &rewriter) const override {
    TiledOp tiledOp;
    // Check for failure.
    if (failed(TiledOpInterfaceBaseTilingPattern::matchAndRewriteBase(
            tilableOp, rewriter, tiledOp))) {
      return failure();
    }
    // Check for do-nothing case.
    if (!tiledOp.op) return failure();
    if (tiledOp.op != tilableOp) {
      if (tiledOp.results.empty()) {
        rewriter.eraseOp(tilableOp);
      } else {
        rewriter.replaceOp(tilableOp, tiledOp.results);
      }
    }
    return success();
  }
};

}  // namespace LinalgExt
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
