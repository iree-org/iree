// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_PASSES_TRANSFORMS_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_PASSES_TRANSFORMS_H_

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

/// Structure to represent the result of tiling operation.
struct TiledOp {
  /// Tiled operations that are created during tiling.
  SmallVector<Operation *> op;
  /// Loops generated during tiling.
  SmallVector<Operation *> loops;
  /// Values that are replacements for the untiled operations.
  SmallVector<Value> results;
};

/// Main entry point for tiling LinalgExtOps using TiledOpInterface.
FailureOr<TiledOp> tileLinalgExtOp(OpBuilder &b, TilingInterface tilableOp,
                                   const linalg::LinalgTilingOptions &options);

/// Base rewrite pattern to tile and distribute operations that implement the
/// `TiledOpInterface`.
/// Base pattern for tiling TiledOpInterfaceOps.
struct TilingInterfaceBaseTilingPattern
    : public OpInterfaceRewritePattern<TilingInterface> {
  TilingInterfaceBaseTilingPattern(
      MLIRContext *context, linalg::LinalgTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), filter(filter),
        options(options) {}

  LogicalResult matchAndRewriteBase(TilingInterface tilableOp,
                                    PatternRewriter &rewriter,
                                    TiledOp &result) const;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Options to control tiling;
  linalg::LinalgTilingOptions options;
};

struct TilingInterfaceTilingPattern : public TilingInterfaceBaseTilingPattern {
  TilingInterfaceTilingPattern(
      MLIRContext *context, linalg::LinalgTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : TilingInterfaceBaseTilingPattern(context, options, filter, benefit) {}

  LogicalResult matchAndRewrite(TilingInterface tilableOp,
                                PatternRewriter &rewriter) const;
};

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_PASSES_TRANSFORMS_H_
