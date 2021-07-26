// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_

#include "iree/compiler/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace linalg_ext {

/// Structure to represent the result of tiling operation.
struct TiledOp {
  /// Tiled op.
  Operation *op;
  /// Loops generated during tiling.
  SmallVector<Operation *> loops;
  /// Values that are replacements for the untiled operations.
  SmallVector<Value> results;
};

/// Main entry point for tiling LinalgExtOps using TiledOpInterface.  If the
/// `op` does not implement the `TiledOpInterface` returns a `TiledOp{}` value.
FailureOr<TiledOp> tileLinalgExtOp(OpBuilder &b, Operation *op, ValueRange dest,
                                   const linalg::LinalgTilingOptions &options);

/// Base rewrite pattern to tile and distribute operations that implement the
/// `TiledOpInterface`.
/// Base pattern for tiling TiledOpInterfaceOps.
struct TiledOpInterfaceBaseTilingPattern : public RewritePattern {
  TiledOpInterfaceBaseTilingPattern(MLIRContext *context,
                                    linalg::LinalgTilingOptions options,
                                    linalg::LinalgTransformationFilter filter =
                                        linalg::LinalgTransformationFilter(),
                                    PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        filter(filter),
        options(options) {}

  LogicalResult matchAndRewriteBase(Operation *op, ValueRange dest,
                                    PatternRewriter &rewriter,
                                    TiledOp &result) const;

 private:
  /// LinalgTransformMarker handles special attribute manipulations.
  linalg::LinalgTransformationFilter filter;
  /// Options to control tiling;
  linalg::LinalgTilingOptions options;
};

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
