//===- Transforms.h - Custom Transforms: TileGeneric+Bufferize --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef THIRD_PARTY_IREE_EXPERIMENTAL_RUNNERS_TRANSFORMS_H_
#define THIRD_PARTY_IREE_EXPERIMENTAL_RUNNERS_TRANSFORMS_H_

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Bufferize.h"

namespace mlir {
class BufferizeTypeConverter;
class FrozenRewritePatternList;

namespace linalg {

/// Specific pass and options to target a tiled and distributed nested linalg
/// abstraction.
struct TileAndDistributeOptions {
  LinalgTilingOptions tilingOptions;
};

struct TileAndDistributedLinalgOp {
  Operation *tiledGenericOp;
  Operation *tiledInnerGenericOp;
  LinalgOp tiledLinalgOp;
  TileAndDistributedLinalgOp &operator=(const TileAndDistributedLinalgOp &) =
      default;
};

Optional<TileAndDistributedLinalgOp> tileAndDistributeLinalgOp(
    PatternRewriter &rewriter, LinalgOp op,
    const TileAndDistributeOptions &options);

struct TileAndDistributePattern : public RewritePattern {
  /// MatchAnyOpTag-based constructor with a mandatory `filter`.
  TileAndDistributePattern(TileAndDistributeOptions options,
                           LinalgTransformationFilter filter,
                           PatternBenefit benefit = 1);
  /// Name-based constructor with an optional `filter`.
  TileAndDistributePattern(
      TileAndDistributeOptions options, StringRef opName, MLIRContext *context,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

 private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Options.
  TileAndDistributeOptions options;
};

}  // namespace linalg
}  // namespace mlir

#endif  // THIRD_PARTY_IREE_EXPERIMENTAL_RUNNERS_TRANSFORMS_H_
