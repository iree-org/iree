// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- ResolveShapeOps.cpp - Pass to resolve shape related ops ------------===//
//
// This file implements functionalities to resolve shape related ops. For
// dynamic shape dimensions, this pass traces them back to the original HAL
// interface bindings.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/PassDetail.h"
#include "iree/compiler/Conversion/Passes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
/// Replaces `std.dim` on `shapex.tie_shape` with `shapex.ranked_shape` so that
/// later this shape dimension query can be folded away via canonicalization on
/// `shapex.ranked_shape`.
///
/// This pattern translates the following IR sequence:
///
/// ```mlir
/// %dynamic_dim = ... : index
/// %shape = shapex.make_ranked_shape %dynamic_dim ...
/// %tie_shape = shapex.tie_shape ..., %shape : ...
/// ...
/// %get_dim = std.dim %tie_shape ...
/// ```
///
/// Into:
///
/// ```mlir
/// %dynamic_dim = ... : index
/// %shape = shapex.make_ranked_shape %dynamic_dim ...
/// %tie_shape = shapex.tie_shape ..., %shape : ...
/// ...
/// %get_dim = shapex.ranked_dim %shape ...
/// ```
struct StdDimResolver final : public OpRewritePattern<memref::DimOp> {
  using OpRewritePattern<memref::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto tieShapeOp =
        dyn_cast<Shape::TieShapeOp>(dimOp.memrefOrTensor().getDefiningOp());
    if (!tieShapeOp) return failure();

    Optional<int64_t> index = dimOp.getConstantIndex();
    assert(index.hasValue() && "expect constant index in `std.dim` operation");

    rewriter.replaceOpWithNewOp<Shape::RankedDimOp>(dimOp, tieShapeOp.shape(),
                                                    index.getValue());
    return success();
  }
};

/// Elides all `shapex.tie_shape` ops.
struct TieShapeElider final : public OpRewritePattern<Shape::TieShapeOp> {
  using OpRewritePattern<Shape::TieShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Shape::TieShapeOp tieOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(tieOp, tieOp.operand());
    return success();
  }
};

struct ResolveShapeOpsPass : public ResolveShapeOpsBase<ResolveShapeOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ShapeDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void ResolveShapeOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();

  OwningRewritePatternList dimPatterns(&getContext());
  dimPatterns.insert<StdDimResolver>(context);

  // Set up a target to convert all std.dim ops. We need a conversion target
  // here to error out early if some std.dim op cannot be converted.
  ConversionTarget target(*context);
  target.addIllegalOp<memref::DimOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(applyFullConversion(getOperation(), target,
                                 std::move(dimPatterns)))) {
    return signalPassFailure();
  }

  OwningRewritePatternList shapePatterns(&getContext());
  shapePatterns.insert<TieShapeElider>(context);
  Shape::RankedDimOp::getCanonicalizationPatterns(shapePatterns, context);

  // Then elide all shapex.tie_shape ops and canonicalize shapex.ranked_dim
  // given that we don't need the shape annotation anymore.
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(shapePatterns));
}

std::unique_ptr<OperationPass<FuncOp>> createResolveShapeOpsPass() {
  return std::make_unique<ResolveShapeOpsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
