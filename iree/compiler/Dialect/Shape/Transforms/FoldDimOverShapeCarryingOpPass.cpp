// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

namespace {

template <typename DimOp>
class FoldDimOp : public OpRewritePattern<DimOp> {
  using OpRewritePattern<DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DimOp op,
                                PatternRewriter &rewriter) const override {
    auto shapeCarryingOp =
        dyn_cast<ShapeCarryingInterface>(op.source().getDefiningOp());
    if (!shapeCarryingOp) return failure();

    IntegerAttr index;
    if (!matchPattern(op.index(), m_Constant(&index))) return failure();

    auto shapeOp =
        shapeCarryingOp.buildResultValueRankedShape(op.source(), rewriter);
    rewriter.replaceOpWithNewOp<RankedDimOp>(op, op.getType(), shapeOp, index);
    return success();
  }
};

class FoldDimOverShapeCarryingOpPass
    : public PassWrapper<FoldDimOverShapeCarryingOpPass, FunctionPass> {
  StringRef getArgument() const override {
    return "iree-fold-dim-over-shape-carrying-op";
  }

  StringRef getDescription() const override {
    return "Fold tensor.dim/memref.dim ops taking shape carrying ops as "
           "operands";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ShapeDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<FoldDimOp<memref::DimOp>, FoldDimOp<tensor::DimOp>>(
        &getContext());
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

}  // namespace

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OperationPass<FuncOp>> createFoldDimOverShapeCarryingOpPass() {
  return std::make_unique<Shape::FoldDimOverShapeCarryingOpPass>();
}

static PassRegistration<Shape::FoldDimOverShapeCarryingOpPass> pass;

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
