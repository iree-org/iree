// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

namespace {

class CleanupTieShapePattern : public OpRewritePattern<Shape::TieShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TieShapeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.operand());
    return success();
  }
};

class CleanupShapePlaceholdersPass
    : public PassWrapper<CleanupShapePlaceholdersPass, FunctionPass> {
  StringRef getArgument() const override {
    return "iree-shape-cleanup-placeholders";
  }

  StringRef getDescription() const override {
    return "Cleans up unnecessary shape placeholders.";
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<CleanupTieShapePattern>(&getContext());
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

}  // namespace

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OperationPass<FuncOp>> createCleanupShapePlaceholdersPass() {
  return std::make_unique<Shape::CleanupShapePlaceholdersPass>();
}

static PassRegistration<Shape::CleanupShapePlaceholdersPass> pass;

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
