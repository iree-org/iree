// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"

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
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<CleanupTieShapePattern>(&getContext());
    applyPatternsGreedily(getFunction(), patterns);
  }
};

}  // namespace
}  // namespace Shape

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OperationPass<FuncOp>> createCleanupShapePlaceholdersPass() {
  return std::make_unique<Shape::CleanupShapePlaceholdersPass>();
}

static PassRegistration<Shape::CleanupShapePlaceholdersPass> pass(
    "iree-shape-cleanup-placeholders",
    "Cleans up unnecessary shape placeholders.");

}  // namespace iree_compiler
}  // namespace mlir
