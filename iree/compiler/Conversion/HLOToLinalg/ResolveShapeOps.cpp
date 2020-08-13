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

//===- ResolveShapeOps.cpp - Pass to resolve shape related ops ------------===//
//
// This file implements functionalities to resolve shape related ops. For
// dynamic shape dimensions, this pass traces them back to the original HAL
// interface bindings.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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
struct StdDimResolver final : public OpRewritePattern<DimOp> {
  using OpRewritePattern<DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DimOp dimOp,
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

struct ResolveShapeOpsPass
    : public PassWrapper<ResolveShapeOpsPass, FunctionPass> {
  void runOnFunction() override;
};
}  // namespace

void ResolveShapeOpsPass::runOnFunction() {
  MLIRContext *context = &getContext();

  OwningRewritePatternList dimPatterns;
  dimPatterns.insert<StdDimResolver>(context);

  // Set up a target to convert all std.dim ops. We need a conversion target
  // here to error out early if some std.dim op cannot be converted.
  ConversionTarget target(*context);
  target.addIllegalOp<DimOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(applyFullConversion(getFunction(), target, dimPatterns))) {
    return signalPassFailure();
  }

  OwningRewritePatternList shapePatterns;
  shapePatterns.insert<TieShapeElider>(context);
  Shape::RankedDimOp::getCanonicalizationPatterns(shapePatterns, context);

  // Then elide all shapex.tie_shape ops and canonicalize shapex.ranked_dim
  // given that we don't need the shape annotation anymore.
  applyPatternsAndFoldGreedily(getFunction(), shapePatterns);
}

std::unique_ptr<OperationPass<FuncOp>> createResolveShapeOpsPass() {
  return std::make_unique<ResolveShapeOpsPass>();
}

static PassRegistration<ResolveShapeOpsPass> pass("iree-codegen-resolve-shape",
                                                  "resolve shape");
}  // namespace iree_compiler
}  // namespace mlir
