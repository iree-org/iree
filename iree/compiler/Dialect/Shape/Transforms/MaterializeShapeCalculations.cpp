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

#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeInterface.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/Shape/Plugins/XLA/XlaHloShapeBuilder.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

namespace {

Value rewriteShapexRankedBroadcastShape(RankedBroadcastShapeOp bcastOp,
                                        OpBuilder &builder) {
  auto lhsRs = bcastOp.lhs().getType().cast<RankedShapeType>();
  auto rhsRs = bcastOp.rhs().getType().cast<RankedShapeType>();

  auto loc = bcastOp.getLoc();
  auto resultRs = bcastOp.getResult().getType().cast<RankedShapeType>();
  auto dimType = resultRs.getDimType();

  // Pairs of the shape dim and corresponding value if dynamic.
  SmallVector<std::pair<Optional<int>, Value>, 4> lhsDims;
  SmallVector<std::pair<Optional<int>, Value>, 4> rhsDims;
  lhsDims.resize(resultRs.getRank());
  rhsDims.resize(resultRs.getRank());

  // Populate the lhs dims.
  for (auto dimMap : llvm::enumerate(bcastOp.lhs_broadcast_dimensions())) {
    auto inputDimIndex = dimMap.index();
    auto outputDimIndex = dimMap.value().getZExtValue();
    assert(outputDimIndex < lhsDims.size());
    if (!resultRs.isDimDynamic(outputDimIndex)) {
      // No need to populate fully static dimensions.
      continue;
    }
    if (lhsRs.isDimDynamic(inputDimIndex)) {
      lhsDims[outputDimIndex] =
          std::make_pair(-1, builder.create<RankedDimOp>(
                                 loc, dimType, bcastOp.lhs(),
                                 builder.getI64IntegerAttr(inputDimIndex)));
    } else {
      lhsDims[outputDimIndex] = std::make_pair(inputDimIndex, nullptr);
    }
  }

  // Populate the rhs dims.
  for (auto dimMap : llvm::enumerate(bcastOp.rhs_broadcast_dimensions())) {
    auto inputDimIndex = dimMap.index();
    auto outputDimIndex = dimMap.value().getZExtValue();
    assert(outputDimIndex < rhsDims.size());
    if (!resultRs.isDimDynamic(outputDimIndex)) {
      // No need to populate fully static dimensions.
      continue;
    }
    if (rhsRs.isDimDynamic(inputDimIndex)) {
      rhsDims[outputDimIndex] =
          std::make_pair(-1, builder.create<RankedDimOp>(
                                 loc, dimType, bcastOp.rhs(),
                                 builder.getI64IntegerAttr(inputDimIndex)));
    } else {
      rhsDims[outputDimIndex] = std::make_pair(inputDimIndex, nullptr);
    }
  }

  // Now compute dynamic dims for each output dim.
  SmallVector<Value, 4> dynamicDims;
  for (int i = 0; i < lhsDims.size(); ++i) {
    if (!resultRs.isDimDynamic(i)) continue;
    auto lhsDimInfo = lhsDims[i];
    auto lhsDimSize = lhsDimInfo.first ? *lhsDimInfo.first : -1;
    auto rhsDimInfo = rhsDims[i];
    auto rhsDimSize = rhsDimInfo.first ? *rhsDimInfo.first : -1;

    if (lhsDimSize > 1) {
      // Non-degenerate static.
      bcastOp.emitRemark(
          "broadcast of non-degenerate lhs static dim not implemented");
      return nullptr;
    } else if (rhsDimSize > 1) {
      // Non-degenerate static.
      bcastOp.emitRemark(
          "broadcast of non-degenerate rhs static dim not implemented");
      return nullptr;
    } else if (lhsDimSize == 1) {
      // Degenerate static.
      bcastOp.emitRemark(
          "broadcast of degenerate lhs static dim not implemented");
      return nullptr;
    } else if (rhsDimSize == 1) {
      // Degenerate static.
      bcastOp.emitRemark(
          "broadcast of degenerate rhs static dim not implemented");
      return nullptr;
    } else {
      // Dynamic.
      // TODO: Generate code to assert.
      if (lhsDimInfo.second) {
        dynamicDims.push_back(lhsDimInfo.second);
      } else if (rhsDimInfo.second) {
        dynamicDims.push_back(rhsDimInfo.second);
      } else {
        return nullptr;
      }
    }
  }

  // And make the result shape.
  return builder.create<MakeRankedShapeOp>(loc, resultRs, dynamicDims);
}

class ExpandRankedBroadcastShapePattern
    : public OpRewritePattern<RankedBroadcastShapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RankedBroadcastShapeOp bcastOp,
                                PatternRewriter &rewriter) const override {
    auto newValue = rewriteShapexRankedBroadcastShape(bcastOp, rewriter);
    if (!newValue) return failure();

    rewriter.replaceOp(bcastOp, newValue);
    return success();
  }
};

class MaterializeRunTimeRankedShapePattern
    : public OpRewritePattern<Shape::GetRankedShapeOp> {
 public:
  MaterializeRunTimeRankedShapePattern(MLIRContext *context)
      : OpRewritePattern(context, 1) {}

  LogicalResult matchAndRewrite(GetRankedShapeOp getShapeOp,
                                PatternRewriter &rewriter) const override {
    auto shapeType = getShapeOp.shape().getType().dyn_cast<RankedShapeType>();
    SmallVector<Value, 4> dynamicDims;
    for (int64_t i = 0, e = shapeType.getRank(); i < e; ++i) {
      if (!shapeType.isDimDynamic(i)) continue;
      dynamicDims.push_back(
          rewriter.create<DimOp>(getShapeOp.getLoc(), getShapeOp.operand(), i));
    }

    // TODO(laurenzo): Remove once further along (it is fine to be unsupported
    // as it will fall back to generic), but in these early phases, it is
    // extremely useful to be chatty about this fallback.
    auto inputOperation = getShapeOp.operand().getDefiningOp();
    if (inputOperation) {
      inputOperation->emitRemark()
          << "unable to materialize shape calculation (unsupported op '"
          << inputOperation->getName() << "'?): falling back to runtime "
          << "resolution";
    }

    rewriter.replaceOpWithNewOp<MakeRankedShapeOp>(getShapeOp, shapeType,
                                                   dynamicDims);
    return success();
  }
};

class MaterializeCompileTimeRankedShapePattern
    : public OpRewritePattern<Shape::GetRankedShapeOp> {
 public:
  MaterializeCompileTimeRankedShapePattern(
      const CustomOpShapeBuilderList *customOpShapeBuilder,
      MLIRContext *context)
      : OpRewritePattern(context, 10),
        customOpShapeBuilder(customOpShapeBuilder) {}

  LogicalResult matchAndRewrite(GetRankedShapeOp getShapeOp,
                                PatternRewriter &rewriter) const override {
    // Check for static shape and elide.
    auto operandType =
        getShapeOp.operand().getType().dyn_cast<RankedTensorType>();
    auto shapeType = getShapeOp.shape().getType().dyn_cast<RankedShapeType>();
    if (operandType && shapeType && operandType.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<ConstRankedShapeOp>(getShapeOp, shapeType);
      return success();
    }

    // Check for input operation (unless if a small set of shape ops).
    Operation *inputOperation = getShapeOp.operand().getDefiningOp();
    if (inputOperation && !llvm::isa<TieShapeOp>(inputOperation)) {
      return rewriteInputOp(getShapeOp, inputOperation, rewriter);
    }

    return failure();
  }

 private:
  // Matches the case where the input to a GetRankedShapeOp is another
  // operation. This is the primary supported case as other rewrites should
  // have isolated function/block boundaries with TieShape ops.
  LogicalResult rewriteInputOp(GetRankedShapeOp getShapeOp,
                               Operation *inputOperation,
                               PatternRewriter &rewriter) const {
    // SameOperandsAndResultShape trait.
    if (inputOperation->hasTrait<OpTrait::SameOperandsAndResultShape>() ||
        inputOperation->hasTrait<OpTrait::SameOperandsAndResultType>()) {
      return rewriteSameOperandsAndResultShape(getShapeOp, inputOperation,
                                               rewriter);
    }

    // Custom shapes.
    if (customOpShapeBuilder) {
      auto resultShape = getShapeOp.getRankedShape();
      for (auto &shapeBuilder : *customOpShapeBuilder) {
        Value customShape = shapeBuilder->buildRankedShape(
            resultShape, inputOperation, rewriter);
        if (customShape) {
          rewriter.replaceOp(getShapeOp, customShape);
          return success();
        }
      }
    }

    return failure();
  }

  LogicalResult rewriteSameOperandsAndResultShape(
      GetRankedShapeOp getShapeOp, Operation *inputOperation,
      PatternRewriter &rewriter) const {
    llvm::SmallVector<Value, 4> inputOperands(inputOperation->getOperands());
    auto combinedShapeOp = buildCastInputsToResultShape(
        inputOperation->getLoc(), getShapeOp.getRankedShape(), inputOperands,
        rewriter);
    if (!combinedShapeOp) return failure();
    rewriter.replaceOp(getShapeOp, {combinedShapeOp});
    return success();
  }

  const CustomOpShapeBuilderList *customOpShapeBuilder;
};

class MaterializeShapeCalculationsPass
    : public FunctionPass<MaterializeShapeCalculationsPass> {
 public:
  // Gets a CustomOpShapeBuilderList for expanding shapes of custom ops.
  // By default, returns nullptr, which will not handle custom op shapes.
  // TODO(laurenzo): Since it isn't clear yet whether we need this facility
  // long term (i.e. this should come from the ops themselves), we are just
  // hard-linking it here at the expense of a dependency problem. Decouple this
  // if the facility persists.
  const CustomOpShapeBuilderList *getCustomOpShapeBuilder() {
    static CustomOpShapeBuilderList globalBuilders = ([]() {
      CustomOpShapeBuilderList builders;
      xla_hlo::populateXlaHloCustomOpShapeBuilder(builders);
      return builders;
    })();
    return &globalBuilders;
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    // Always include certain canonicalizations that interop.
    CastCompatibleShapeOp::getCanonicalizationPatterns(patterns, &getContext());
    GetRankedShapeOp::getCanonicalizationPatterns(patterns, &getContext());
    MakeRankedShapeOp::getCanonicalizationPatterns(patterns, &getContext());
    RankedDimOp::getCanonicalizationPatterns(patterns, &getContext());
    TieShapeOp::getCanonicalizationPatterns(patterns, &getContext());
    patterns.insert<ExpandRankedBroadcastShapePattern>(&getContext());
    patterns.insert<MaterializeCompileTimeRankedShapePattern>(
        getCustomOpShapeBuilder(), &getContext());
    patterns.insert<MaterializeRunTimeRankedShapePattern>(&getContext());
    applyPatternsGreedily(getFunction(), patterns);

    OwningRewritePatternList fallbackPatterns;
  }
};

}  // namespace
}  // namespace Shape

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OpPassBase<FuncOp>> createMaterializeShapeCalculationsPass() {
  return std::make_unique<Shape::MaterializeShapeCalculationsPass>();
}

static PassRegistration<Shape::MaterializeShapeCalculationsPass> pass(
    "iree-shape-materialize-calculations", "Materializes shape calculations.");

}  // namespace iree_compiler
}  // namespace mlir
