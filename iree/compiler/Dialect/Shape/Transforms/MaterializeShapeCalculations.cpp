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
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeInterface.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/Shape/Plugins/VMLA/VMLAShapeBuilder.h"
#include "iree/compiler/Dialect/Shape/Plugins/XLA/XlaHloShapeBuilder.h"
#include "iree/compiler/Dialect/Shape/Transforms/Patterns.h"
#include "iree/compiler/Utils/PatternUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-shape"

namespace mlir {
namespace iree_compiler {
namespace Shape {

namespace {

// Gets a CustomOpShapeBuilderList for expanding shapes of custom ops.
// By default, returns nullptr, which will not handle custom op shapes.
// TODO(laurenzo): Since it isn't clear yet whether we need this facility
// long term (i.e. this should come from the ops themselves), we are just
// hard-linking it here at the expense of a dependency problem. Decouple this
// if the facility persists.
const CustomOpShapeBuilderList *getCustomOpShapeBuilder() {
  static CustomOpShapeBuilderList globalBuilders = ([]() {
    CustomOpShapeBuilderList builders;
    mhlo::populateXlaHloCustomOpShapeBuilder(builders);
    IREE::VMLA::populateVMLACustomOpShapeBuilder(builders);
    return builders;
  })();
  return &globalBuilders;
}

Value rewriteShapexRankedBroadcastShape(
    RankedBroadcastShapeOp op, RankedBroadcastShapeOp::Adaptor operands,
    OpBuilder &builder) {
  auto lhs = operands.lhs();
  auto rhs = operands.rhs();
  auto loc = op.getLoc();
  auto resultRs = op.getResult().getType().cast<RankedShapeType>();

  auto c1 = builder.create<ConstantIndexOp>(loc, 1);
  // Entries are the extent of the output along that dimension corresponding to
  // the given side, or 1 (which is neutral w.r.t. broadcasting).
  SmallVector<Value, 4> lhsResultExtents(resultRs.getRank(), c1);
  SmallVector<Value, 4> rhsResultExtents(resultRs.getRank(), c1);

  for (auto dim : llvm::enumerate(op.lhs_broadcast_dimensions())) {
    auto inputDim = dim.index();
    auto outputDim = dim.value().getZExtValue();
    lhsResultExtents[outputDim] =
        builder.create<RankedDimOp>(loc, lhs, inputDim);
  }
  for (auto dim : llvm::enumerate(op.rhs_broadcast_dimensions())) {
    auto inputDim = dim.index();
    auto outputDim = dim.value().getZExtValue();
    rhsResultExtents[outputDim] =
        builder.create<RankedDimOp>(loc, rhs, inputDim);
  }

  SmallVector<Value, 4> resultExtents;
  for (auto t : llvm::zip(lhsResultExtents, rhsResultExtents)) {
    auto lhsExtent = std::get<0>(t);
    auto rhsExtent = std::get<1>(t);
    auto ugt =
        builder.create<CmpIOp>(loc, CmpIPredicate::ugt, lhsExtent, rhsExtent);
    auto max = builder.create<SelectOp>(loc, ugt, lhsExtent, rhsExtent);
    resultExtents.push_back(max);
    // TODO(silvasean): Create error handling code for invalid broadcasts.
    // Use vm.cond_fail (or something that lowers to that).
  }

  // MakeRankedShapeOp only accepts the dynamic dims, so filter appropriately.
  SmallVector<Value, 4> filteredResultExtents;
  for (int i = 0, e = resultRs.getRank(); i < e; i++) {
    if (resultRs.isDimDynamic(i)) {
      filteredResultExtents.push_back(resultExtents[i]);
    }
  }

  return builder.create<MakeRankedShapeOp>(loc, resultRs,
                                           filteredResultExtents);
}

LogicalResult expandGatherExtentsOp(GatherExtentsOp op,
                                    GatherExtentsOp::Adaptor operands,
                                    PatternRewriter &rewriter) {
  // Calculate cumulative sums of the ranks of each operand, which allows
  // us to map each index to its corresponding operand easily.
  SmallVector<int64_t, 6> cumsum;
  cumsum.push_back(0);
  for (auto operand : operands.shapes()) {
    auto rank = operand.getType().cast<Shape::RankedShapeType>().getRank();
    cumsum.push_back(cumsum.back() + rank);
  }

  // For each index, extract the relevant extent from the operands.
  SmallVector<Value, 6> extents;
  for (auto index : op.indices().getValues<int64_t>()) {
    auto it = llvm::upper_bound(cumsum, index) - 1;
    auto operandNum = std::distance(cumsum.begin(), it);
    auto dimNum = index - *it;
    auto extent = rewriter.create<Shape::RankedDimOp>(
        op.getLoc(), operands.shapes()[operandNum], dimNum);
    extents.push_back(extent);
  }

  // Due to a quirk of MakeRankedShapeOp, we only want the dynamic
  // dimensions.
  SmallVector<Value, 6> onlyDynamicExtents;
  auto resultType = op.result().getType().cast<Shape::RankedShapeType>();
  for (int i = 0, e = resultType.getRank(); i < e; i++) {
    if (resultType.isDimDynamic(i)) {
      onlyDynamicExtents.push_back(extents[i]);
    }
  }

  rewriter.replaceOpWithNewOp<Shape::MakeRankedShapeOp>(op, resultType,
                                                        onlyDynamicExtents);
  return success();
}

LogicalResult expandRankedBroadcastShapePattern(
    RankedBroadcastShapeOp bcastOp, RankedBroadcastShapeOp::Adaptor operands,
    PatternRewriter &rewriter) {
  auto newValue =
      rewriteShapexRankedBroadcastShape(bcastOp, operands, rewriter);
  if (!newValue) return failure();

  rewriter.replaceOp(bcastOp, newValue);
  return success();
}

LogicalResult rewriteSameOperandsAndResultShape(GetRankedShapeOp getShapeOp,
                                                Operation *inputOperation,
                                                PatternRewriter &rewriter) {
  llvm::SmallVector<Value, 4> inputOperands(inputOperation->getOperands());
  auto combinedShapeOp = buildCastInputsToResultShape(
      inputOperation->getLoc(), getShapeOp.getRankedShape(), inputOperands,
      rewriter);
  if (!combinedShapeOp) return failure();
  rewriter.replaceOp(getShapeOp, {combinedShapeOp});
  return success();
}

// Matches the case where the input to a GetRankedShapeOp is another
// operation. This is the primary supported case as other rewrites should
// have isolated function/block boundaries with TieShape ops.
LogicalResult rewriteInputOp(GetRankedShapeOp getShapeOp,
                             GetRankedShapeOp::Adaptor operands,
                             Operation *inputOperation,
                             PatternRewriter &rewriter) {
  // SameOperandsAndResultShape trait.
  if (inputOperation->hasTrait<OpTrait::SameOperandsAndResultShape>() ||
      inputOperation->hasTrait<OpTrait::SameOperandsAndResultType>()) {
    return rewriteSameOperandsAndResultShape(getShapeOp, inputOperation,
                                             rewriter);
  }

  // Custom shapes.
  auto customOpShapeBuilder = getCustomOpShapeBuilder();
  if (customOpShapeBuilder) {
    auto resultShape = getShapeOp.getRankedShape();
    for (auto &shapeBuilder : *customOpShapeBuilder) {
      Value customShape =
          shapeBuilder->buildRankedShape(resultShape, inputOperation, rewriter);
      if (customShape) {
        rewriter.replaceOp(getShapeOp, customShape);
        return success();
      }
    }
  }

  return failure();
}

void rewriteRuntimeShape(GetRankedShapeOp getShapeOp,
                         GetRankedShapeOp::Adaptor operands,
                         PatternRewriter &rewriter) {
  auto shapeType = getShapeOp.shape().getType().dyn_cast<RankedShapeType>();
  SmallVector<Value, 4> dynamicDims;
  for (int64_t i = 0, e = shapeType.getRank(); i < e; ++i) {
    if (!shapeType.isDimDynamic(i)) continue;
    dynamicDims.push_back(
        rewriter.create<DimOp>(getShapeOp.getLoc(), operands.operand(), i));
  }

  // TODO(laurenzo): Remove once further along (it is fine to be unsupported
  // as it will fall back to generic), but in these early phases, it is
  // extremely useful to be chatty about this fallback.
  auto inputOperation = operands.operand().getDefiningOp();
  if (inputOperation) {
    inputOperation->emitRemark()
        << "unable to materialize shape calculation (unsupported op '"
        << inputOperation->getName() << "'?): falling back to runtime "
        << "resolution";
  }

  rewriter.replaceOpWithNewOp<MakeRankedShapeOp>(getShapeOp, shapeType,
                                                 dynamicDims);
}

// Low benefit fallback pattern to materialize a ranked shape.
LogicalResult materializeRankedShapePattern(GetRankedShapeOp getShapeOp,
                                            GetRankedShapeOp::Adaptor operands,
                                            PatternRewriter &rewriter) {
  // Check for static shape and elide.
  auto operandType = operands.operand().getType().dyn_cast<RankedTensorType>();
  auto shapeType = getShapeOp.shape().getType().dyn_cast<RankedShapeType>();
  if (operandType && shapeType && operandType.hasStaticShape()) {
    rewriter.replaceOpWithNewOp<ConstRankedShapeOp>(getShapeOp, shapeType);
    return success();
  }

  // Check for input operation (unless if a small set of shape ops).
  if (auto inputOperation = operands.operand().getDefiningOp()) {
    // Materialize a shape function if possible.
    LLVM_DEBUG(llvm::dbgs() << "** SHAPE: MATERIALIZE FOR INPUT OP: "
                            << *inputOperation << "\n");
    if (!failed(
            rewriteInputOp(getShapeOp, operands, inputOperation, rewriter))) {
      return success();
    }
  }

  // Runtime fallback.
  LLVM_DEBUG(llvm::dbgs() << "** SHAPE: RUNTIME RESOLUTION\n");
  rewriteRuntimeShape(getShapeOp, operands, rewriter);
  return success();
}

// Matches a tie_shape -> get_ranked_shape pattern and resolves it statically.
// This must be a higher benefit than materializeRankedShapePattern.
LogicalResult passThroughTiedGetRankedShapePattern(
    GetRankedShapeOp getShapeOp, GetRankedShapeOp::Adaptor operands,
    PatternRewriter &rewriter) {
  // Check for input operation (unless if a small set of shape ops).
  Operation *inputOperation = operands.operand().getDefiningOp();
  if (auto tieOp = llvm::dyn_cast_or_null<TieShapeOp>(inputOperation)) {
    LLVM_DEBUG(llvm::dbgs() << "** SHAPE: PASS-THROUGH tie_shape\n");
    rewriter.replaceOp(getShapeOp, tieOp.shape());
    return success();
  }
  return failure();
}

}  // namespace

void setupMaterializeShapeCalculationsLegality(ConversionTarget &target) {
  // We explicitly want to convert these ops, eliminating them.
  target.addIllegalOp<GetRankedShapeOp>();
  target.addIllegalOp<RankedBroadcastShapeOp>();
  target.addIllegalOp<GatherExtentsOp>();
}

void populateMaterializeShapeCalculationsConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  // Fallback patterns.
  insertConversionPattern(patterns, context, expandRankedBroadcastShapePattern,
                          /*benefit=*/1);
  insertConversionPattern(patterns, context, expandGatherExtentsOp,
                          /*benefit=*/1);
  insertConversionPattern(patterns, context, materializeRankedShapePattern,
                          /*benefit=*/1);

  // High benefit patterns.
  insertConversionPattern(patterns, context,
                          passThroughTiedGetRankedShapePattern,
                          /*benefit=*/10);
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
