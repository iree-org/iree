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
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Utils/PatternUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

LogicalResult safeCastCompatibleShapePattern(
    CastCompatibleShapeOp op, CastCompatibleShapeOpOperandAdaptor operands,
    PatternRewriter &rewriter) {
  // TODO(laurenzo): This is just eliding if everything is the same. Make
  // it generic.
  auto resultRs = op.result().getType().dyn_cast<RankedShapeType>();
  if (resultRs) {
    // Casting to a ranked shape.
    for (auto operand : operands.operands()) {
      auto operandType = operand.getType();
      auto operandRs = operandType.dyn_cast<RankedShapeType>();
      if (!operandRs || operandRs != resultRs) {
        return failure();
      }
    }
    rewriter.replaceOp(op, operands.operands()[0]);
    return success();
  }

  return failure();
}

LogicalResult elideTiedGetRankedShapePattern(
    GetRankedShapeOp op, GetRankedShapeOpOperandAdaptor operands,
    PatternRewriter &rewriter) {
  // If the immediate predecessor is a TieShapeOp, then this op can be
  // erased in favor of the input to the tie op.
  auto tieOp = dyn_cast_or_null<TieShapeOp>(operands.operand().getDefiningOp());
  if (!tieOp) return failure();

  rewriter.replaceOp(op, tieOp.shape());
  return success();
}

LogicalResult elideDuplicateGetRankedShapePattern(
    GetRankedShapeOp op, GetRankedShapeOpOperandAdaptor operands,
    PatternRewriter &rewriter) {
  // If the immediate predecessor is a GetRankedShapeOp, then this op can be
  // erased in favor of the input to the tie op.
  auto precedingGetRankedShapeOp =
      dyn_cast_or_null<GetRankedShapeOp>(operands.operand().getDefiningOp());
  if (!precedingGetRankedShapeOp) return failure();

  rewriter.replaceOp(op, precedingGetRankedShapeOp.shape());
  return success();
}

LogicalResult elideStaticGetRankedShapePattern(
    GetRankedShapeOp op, GetRankedShapeOpOperandAdaptor operands,
    PatternRewriter &rewriter) {
  auto operandType = operands.operand().getType().dyn_cast<RankedTensorType>();
  auto resultShapeType = op.shape().getType().dyn_cast<RankedShapeType>();
  if (!operandType || !resultShapeType || !operandType.hasStaticShape()) {
    return failure();
  }

  rewriter.replaceOpWithNewOp<ConstRankedShapeOp>(op, resultShapeType);
  return success();
}

LogicalResult identityMakeRankedShapePattern(
    MakeRankedShapeOp op, MakeRankedShapeOpOperandAdaptor operands,
    PatternRewriter &rewriter) {
  if (operands.dynamic_dimensions().empty()) {
    // Do not match static shapes.
    return failure();
  }

  // Detects make_ranked_shape ops whose dynamic dimensions are provided by
  // ranked_dim ops that extract dimensions from an identical ranked_shape.
  auto rankedShape = op.getRankedShapeType();
  RankedDimOp commonRankedDimOp;
  unsigned previousProvidingIndex = 0;
  for (auto providingDim : operands.dynamic_dimensions()) {
    auto rankedDimOp =
        llvm::dyn_cast_or_null<RankedDimOp>(providingDim.getDefiningOp());
    if (!rankedDimOp) return failure();

    // Shapes must match and refer to a dynamic index.
    unsigned providingIndex = rankedDimOp.getIndex();
    if (rankedDimOp.getRankedShapeType() != rankedShape ||
        !rankedShape.isDimDynamic(providingIndex)) {
      return failure();
    }

    if (commonRankedDimOp) {
      // Not first dim: verify same providing shape and indexes into next
      // dynamic dim.
      if (rankedDimOp.shape() != commonRankedDimOp.shape() ||
          providingIndex <= previousProvidingIndex) {
        return failure();
      }
    }

    commonRankedDimOp = rankedDimOp;
    previousProvidingIndex = rankedDimOp.getIndex();
  }

  // Fall-through: this op produces an identical shape as
  // commonRankedDimOp.
  assert(commonRankedDimOp &&
         "dynamic ranked_shape did not find a common provider");

  rewriter.replaceOp(op, commonRankedDimOp.shape());
  return success();
}

LogicalResult dynamicMakeRankedShapeDimPattern(
    RankedDimOp op, RankedDimOpOperandAdaptor operands,
    PatternRewriter &rewriter) {
  // If the immediate predecessor is a MakeRankedShapeOp, then this op can be
  // erased in favor of the corresponding input to that op.
  auto shapeInput = operands.shape();
  auto makeRsOp =
      dyn_cast_or_null<MakeRankedShapeOp>(shapeInput.getDefiningOp());
  if (!makeRsOp) return failure();

  RankedShapeType rsType = shapeInput.getType().cast<RankedShapeType>();
  unsigned index = op.getIndex();
  auto allDims = rsType.getAllDims();
  assert(index < allDims.size());
  if (allDims[index] >= 0) {
    // Not dynamic.
    return failure();
  }

  // Map the overall index to the dynamic dim index.
  int dynamicDimIndex = 0;
  for (unsigned i = 0; i < index; ++i) {
    if (allDims[i] < 0) dynamicDimIndex++;
  }

  assert(dynamicDimIndex < makeRsOp.dynamic_dimensions().size());
  rewriter.replaceOp(op, makeRsOp.dynamic_dimensions()[dynamicDimIndex]);
  return success();
}

LogicalResult expandRankedShapeDimsPattern(RankedDimsOp op,
                                           RankedDimsOpOperandAdaptor operands,
                                           PatternRewriter &rewriter) {
  auto shapeInput = operands.shape();
  auto rsType = shapeInput.getType().cast<RankedShapeType>();
  SmallVector<Value, 4> dims(rsType.getRank());
  for (int i = 0; i < rsType.getRank(); ++i) {
    dims[i] = rewriter.createOrFold<RankedDimOp>(
        op.getLoc(), op.getResult(i).getType(), shapeInput, i);
  }
  rewriter.replaceOp(op, dims);
  return success();
}

LogicalResult elideDuplicateTieShapePattern(TieShapeOp op,
                                            TieShapeOpOperandAdaptor operands,
                                            PatternRewriter &rewriter) {
  // If the immediate predecessor is a TieShapeOp, then it can be possible
  // to merge these. This can often happen when function/block tie_shape
  // placeholders are inserted prior to materializing later parts of the
  // computation.
  auto precedingTieShapeOp =
      dyn_cast_or_null<TieShapeOp>(operands.operand().getDefiningOp());
  if (!precedingTieShapeOp) return failure();

  if (operands.shape() != precedingTieShapeOp.shape()) {
    // This can happen in intermediate states before all shape calculations
    // are collapsed (i.e. the shapes may actually be equivalent but
    // constructed through different branches).
    return failure();
  }

  rewriter.replaceOp(op, precedingTieShapeOp.result());
  return success();
}

//===----------------------------------------------------------------------===//
// shape.tie_shape
//===----------------------------------------------------------------------===//

void TieShapeOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *context) {
  insertGreedyPattern(patterns, context, elideDuplicateTieShapePattern);
}

//===----------------------------------------------------------------------===//
// shape.cast_compatible_shape
//===----------------------------------------------------------------------===//

void CastCompatibleShapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  insertGreedyPattern(patterns, context, safeCastCompatibleShapePattern);
}

//===----------------------------------------------------------------------===//
// shape.get_ranked_shape
//===----------------------------------------------------------------------===//

void GetRankedShapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  insertGreedyPattern(patterns, context, elideTiedGetRankedShapePattern);
  insertGreedyPattern(patterns, context, elideDuplicateGetRankedShapePattern);
  insertGreedyPattern(patterns, context, elideStaticGetRankedShapePattern);
}

//===----------------------------------------------------------------------===//
// shape.make_ranked_shape
//===----------------------------------------------------------------------===//

void MakeRankedShapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  insertGreedyPattern(patterns, context, identityMakeRankedShapePattern);
}

//===----------------------------------------------------------------------===//
// shape.ranked_dim
//===----------------------------------------------------------------------===//

OpFoldResult RankedDimOp::fold(ArrayRef<Attribute> operand) {
  auto rsType = shape().getType().cast<RankedShapeType>();
  int index = getIndex();
  if (!rsType.isDimDynamic(index)) {
    auto dimSize = rsType.getStaticDim(index);
    return IntegerAttr::get(getType(), dimSize);
  }
  return {};
}

void RankedDimOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  insertGreedyPattern(patterns, context, dynamicMakeRankedShapeDimPattern);
}

//===----------------------------------------------------------------------===//
// shape.ranked_dims
//===----------------------------------------------------------------------===//

void RankedDimsOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  insertGreedyPattern(patterns, context, expandRankedShapeDimsPattern);
}

//===----------------------------------------------------------------------===//
// Standard folding and canonicalization conversion patterns.
//===----------------------------------------------------------------------===//

// Since tie_shape ops are an identity, a pattern must exist for type conversion
// to properly propagate across the operand->result edge.
struct TieShapeTypeConversionPattern : public OpConversionPattern<TieShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TieShapeOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    TieShapeOpOperandAdaptor adaptor(operands);
    Type operandType = adaptor.operand().getType();
    if (operandType == srcOp.getType()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TieShapeOp>(srcOp, operandType,
                                            adaptor.operand(), adaptor.shape());
    return success();
  }
};

void populateFoldConversionPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns) {
  patterns.insert<TieShapeTypeConversionPattern>(context);
  insertConversionPattern(patterns, context, dynamicMakeRankedShapeDimPattern);
  insertConversionPattern(patterns, context,
                          elideDuplicateGetRankedShapePattern);
  insertConversionPattern(patterns, context, elideDuplicateTieShapePattern);
  insertConversionPattern(patterns, context, elideTiedGetRankedShapePattern);
  insertConversionPattern(patterns, context, expandRankedShapeDimsPattern);
  insertConversionPattern(patterns, context, identityMakeRankedShapePattern);
  insertConversionPattern(patterns, context, elideStaticGetRankedShapePattern);
  insertConversionPattern(patterns, context, safeCastCompatibleShapePattern);
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
