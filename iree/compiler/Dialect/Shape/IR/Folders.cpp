// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Utils/PatternUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

static LogicalResult elideShapeCarryingGetRankedShapePattern(
    GetRankedShapeOp op, GetRankedShapeOp::Adaptor operands,
    PatternRewriter &rewriter) {
  auto carryingOp = dyn_cast_or_null<ShapeCarryingInterface>(
      operands.operand().getDefiningOp());
  if (!carryingOp) {
    return rewriter.notifyMatchFailure(op,
                                       "no associated dynamic-shape aware op");
  }
  rewriter.replaceOp(
      op, carryingOp.buildResultValueRankedShape(operands.operand(), rewriter));
  return success();
}

static LogicalResult elideDuplicateGetRankedShapePattern(
    GetRankedShapeOp op, GetRankedShapeOp::Adaptor operands,
    PatternRewriter &rewriter) {
  // If the immediate predecessor is a GetRankedShapeOp, then this op can be
  // erased in favor of the input to the tie op.
  auto precedingGetRankedShapeOp =
      dyn_cast_or_null<GetRankedShapeOp>(operands.operand().getDefiningOp());
  if (!precedingGetRankedShapeOp) return failure();

  rewriter.replaceOp(op, precedingGetRankedShapeOp.shape());
  return success();
}

static LogicalResult elideStaticGetRankedShapePattern(
    GetRankedShapeOp op, GetRankedShapeOp::Adaptor operands,
    PatternRewriter &rewriter) {
  auto operandType = operands.operand().getType().dyn_cast<RankedTensorType>();
  auto resultShapeType = op.shape().getType().dyn_cast<RankedShapeType>();
  if (!operandType || !resultShapeType || !operandType.hasStaticShape()) {
    return failure();
  }

  rewriter.replaceOpWithNewOp<ConstRankedShapeOp>(op, resultShapeType);
  return success();
}

static LogicalResult identityMakeRankedShapePattern(
    MakeRankedShapeOp op, MakeRankedShapeOp::Adaptor operands,
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

// TODO(silvasean): Better handling of "erase unused ops for legality".
// Currently, the way that we legalize !shapex.ranked_shape into individual SSA
// values per dimension is to iteratively reduce other ops to
// shapex.ranked_dim/shapex.ranked_dims and shapex.make_ranked_shape and then
// have patterns that know how to resolve the
// shapex.ranked_dim/shapex.ranked_dims to scalar values by looking through the
// shapex.make_ranked_shape ops, with the eventual goal of not having any uses
// of the shapex.make_ranked_shape op itself, instead the main computation flow
// using the individual SSA values. This naturally produces a lot of unused
// shapex.make_ranked_shape ops which we need to delete for legality reasons.
// This pattern allows conversions to erase those ops.
static LogicalResult eraseUnusedMakeRankedShapeOp(
    MakeRankedShapeOp op, MakeRankedShapeOp::Adaptor operands,
    PatternRewriter &rewriter) {
  if (!op.getResult().use_empty())
    return rewriter.notifyMatchFailure(op, "op has uses");
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult dynamicMakeRankedShapeDimPattern(
    RankedDimOp op, RankedDimOp::Adaptor operands, PatternRewriter &rewriter) {
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

static LogicalResult elideDuplicateTieShapePattern(TieShapeOp op,
                                                   TieShapeOp::Adaptor operands,
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

// Removes tie_shape ops when the operand is produced by a shape-aware op.
static LogicalResult elideShapeCarryingOperandTieShapePattern(
    TieShapeOp op, TieShapeOp::Adaptor operands, PatternRewriter &rewriter) {
  auto definingOp = operands.operand().getDefiningOp();
  if (!definingOp) return failure();
  if (isa<TieShapeOp>(definingOp)) {
    return failure();  // ignore tie-shape handled above
  } else if (isa<ShapeCarryingInterface>(definingOp)) {
    rewriter.replaceOp(op, operands.operand());
    return success();
  } else {
    return failure();
  }
}

// Reroutes uses of tie_shape ops by ops that are shape-aware or dim ops.
static LogicalResult elideTieShapeUsagePattern(TieShapeOp op,
                                               TieShapeOp::Adaptor operands,
                                               PatternRewriter &rewriter) {
  bool didAnything = false;
  for (auto &use : llvm::make_early_inc_range(op.result().getUses())) {
    if (auto carryingOp = dyn_cast<ShapeCarryingInterface>(use.getOwner())) {
      carryingOp->setOperand(use.getOperandNumber(), operands.operand());
      didAnything = true;
    } else if (auto dimOp = dyn_cast<tensor::DimOp>(use.getOwner())) {
      auto index = dimOp.getConstantIndex();
      if (index.hasValue()) {
        rewriter.replaceOpWithNewOp<RankedDimOp>(dimOp, op.shape(),
                                                 index.getValue());
        didAnything = true;
      }
    }
  }
  return didAnything ? success() : failure();
}

//===----------------------------------------------------------------------===//
// shapex.tie_shape
//===----------------------------------------------------------------------===//

void TieShapeOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *context) {
  insertGreedyPattern(patterns, context, elideDuplicateTieShapePattern);
  insertGreedyPattern(patterns, context,
                      elideShapeCarryingOperandTieShapePattern);
  insertGreedyPattern(patterns, context, elideTieShapeUsagePattern);
}

//===----------------------------------------------------------------------===//
// shapex.get_ranked_shape
//===----------------------------------------------------------------------===//

void GetRankedShapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  insertGreedyPattern(patterns, context,
                      elideShapeCarryingGetRankedShapePattern);
  insertGreedyPattern(patterns, context, elideDuplicateGetRankedShapePattern);
  insertGreedyPattern(patterns, context, elideStaticGetRankedShapePattern);
}

//===----------------------------------------------------------------------===//
// shapex.make_ranked_shape
//===----------------------------------------------------------------------===//

void MakeRankedShapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  insertGreedyPattern(patterns, context, identityMakeRankedShapePattern);
}

//===----------------------------------------------------------------------===//
// shapex.ranked_dim
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
// Standard folding and canonicalization conversion patterns.
//===----------------------------------------------------------------------===//

// Since tie_shape ops are an identity, a pattern must exist for type conversion
// to properly propagate across the operand->result edge.
struct TieShapeTypeConversionPattern : public OpConversionPattern<TieShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TieShapeOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    TieShapeOp::Adaptor adaptor(operands);
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
  insertConversionPattern(patterns, context, eraseUnusedMakeRankedShapeOp);
  insertConversionPattern(patterns, context, dynamicMakeRankedShapeDimPattern);
  insertConversionPattern(patterns, context,
                          elideDuplicateGetRankedShapePattern);
  insertConversionPattern(patterns, context, elideDuplicateTieShapePattern);
  insertConversionPattern(patterns, context,
                          elideShapeCarryingOperandTieShapePattern);
  insertConversionPattern(patterns, context, elideTieShapeUsagePattern);
  insertConversionPattern(patterns, context,
                          elideShapeCarryingGetRankedShapePattern);
  insertConversionPattern(patterns, context, identityMakeRankedShapePattern);
  insertConversionPattern(patterns, context, elideStaticGetRankedShapePattern);
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
