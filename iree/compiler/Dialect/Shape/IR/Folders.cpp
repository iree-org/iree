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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

void populateFoldConversionPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns) {
  insertConversionPattern(patterns, context, eraseUnusedMakeRankedShapeOp);
  insertConversionPattern(patterns, context, dynamicMakeRankedShapeDimPattern);
  insertConversionPattern(patterns, context, identityMakeRankedShapePattern);
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
