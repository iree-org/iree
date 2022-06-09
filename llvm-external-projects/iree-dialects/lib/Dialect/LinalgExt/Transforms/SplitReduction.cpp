// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

namespace {

SmallVector<int64_t> getExpandedShape(ArrayRef<int64_t> inputShape,
                                      int64_t splitReductionRatio,
                                      int64_t splitDimParallel) {
  SmallVector<int64_t> ans;
  ans.reserve(inputShape.size() + 1);
  ans.assign(inputShape.begin(), inputShape.end());
  ans[splitDimParallel] = splitReductionRatio;
  ans.insert(std::next(ans.begin(), splitDimParallel + 1),
             inputShape[splitDimParallel] / splitReductionRatio);

  return ans;
}

SmallVector<int64_t> getCollapsedShape(ArrayRef<int64_t> inputShape,
                                       int64_t splitReductionRatio, int64_t k,
                                       int64_t kDimOrig) {
  SmallVector<int64_t> ans;
  for (auto elem : inputShape) {
    ans.push_back(elem);
  }
  ans[kDimOrig] = k * splitReductionRatio;
  return ans;
}

SmallVector<ReassociationIndices>
getReassociationIndices(int64_t rank, int64_t splitDimParallel) {
  SmallVector<ReassociationIndices> reassociationIndices;
  for (int i = 0; i < rank; ++i) {
    if (i < splitDimParallel) {
      reassociationIndices.push_back({i});
    } else if (i == splitDimParallel) {
      reassociationIndices.push_back({i, i + 1});
    } else if (i > splitDimParallel) {
      reassociationIndices.push_back({i + 1});
    }
  }
  return reassociationIndices;
}

LogicalResult shouldParallelTopk(iree_compiler::IREE::LinalgExt::TopkOp topkOp,
                                 PatternRewriter &rewriter, int64_t kDimOrig,
                                 int64_t splitReductionRatio) {
  // Determine if we should split the reduction. Requires aligned static shapes
  // and no input indicies.
  ShapedType valuesOrigType = topkOp.values().getType().cast<ShapedType>();
  if (valuesOrigType.getDimSize(kDimOrig) == ShapedType::kDynamicSize) {
    return rewriter.notifyMatchFailure(topkOp,
                                       "cannot split dynamic dimension");
  }
  if (topkOp.indices()) {
    return rewriter.notifyMatchFailure(topkOp,
                                       "input indices aren't supported");
  }
  if (splitReductionRatio <= 1) {
    return rewriter.notifyMatchFailure(topkOp, "no split from control fn");
  }
  if (valuesOrigType.getDimSize(kDimOrig) % splitReductionRatio != 0) {
    return rewriter.notifyMatchFailure(
        topkOp,
        "reduction dimension must be perfectly aligned to (divisible by) the "
        "split ratio");
  }
  return success();
}

// Creates the first phase of the topk split reduction by reshaping the input
// into parallel computations then feeding them into a topk op.
iree_compiler::IREE::LinalgExt::TopkOp
computeParallelTopk(Location loc, PatternRewriter &rewriter,
                    iree_compiler::IREE::LinalgExt::TopkOp topkOp,
                    ArrayRef<ReassociationIndices> reassociationIndices,
                    int64_t splitReductionRatio, int64_t splitDimParallel,
                    int64_t kDimParallel, int64_t kSize) {
  Value valuesOrig = topkOp.values();
  ShapedType valuesOrigType = valuesOrig.getType().cast<ShapedType>();
  Type valueElementType = valuesOrigType.getElementType();
  Type indicesElementType =
      topkOp.getResultTypes()[1].cast<ShapedType>().getElementType();

  SmallVector<int64_t> expandedShape = getExpandedShape(
      valuesOrigType.getShape(), splitReductionRatio, splitDimParallel);
  RankedTensorType valuesExpandedType =
      RankedTensorType::get(expandedShape, valueElementType);

  // Expand input values shape for parallel processing
  Value valuesExpanded = rewriter.create<tensor::ExpandShapeOp>(
      loc, valuesExpandedType, valuesOrig, reassociationIndices);

  // Define the expanded output types
  SmallVector<int64_t> expandedResultShape = expandedShape;
  expandedResultShape[kDimParallel] = kSize;
  RankedTensorType outputValuesExpandedType =
      RankedTensorType::get(expandedResultShape, valueElementType);
  RankedTensorType outputIndicesExpandedType =
      RankedTensorType::get(expandedResultShape, indicesElementType);

  // Initialize the expanded output values
  SmallVector<Value> dynSizes;
  for (auto en : llvm::enumerate(valuesExpandedType.getShape())) {
    if (en.value() == ShapedType::kDynamicSize) {
      dynSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, valuesExpanded, en.index()));
    }
  }
  Value initTensorOutputValues = rewriter.create<mlir::linalg::InitTensorOp>(
      loc, dynSizes, outputValuesExpandedType.getShape(), valueElementType);
  Value initTensorOutputIndices = rewriter.create<mlir::linalg::InitTensorOp>(
      loc, dynSizes, outputIndicesExpandedType.getShape(), indicesElementType);

  // Initialize indices to positive infinity and values to negative infinity
  // for a top (maxk) comparison.
  Attribute negInfAttr;
  if (auto intType = valueElementType.dyn_cast<IntegerType>()) {
    negInfAttr = rewriter.getIntegerAttr(
        intType, APInt::getSignedMinValue(intType.getWidth()));
  } else {
    auto negApFloat =
        APFloat::getInf(valueElementType.cast<FloatType>().getFloatSemantics(),
                        /*Negative=*/true);
    negInfAttr = rewriter.getFloatAttr(valueElementType, negApFloat);
  }
  Value negInf = rewriter.create<arith::ConstantOp>(loc, negInfAttr);
  Attribute posInfAttr =
      rewriter.getIntegerAttr(indicesElementType, APInt::getSignedMaxValue(32));
  Value posInf = rewriter.create<arith::ConstantOp>(loc, posInfAttr);
  Value negInfTensor =
      rewriter.create<linalg::FillOp>(loc, negInf, initTensorOutputValues)
          .result();
  Value posInfTensor =
      rewriter.create<linalg::FillOp>(loc, posInf, initTensorOutputIndices)
          .result();

  SmallVector<Type> parallelTopkResultTypes{outputValuesExpandedType,
                                            outputIndicesExpandedType};
  SmallVector<Value> parallelTopkIns{valuesExpanded};
  SmallVector<Value> parallelTopkOuts{negInfTensor, posInfTensor};

  // Parallel topk
  auto parallelTopkOp = rewriter.create<iree_compiler::IREE::LinalgExt::TopkOp>(
      loc,
      /* resultTypes= */
      parallelTopkResultTypes,
      /* ins= */ parallelTopkIns,
      /* outs= */ parallelTopkOuts, kDimParallel);
  rewriter.cloneRegionBefore(topkOp.region(), parallelTopkOp.region(),
                             parallelTopkOp.region().end());

  return parallelTopkOp;
}

// Update the output indices from the parallel TopK with the correct offsets.
// Each parallel computation uses implicit indices (starting from 0) during
// selection, but the values are part of the large input space split into M =
// splitReductionFn() ways. The following linalg.generic adds the appropriate
// offset to reflect to values original position. "Updated pos" = "initial
// pos" + m * "parallel computation index"
Value offsetParallelIndices(Location loc, PatternRewriter &rewriter,
                            Value parallelIndices, int64_t splitReductionRatio,
                            int64_t splitDimParallel) {
  ShapedType parallelIndicesType = parallelIndices.getType().cast<ShapedType>();
  size_t parallelIndicesRank = parallelIndicesType.getRank();
  AffineMap mapIdentity = rewriter.getMultiDimIdentityMap(parallelIndicesRank);
  SmallVector<AffineMap> indexingMaps{mapIdentity};
  SmallVector<StringRef> iterators{parallelIndicesRank, "parallel"};
  Value mSplitVal = rewriter.create<arith::ConstantIntOp>(
      loc, splitReductionRatio, parallelIndicesType.getElementType());
  return rewriter
      .create<linalg::GenericOp>(
          loc,
          /* resultType= */ parallelIndicesType,
          /* inputs= */ ValueRange{},
          /* outputs= */ ValueRange{parallelIndices}, indexingMaps, iterators,
          [=](OpBuilder &b, Location loc, ValueRange args) {
            Value splitIndex = b.create<linalg::IndexOp>(loc, splitDimParallel);
            Value splitIndexInt = b.create<arith::IndexCastOp>(
                loc, parallelIndicesType.getElementType(), splitIndex);
            Value mOffset =
                b.create<arith::MulIOp>(loc, mSplitVal, splitIndexInt);
            Value updatedParallelIndex =
                b.create<arith::AddIOp>(loc, mOffset, args[0]);
            b.create<linalg::YieldOp>(loc, updatedParallelIndex);
          })
      .getResult(0);
}

// Creates the second phase of the topk split reduction by collapsing output
// from parallel topk and computing the final combined result.
iree_compiler::IREE::LinalgExt::TopkOp
computeReductionTopk(Location loc, PatternRewriter &rewriter,
                     iree_compiler::IREE::LinalgExt::TopkOp topkOp,
                     iree_compiler::IREE::LinalgExt::TopkOp parallelTopkOp,
                     Value updatedParallelIndices,
                     ArrayRef<ReassociationIndices> reassociationIndices,
                     int64_t splitReductionRatio, int64_t kDimOrig,
                     int64_t kSize) {
  Value valuesOrig = topkOp.values();
  ShapedType valuesOrigType = valuesOrig.getType().cast<ShapedType>();
  Type valueElementType = valuesOrigType.getElementType();
  Type indicesElementType =
      topkOp.getResultTypes()[1].cast<ShapedType>().getElementType();

  // Define the collapsed input shapes
  SmallVector<int64_t> collapsedShape = getCollapsedShape(
      valuesOrigType.getShape(), splitReductionRatio, kSize, kDimOrig);
  RankedTensorType valuesCollapsedType =
      RankedTensorType::get(collapsedShape, valueElementType);
  RankedTensorType indicesCollapsedType =
      RankedTensorType::get(collapsedShape, indicesElementType);

  // Collapse collapse parallel output for the input of final reduction
  Value valuesCollapsed = rewriter.create<tensor::CollapseShapeOp>(
      loc, valuesCollapsedType, parallelTopkOp.getResults()[0],
      reassociationIndices);
  Value indicesCollapsed = rewriter.create<tensor::CollapseShapeOp>(
      loc, indicesCollapsedType, updatedParallelIndices, reassociationIndices);

  // Combined final topk
  auto reductionTopkOp =
      rewriter.create<iree_compiler::IREE::LinalgExt::TopkOp>(
          loc,
          /* resultTypes= */ topkOp->getResultTypes(),
          /* ins= */ ValueRange{valuesCollapsed, indicesCollapsed},
          /* outs= */ topkOp.outputs(), kDimOrig);
  rewriter.cloneRegionBefore(topkOp.region(), reductionTopkOp.region(),
                             reductionTopkOp.region().end());
  return reductionTopkOp;
}

} // namespace

// Transforms an applicable standard single reduction TopkOp into a parallel
// reduction TopkOp with a reduce step following.
//
// Handles parallel reductions in 2 phases: A "map" parallel phase and the a
// single "reduce" reduction phase. The first phase expands the input tensor
// shape by breaking the reduction dimension into multiple parallel reductions
// (upping the rank of the input). Topk is run on these dimensions in parallel
// The second phase collapses the parallel results into a single final reduce.
// Topk is run again on the combined output to produce a final output.
//
// Currently only topk operations without input indices are supported.
FailureOr<TopkOp> mlir::iree_compiler::IREE::LinalgExt::TopkOpSplitReduction::
    returningMatchAndRewrite(iree_compiler::IREE::LinalgExt::TopkOp topkOp,
                             PatternRewriter &rewriter,
                             TopkSplitReductionControlFn splitReductionFn,
                             linalg::LinalgTransformationFilter filter) const {
  if (failed(filter.checkAndNotify(rewriter, topkOp))) {
    return rewriter.notifyMatchFailure(topkOp, "preconditions not met");
  }
  Location loc = topkOp.getLoc();
  // Original reduction dimension used for the final combined reduction
  int64_t kDimOrig = topkOp.dimension();
  // For parallel topk: the dimension that we compute parallel reductions
  int64_t splitDimParallel = kDimOrig;
  // For parallel topk: the dimension that we reduce
  int64_t kDimParallel = kDimOrig + 1;
  int64_t kSize =
      topkOp.getResult(0).getType().cast<ShapedType>().getDimSize(kDimOrig);
  int64_t splitReductionRatio = splitReductionFn(topkOp);
  SmallVector<ReassociationIndices> reassociationIndices =
      getReassociationIndices(
          topkOp.values().getType().cast<ShapedType>().getRank(),
          splitDimParallel);

  // Determine if should compute parallel topk
  LogicalResult shouldParallelTopkResult =
      shouldParallelTopk(topkOp, rewriter, kDimOrig, splitReductionRatio);
  if (shouldParallelTopkResult.failed()) {
    return shouldParallelTopkResult;
  }

  // Topk parallel reduction
  IREE::LinalgExt::TopkOp parallelTopkOp = computeParallelTopk(
      loc, rewriter, topkOp, reassociationIndices, splitReductionRatio,
      splitDimParallel, kDimParallel, kSize);

  // Update parallel indices to correct offsets
  Value parallelIndices = parallelTopkOp.getResult(1);
  Value updatedParallelIndices = offsetParallelIndices(
      loc, rewriter, parallelIndices, splitReductionRatio, splitDimParallel);

  // Topk final reduction
  IREE::LinalgExt::TopkOp reductionTopkOp = computeReductionTopk(
      loc, rewriter, topkOp, parallelTopkOp, updatedParallelIndices,
      reassociationIndices, splitReductionRatio, kDimOrig, kSize);

  // Replace and update result
  rewriter.replaceOp(topkOp, reductionTopkOp.getResults());
  filter.replaceLinalgTransformationFilter(rewriter, parallelTopkOp);
  filter.replaceLinalgTransformationFilter(rewriter, reductionTopkOp);
  return reductionTopkOp;
}
