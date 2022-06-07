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
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

namespace {

SmallVector<int64_t> getExpandedShape(ArrayRef<int64_t> inputShape,
                                          int64_t mSplitArity,
                                          int64_t splitDimParallel) {
  SmallVector<int64_t> ans;
  for (auto elem : llvm::enumerate(inputShape)) {
    if (elem.index() == splitDimParallel) {
      ans.push_back(mSplitArity);
      ans.push_back(inputShape[splitDimParallel] / mSplitArity);
    } else {
      ans.push_back(elem.value());
    }
  }
  return ans;
}

SmallVector<int64_t> getCollapsedShape(ArrayRef<int64_t> inputShape,
                                           int64_t mSplitArity, int64_t k,
                                           int64_t kDimOrig) {
  SmallVector<int64_t> ans;
  for (auto elem : inputShape) {
    ans.push_back(elem);
  }
  ans[kDimOrig] = k * mSplitArity;
  return ans;
}

} // namespace

FailureOr<TopkOp> mlir::iree_compiler::IREE::LinalgExt::TopkOpSplitReduction::
    returningMatchAndRewrite(iree_compiler::IREE::LinalgExt::TopkOp topkOp,
                             PatternRewriter &rewriter,
                             ControlTopkSplitReductionFn splitReductionFn,
                             linalg::LinalgTransformationFilter filter) const {
  if (failed(filter.checkAndNotify(rewriter, topkOp))) {
    return rewriter.notifyMatchFailure(topkOp, "preconditions not met");
  }
  Location loc = topkOp.getLoc();
  // Original K dimension used for the final combined reduction
  int64_t kDimOrig = topkOp.dimension();
  // For parallel topk, the dimension that we compute parallel reductions
  int64_t splitDimParallel = kDimOrig;
  // For parallel topk, the dimension that we reduce
  int64_t kDimParallel = kDimOrig + 1;
  int64_t k =
      topkOp.getResult(0).getType().cast<ShapedType>().getDimSize(kDimOrig);

  Value valuesOrig = topkOp.values();
  ShapedType valuesOrigType = valuesOrig.getType().cast<ShapedType>();
  Type valueElementType = valuesOrigType.getElementType();
  Type indicesElementType =
      topkOp.getResultTypes()[1].cast<ShapedType>().getElementType();

  // Determine if we should split the reduction (assuming there is enough work
  // to do). Only static reduction dimensions are supported at the moment.
  if (valuesOrigType.getDimSize(kDimOrig) == ShapedType::kDynamicSize) {
    return rewriter.notifyMatchFailure(
        topkOp, "cannot split dynamic dimension");
  }

  // Define the M number of splits to divide the reduction dimension
  int64_t mSplitArity = splitReductionFn(topkOp);
  if (mSplitArity < 1) {
        return rewriter.notifyMatchFailure(
        topkOp,
        "split ratio less than 1");
  }

  if (valuesOrigType.getDimSize(kDimOrig) % mSplitArity != 0) {
    return rewriter.notifyMatchFailure(
        topkOp,
        "reduction dimension must be perfectly aligned to (divisible by) the "
        "split ratio");
  }

  // Expand shape of input values for parallel processing
  // Note this assumes we're already aligned at this point
  SmallVector<int64_t> expandedShape = getExpandedShape(
      valuesOrigType.getShape(), mSplitArity, splitDimParallel);
  RankedTensorType valuesExpandedType =
      RankedTensorType::get(expandedShape, valueElementType);
  SmallVector<ReassociationIndices> reassociationIndices;
  for (int i = 0; i < valuesOrigType.getRank(); ++i) {
    if (i < splitDimParallel) {
      reassociationIndices.push_back({i});
    } else if (i == splitDimParallel) {
      reassociationIndices.push_back({i, i + 1});
    } else if (i > splitDimParallel) {
      reassociationIndices.push_back({i + 1});
    }
  }
  Value valuesExpanded = rewriter.create<tensor::ExpandShapeOp>(
      loc, valuesExpandedType, valuesOrig, reassociationIndices);

  // Expand the input indicies if they exist
  RankedTensorType indicesExpandedType;
  Value indicesExpanded;
  if (auto indicesOrig = topkOp.indices()) {
    indicesExpandedType =
        RankedTensorType::get(expandedShape, indicesElementType);
    indicesExpanded = rewriter.create<tensor::ExpandShapeOp>(
        loc, indicesExpandedType, *indicesOrig, reassociationIndices);
  }

  // Define the expanded output types
  SmallVector<int64_t> expandedResultShape = expandedShape;
  expandedResultShape[kDimParallel] = k;
  RankedTensorType outputValuesExpandedType =
      RankedTensorType::get(expandedResultShape, valueElementType);
  RankedTensorType outputIndicesExpandedType =
      RankedTensorType::get(expandedResultShape, indicesElementType);
  SmallVector<Value> dynSizes;
  for (auto en : llvm::enumerate(valuesExpandedType.getShape())) {
    if (en.value() == ShapedType::kDynamicSize) {
      dynSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, valuesExpanded, en.index()));
    }
  }

  // Initialize the expanded output values
  Value initTensorOutputValues = rewriter.create<mlir::linalg::InitTensorOp>(
      loc, dynSizes, outputValuesExpandedType.getShape(), valueElementType);
  Value initTensorOutputIndices = rewriter.create<mlir::linalg::InitTensorOp>(
      loc, dynSizes, outputIndicesExpandedType.getShape(), indicesElementType);
  // Initialize indices to positive infinity and values to negative infinity
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

  // Spit reduction topk (first phase)
  SmallVector<Value> expandedIns = {valuesExpanded};
  if (topkOp.indices()) {
    expandedIns.push_back(indicesExpanded);
  }
  auto parallelTopkOp = rewriter.create<IREE::LinalgExt::TopkOp>(
      loc,
      /* resultTypes= */
      TypeRange{outputValuesExpandedType, outputIndicesExpandedType},
      /* ins= */ expandedIns,
      /* outs= */ ValueRange{negInfTensor, posInfTensor}, kDimParallel);
  // rewriter.inlineRegionBefore(topkOp.region(), parallelTopkOp.region(),
  // parallelTopkOp.region().end());
  rewriter.cloneRegionBefore(topkOp.region(), parallelTopkOp.region(),
                             parallelTopkOp.region().end());

  // Update the output indices from the parallel TopK with the M split offsets.
  //
  Value parallelIndices = parallelTopkOp.getResult(1);
  ShapedType parallelIndicesType = parallelIndices.getType().cast<ShapedType>();
  size_t parallelIndicesRank = parallelIndicesType.getRank();
  AffineMap mapIdentity = rewriter.getMultiDimIdentityMap(parallelIndicesRank);
  SmallVector<AffineMap> indexingMaps{mapIdentity};
  SmallVector<StringRef> iterators{parallelIndicesRank, "parallel"};
  Value mSplitVal = rewriter.create<arith::ConstantIntOp>(
      loc, mSplitArity, parallelIndicesType.getElementType());
  Value updatedParallelIndices =
      rewriter
          .create<linalg::GenericOp>(
              loc, parallelIndicesType /* result type */,
              ValueRange{} /* inputs */,
              ValueRange{parallelIndices} /* outputs */, indexingMaps,
              iterators,
              [=](OpBuilder &b, Location loc, ValueRange args) {
                Value splitIndex =
                    b.create<linalg::IndexOp>(loc, splitDimParallel);
                Value splitIndexInt = b.create<arith::IndexCastOp>(
                    loc, parallelIndicesType.getElementType(), splitIndex);
                Value mOffset =
                    b.create<arith::MulIOp>(loc, mSplitVal, splitIndexInt);
                Value updatedParallelIndex =
                    b.create<arith::AddIOp>(loc, mOffset, args[0]);
                b.create<linalg::YieldOp>(loc, updatedParallelIndex);
              })
          .getResult(0);

  // Define the collapsed input shapes

  SmallVector<int64_t> collapsedShape = getCollapsedShape(
      valuesOrigType.getShape(), mSplitArity, k, kDimOrig);
  RankedTensorType valuesCollapsedType =
      RankedTensorType::get(collapsedShape, valueElementType);
  RankedTensorType indicesCollapsedType =
      RankedTensorType::get(collapsedShape, indicesElementType);

  // Collapse shapes for linear reduction
  Value valuesCollapsed = rewriter.create<tensor::CollapseShapeOp>(
      loc, valuesCollapsedType, parallelTopkOp.getResults()[0],
      reassociationIndices);
  Value indicesCollapsed = rewriter.create<tensor::CollapseShapeOp>(
      loc, indicesCollapsedType, updatedParallelIndices, reassociationIndices);

  auto reductionTopkOp = rewriter.create<IREE::LinalgExt::TopkOp>(
      loc,
      /* resultTypes= */ topkOp->getResultTypes(),
      /* ins= */ ValueRange{valuesCollapsed, indicesCollapsed},
      /* outs= */ topkOp.outputs(), kDimOrig);
  // rewriter.inlineRegionBefore(topkOp.region(), reductionTopkOp.region(),
  // reductionTopkOp.region().end());
  rewriter.cloneRegionBefore(topkOp.region(), reductionTopkOp.region(),
                             reductionTopkOp.region().end());

  rewriter.replaceOp(topkOp, reductionTopkOp.getResults());

  filter.replaceLinalgTransformationFilter(rewriter, parallelTopkOp);
  filter.replaceLinalgTransformationFilter(rewriter, reductionTopkOp);
  return reductionTopkOp;
}
