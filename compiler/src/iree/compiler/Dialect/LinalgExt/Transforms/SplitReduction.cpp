// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_TOPKSPLITREDUCTIONPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

// Marker used as attribute the depth of the split reduction transformations.
const StringLiteral kSplitReductionDepthMarker = "__split_reduction_depth__";

SmallVector<int64_t> getExpandedShape(ArrayRef<int64_t> shape,
                                      int64_t splitReductionRatio,
                                      int64_t splitDimParallel) {
  SmallVector<int64_t> ans;
  ans.reserve(shape.size() + 1);
  ans.assign(shape.begin(), shape.end());
  ans[splitDimParallel] = splitReductionRatio;
  ans.insert(std::next(ans.begin(), splitDimParallel + 1),
             shape[splitDimParallel] / splitReductionRatio);

  return ans;
}

SmallVector<int64_t> getCollapsedShape(ArrayRef<int64_t> shape,
                                       int64_t splitReductionRatio, int64_t k,
                                       int64_t targetDim) {
  SmallVector<int64_t> ans(shape.begin(), shape.end());
  ans[targetDim] = k * splitReductionRatio;
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
                                 RewriterBase &rewriter, int64_t kDimOrig,
                                 int64_t splitReductionRatio,
                                 int64_t splitReductionDepth) {
  // Determine if we should split the reduction. Requires aligned static shapes
  // and no input indicies.
  auto valuesOriginalType = topkOp.getInputType();
  if (valuesOriginalType.isDynamicDim(kDimOrig)) {
    return rewriter.notifyMatchFailure(topkOp,
                                       "cannot split dynamic dimension");
  }
  if (topkOp.getIndices() && splitReductionDepth == 0) {
    return rewriter.notifyMatchFailure(
        topkOp, "input indices aren't supported for first split");
  }
  if (splitReductionRatio <= 1) {
    return rewriter.notifyMatchFailure(topkOp, "reduction ratio <= 1");
  }
  if (valuesOriginalType.getDimSize(kDimOrig) % splitReductionRatio != 0) {
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
computeParallelTopk(Location loc, RewriterBase &rewriter,
                    iree_compiler::IREE::LinalgExt::TopkOp topkOp,
                    ArrayRef<ReassociationIndices> reassociationIndices,
                    int64_t splitReductionRatio, int64_t splitDimParallel,
                    int64_t kDimParallel, int64_t kSize) {
  Value valuesOrig = topkOp.getValues();
  auto valuesOriginalType = cast<ShapedType>(valuesOrig.getType());
  Type valueElementType = valuesOriginalType.getElementType();
  Type indicesElementType =
      cast<ShapedType>(topkOp.getResultTypes()[1]).getElementType();

  SmallVector<int64_t> expandedShape = getExpandedShape(
      valuesOriginalType.getShape(), splitReductionRatio, splitDimParallel);
  auto valuesExpandedType =
      RankedTensorType::get(expandedShape, valueElementType);

  // Expand input values shape for parallel processing
  Value valuesExpanded = rewriter.create<tensor::ExpandShapeOp>(
      loc, valuesExpandedType, valuesOrig, reassociationIndices);

  // Expand input indices shape for parallel processing if they exist
  std::optional<Value> indicesExpanded;
  if (std::optional<Value> inputIndices = topkOp.getIndices()) {
    // Type inputElementType = cast<ShapedType>(inputIndices->getType());
    Type indicesExpandedType =
        RankedTensorType::get(expandedShape, indicesElementType);
    indicesExpanded = rewriter.create<tensor::ExpandShapeOp>(
        loc, indicesExpandedType, inputIndices.value(), reassociationIndices);
  }

  // Define the expanded output types
  SmallVector<int64_t> expandedResultShape = expandedShape;
  expandedResultShape[kDimParallel] = kSize;
  auto outputValuesExpandedType =
      RankedTensorType::get(expandedResultShape, valueElementType);
  auto outputIndicesExpandedType =
      RankedTensorType::get(expandedResultShape, indicesElementType);

  // Initialize the expanded output values
  SmallVector<Value> dynSizes;
  for (auto i : llvm::seq<int64_t>(0, valuesExpandedType.getRank())) {
    if (valuesExpandedType.isDynamicDim(i)) {
      dynSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, valuesExpanded, i));
    }
  }
  Value emptyTensorOutputValues = rewriter.create<tensor::EmptyOp>(
      loc, outputValuesExpandedType.getShape(), valueElementType, dynSizes);
  Value emptyTensorOutputIndices = rewriter.create<tensor::EmptyOp>(
      loc, outputIndicesExpandedType.getShape(), indicesElementType, dynSizes);

  // Initialize indices to positive infinity and values to negative infinity
  // for a top (maxk) comparison.
  TypedAttr negInfAttr;
  if (auto intType = dyn_cast<IntegerType>(valueElementType)) {
    negInfAttr = rewriter.getIntegerAttr(
        intType, APInt::getSignedMinValue(intType.getWidth()));
  } else {
    auto negApFloat =
        APFloat::getInf(cast<FloatType>(valueElementType).getFloatSemantics(),
                        /*Negative=*/true);
    negInfAttr = rewriter.getFloatAttr(valueElementType, negApFloat);
  }
  Value negInf = rewriter.create<arith::ConstantOp>(loc, negInfAttr);
  TypedAttr posInfAttr =
      rewriter.getIntegerAttr(indicesElementType, APInt::getSignedMaxValue(32));
  Value posInf = rewriter.create<arith::ConstantOp>(loc, posInfAttr);
  Value negInfTensor =
      rewriter.create<linalg::FillOp>(loc, negInf, emptyTensorOutputValues)
          .result();
  Value posInfTensor =
      rewriter.create<linalg::FillOp>(loc, posInf, emptyTensorOutputIndices)
          .result();

  SmallVector<Type> parallelTopkResultTypes = {outputValuesExpandedType,
                                               outputIndicesExpandedType};
  SmallVector<Value> parallelTopkIns = {valuesExpanded};
  if (indicesExpanded) {
    parallelTopkIns.push_back(indicesExpanded.value());
  }
  SmallVector<Value> parallelTopkOuts = {negInfTensor, posInfTensor};

  // Parallel topk
  auto parallelTopkOp = rewriter.create<iree_compiler::IREE::LinalgExt::TopkOp>(
      loc,
      /*resultTypes=*/
      parallelTopkResultTypes,
      /*ins=*/parallelTopkIns,
      /*outs=*/parallelTopkOuts, kDimParallel);
  rewriter.cloneRegionBefore(topkOp.getRegion(), parallelTopkOp.getRegion(),
                             parallelTopkOp.getRegion().end());

  return parallelTopkOp;
}

// Update the output indices from the parallel TopK with the correct offsets.
// Each parallel computation uses implicit indices (starting from 0) during
// selection, but the values are part of the large input space split into M =
// splitReductionFn() ways. The following linalg.generic adds the appropriate
// offset to reflect to values original position. "Updated pos" = "initial
// pos" + "splitDimParallel size * "splitDimParallel index"
Value offsetParallelIndices(Location loc, RewriterBase &rewriter,
                            Value parallelIndices, int64_t kDimParallelSize,
                            int64_t splitDimParallel) {
  auto parallelIndicesType = cast<ShapedType>(parallelIndices.getType());
  size_t parallelIndicesRank = parallelIndicesType.getRank();
  AffineMap mapIdentity = rewriter.getMultiDimIdentityMap(parallelIndicesRank);
  SmallVector<AffineMap> indexingMaps = {mapIdentity};
  SmallVector<utils::IteratorType> iterators(parallelIndicesRank,
                                             utils::IteratorType::parallel);
  Value mSplitVal = rewriter.create<arith::ConstantIntOp>(
      loc, kDimParallelSize, parallelIndicesType.getElementType());
  return rewriter
      .create<linalg::GenericOp>(
          loc,
          /*resultType=*/parallelIndicesType,
          /*inputs=*/ValueRange{},
          /*outputs=*/ValueRange{parallelIndices}, indexingMaps, iterators,
          [&](OpBuilder &b, Location loc, ValueRange args) {
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
TopkOp computeReductionTopk(Location loc, RewriterBase &rewriter, TopkOp topkOp,
                            TopkOp parallelTopkOp, Value updatedParallelIndices,
                            ArrayRef<ReassociationIndices> reassociationIndices,
                            int64_t splitReductionRatio, int64_t kDimOrig,
                            int64_t kSize) {
  Value valuesOrig = topkOp.getValues();
  auto valuesOriginalType = cast<ShapedType>(valuesOrig.getType());
  Type valueElementType = valuesOriginalType.getElementType();
  Type indicesElementType =
      cast<ShapedType>(topkOp.getResultTypes()[1]).getElementType();

  // Define the collapsed input shapes
  SmallVector<int64_t> collapsedShape = getCollapsedShape(
      valuesOriginalType.getShape(), splitReductionRatio, kSize, kDimOrig);
  auto valuesCollapsedType =
      RankedTensorType::get(collapsedShape, valueElementType);
  auto indicesCollapsedType =
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
          /*resultTypes=*/topkOp->getResultTypes(),
          /*ins=*/ValueRange{valuesCollapsed, indicesCollapsed},
          /*outs=*/topkOp.getOutputs(), kDimOrig);
  rewriter.cloneRegionBefore(topkOp.getRegion(), reductionTopkOp.getRegion(),
                             reductionTopkOp.getRegion().end());
  return reductionTopkOp;
}

int64_t getSplitReductionDepth(TopkOp topkOp) {
  auto attr =
      topkOp->template getAttrOfType<IntegerAttr>(kSplitReductionDepthMarker);
  if (attr) {
    return attr.getInt();
  } else {
    return 0;
  }
}

void setSplitReductionDepth(TopkOp topkOp, RewriterBase &rewriter,
                            int64_t depth) {
  topkOp->setAttr(kSplitReductionDepthMarker,
                  rewriter.getI64IntegerAttr(depth));
}
} // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct TopkSplitReductionPass final
    : impl::TopkSplitReductionPassBase<TopkSplitReductionPass> {
  using impl::TopkSplitReductionPassBase<
      TopkSplitReductionPass>::TopkSplitReductionPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    if (splitRatios.empty()) {
      return;
    }

    TopkSplitReductionControlFn splitReductionFn =
        [&](int64_t splitReductionDepth) -> int64_t {
      SmallVector<int64_t> reductionRatios(splitRatios.begin(),
                                           splitRatios.end());
      if (splitReductionDepth >= reductionRatios.size()) {
        return -1;
      } else {
        return reductionRatios[splitReductionDepth];
      }
    };

    IRRewriter rewriter(&getContext());
    auto funcOp = getOperation();
    SmallVector<LinalgExt::TopkOp> topkCandidates;
    funcOp->walk([&](LinalgExt::TopkOp op) { topkCandidates.push_back(op); });
    for (auto op : topkCandidates) {
      (void)splitReduction(rewriter, op, splitReductionFn);
    }
  }
};
} // namespace

LogicalResult
splitReduction(RewriterBase &rewriter, LinalgExt::TopkOp topkOp,
               const TopkSplitReductionControlFn &splitReductionFn) {
  Location loc = topkOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(topkOp);
  // Original reduction dimension used for the final combined reduction
  int64_t kDimOrig = topkOp.getDimension();
  // For parallel topk: the dimension that we compute parallel reductions
  int64_t splitDimParallel = kDimOrig;
  // For parallel topk: the dimension that we reduce
  int64_t kDimParallel = kDimOrig + 1;
  int64_t kSize =
      cast<ShapedType>(topkOp.getResult(0).getType()).getDimSize(kDimOrig);
  int64_t splitReductionDepth = getSplitReductionDepth(topkOp);
  int64_t splitReductionRatio = splitReductionFn(splitReductionDepth);
  SmallVector<ReassociationIndices> reassociationIndices =
      getReassociationIndices(topkOp.getInputRank(), splitDimParallel);

  // Determine if should compute parallel topk
  LogicalResult shouldParallelTopkResult = shouldParallelTopk(
      topkOp, rewriter, kDimOrig, splitReductionRatio, splitReductionDepth);
  if (shouldParallelTopkResult.failed()) {
    return shouldParallelTopkResult;
  }

  // Topk parallel reduction
  TopkOp parallelTopkOp = computeParallelTopk(
      loc, rewriter, topkOp, reassociationIndices, splitReductionRatio,
      splitDimParallel, kDimParallel, kSize);

  // Update parallel indices to correct offsets if input indices weren't
  // provided. If input indices were provided, no offsetting is needed as
  // original original indices are already known.
  Value updatedParallelIndices = parallelTopkOp.getResult(1);
  if (!topkOp.getIndices()) {
    Value parallelIndices = parallelTopkOp.getResult(1);
    SmallVector<int64_t> expandedShape = getExpandedShape(
        cast<ShapedType>(topkOp.getValues().getType()).getShape(),
        splitReductionRatio, splitDimParallel);
    int64_t kDimParallelSize = expandedShape[kDimParallel];
    updatedParallelIndices = offsetParallelIndices(
        loc, rewriter, parallelIndices, kDimParallelSize, splitDimParallel);
  }

  // Topk final reduction
  TopkOp reductionTopkOp = computeReductionTopk(
      loc, rewriter, topkOp, parallelTopkOp, updatedParallelIndices,
      reassociationIndices, splitReductionRatio, kDimOrig, kSize);

  // Replace and update result
  rewriter.replaceOp(topkOp, reductionTopkOp.getResults());
  setSplitReductionDepth(reductionTopkOp, rewriter, splitReductionDepth + 1);

  // Recursively apply split reduction until reaching the target depth.
  if (failed(splitReduction(rewriter, reductionTopkOp, splitReductionFn))) {
    reductionTopkOp->removeAttr(LinalgExt::kSplitReductionDepthMarker);
  }

  return success();
}

struct ArgmaxCombinerOps {
  Operation *maxOp = nullptr;    // arith.maximumf
  Operation *selectOp = nullptr; // arith.select
  Operation *cmpOp = nullptr;    // arith.cmpf
};

// Matches the combiner pattern in a linalg.generic argmax-style reduction:
// Example MLIR:
// %4:2 = linalg.generic {
//     indexing_maps = [...],
//     iterator_types = ["parallel", "reduction"]
//   } ins(%arg0 : tensor<?x128xbf16>) outs(%1, %3 : tensor<?xbf16>,
//   tensor<?xi64>) {
// ^bb0(%in: bf16, %out: bf16, %out_0: i64):
//   %5 = linalg.index 1 : index
//   %6 = arith.index_cast %5 : index to i64
//   %7 = arith.maximumf %in, %out : bf16
//   %8 = arith.cmpf ogt, %in, %out : bf16
//   %9 = arith.select %8, %6, %out_0 : i64
//   linalg.yield %7, %9 : bf16, i64
// } -> (tensor<?xbf16>, tensor<?xi64>)
//
// This function extracts the `arith.maximumf`, `arith.cmpf`, and `arith.select`
// operations from the body to facilitate transformations such as split
// reduction.
static ArgmaxCombinerOps collectArgmaxCombinerOps(linalg::GenericOp genericOp) {
  assert(isArgmaxOp(genericOp) && "expected operation to be an argmax op");

  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());

  // Extract max value producer: arith.maximumf.
  Value maxResult = yieldOp.getOperand(0);
  auto maxOp = cast<arith::MaximumFOp>(maxResult.getDefiningOp());

  // Extract index result producer: arith.select.
  Value indexResult = yieldOp.getOperand(1);
  auto selectOp = cast<arith::SelectOp>(indexResult.getDefiningOp());

  // Extract the condition of the select, expected to be arith.cmpf with
  // predicate OGT.
  auto cmpOp = cast<arith::CmpFOp>(selectOp.getCondition().getDefiningOp());

  ArgmaxCombinerOps ops;
  ops.maxOp = maxOp;
  ops.selectOp = selectOp;
  ops.cmpOp = cmpOp;
  return ops;
}

static Value expandValue(OpBuilder &builder, Location loc, Value value,
                         RankedTensorType expandedType) {
  RankedTensorType originalType = cast<RankedTensorType>(value.getType());
  if (originalType == expandedType) {
    return value;
  }
  auto reassociation =
      getReassociationIndicesForReshape(originalType, expandedType);
  assert(reassociation && "failed to infer reassociation indices from types");
  return builder.create<tensor::ExpandShapeOp>(loc, expandedType, value,
                                               *reassociation);
}

// Returns an expanded input AffineMap by splitting the given reduction
// dimension into [parallel, reduction] or [reduction, parallel], inserting a
// new dimension at `insertSplitDim`.
static AffineMap getExpandedInputIndexingMap(AffineMap oldMap,
                                             unsigned reductionDim,
                                             unsigned insertSplitDim,
                                             bool innerParallel,
                                             MLIRContext *ctx) {
  SmallVector<AffineExpr> exprs;
  for (unsigned idx = 0; idx < oldMap.getNumResults(); ++idx) {
    unsigned dim = oldMap.getDimPosition(idx);
    if (dim != reductionDim) {
      unsigned shifted = (dim < insertSplitDim) ? dim : dim + 1;
      exprs.push_back(getAffineDimExpr(shifted, ctx));
      continue;
    }
    // Expand the reduction dimension into [reduction, parallel] or [parallel,
    // reduction].
    if (innerParallel) {
      exprs.push_back(
          getAffineDimExpr(dim < insertSplitDim ? dim : dim + 1, ctx));
      exprs.push_back(getAffineDimExpr(insertSplitDim, ctx));
    } else {
      exprs.push_back(getAffineDimExpr(insertSplitDim, ctx));
      exprs.push_back(
          getAffineDimExpr(dim < insertSplitDim ? dim : dim + 1, ctx));
    }
  }

  return AffineMap::get(oldMap.getNumDims() + 1, oldMap.getNumSymbols(), exprs,
                        ctx);
}

// Returns an expanded output AffineMap with a dimension inserted at
// `insertSplitDim`.
static AffineMap getExpandedOutputIndexingMap(AffineMap oldMap,
                                              unsigned insertSplitDim,
                                              MLIRContext *ctx) {
  SmallVector<AffineExpr> exprs;
  for (unsigned idx = 0; idx <= oldMap.getNumResults(); ++idx) {
    if (idx == insertSplitDim) {
      exprs.push_back(getAffineDimExpr(insertSplitDim, ctx));
    }
    if (idx < oldMap.getNumResults()) {
      unsigned dim = oldMap.getDimPosition(idx);
      unsigned shifted = (dim < insertSplitDim) ? dim : dim + 1;
      exprs.push_back(getAffineDimExpr(shifted, ctx));
    }
  }

  return AffineMap::get(oldMap.getNumDims() + 1, oldMap.getNumSymbols(), exprs,
                        ctx);
}

static Value getSplitReductionInit(OpBuilder &builder, Location loc,
                                   Value origInit, Attribute identityAttr,
                                   int64_t insertDimSize,
                                   unsigned insertSplitIndex) {
  Type elemType = getElementTypeOrSelf(origInit.getType());
  SmallVector<OpFoldResult> shape =
      tensor::getMixedSizes(builder, loc, origInit);
  shape.insert(shape.begin() + insertSplitIndex,
               builder.getIndexAttr(insertDimSize));
  Value identityVal = builder.create<arith::ConstantOp>(
      loc, elemType, cast<TypedAttr>(identityAttr));
  Value empty = builder.create<tensor::EmptyOp>(loc, shape, elemType);
  return builder.create<linalg::FillOp>(loc, identityVal, empty).getResult(0);
}

FailureOr<linalg::SplitReductionResult>
splitArgmaxReduction(RewriterBase &rewriter, linalg::GenericOp genericOp,
                     linalg::ControlSplitReductionFn controlSplitReductionFn) {
  assert(isArgmaxOp(genericOp) && "expected operation to be an argmax op");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(genericOp);
  Location loc = genericOp->getLoc();

  linalg::SplitReductionOptions control = controlSplitReductionFn(genericOp);
  if (control.innerParallel) {
    // TODO(Bangtian): Currently bail out when innerParallel is true.
    // Much of the logic is already in place based on the upstream
    // implementation, but full support requires early detection and additional
    // handling before performing the split. Will revisit when support becomes
    // necessary.
    return rewriter.notifyMatchFailure(
        genericOp, "innerParallel split is not supported yet");
  }

  int64_t ratio = control.ratio;
  unsigned insertSplitIndex = control.index;
  unsigned insertSplitDimension = control.index;
  if (ratio <= 1) {
    return rewriter.notifyMatchFailure(
        genericOp, "split ratio needs to be greater than 1");
  }

  SmallVector<unsigned> dims;
  genericOp.getReductionDims(dims);
  unsigned reductionDim = dims[0];
  if (control.innerParallel) {
    insertSplitDimension = reductionDim + 1;
  }

  SmallVector<int64_t> loopRanges = genericOp.getStaticLoopRanges();
  int64_t reductionDimSize = loopRanges[reductionDim];
  if (ShapedType::isDynamic(reductionDimSize) ||
      reductionDimSize % ratio != 0) {
    return rewriter.notifyMatchFailure(
        genericOp, "Reduction dimension not divisible by split ratio");
  }

  if (insertSplitIndex >
      genericOp.getShape(genericOp.getDpsInitOperand(0)).size()) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "Insert dimension position too large "
                                       "compared to intermediate tensor size");
  }

  ArgmaxCombinerOps combinerOps = collectArgmaxCombinerOps(genericOp);
  Operation *reductionOp = combinerOps.maxOp;
  std::optional<TypedAttr> identity = arith::getNeutralElement(reductionOp);
  if (!identity.has_value()) {
    return rewriter.notifyMatchFailure(
        genericOp, "Unknown identity value for the reduction");
  }

  SmallVector<Value> newInputs;
  for (OpOperand *operand : genericOp.getDpsInputOperands()) {
    AffineMap map = genericOp.getMatchingIndexingMap(operand);
    ArrayRef<int64_t> oldShape = genericOp.getShape(operand);
    Type elementType = getElementTypeOrSelf(operand->get().getType());
    SmallVector<int64_t> expandedShape;
    for (auto idx : llvm::seq<unsigned>(0, map.getNumResults())) {
      unsigned dim = map.getDimPosition(idx);
      if (dim != reductionDim) {
        expandedShape.push_back(oldShape[idx]);
        continue;
      }
      int64_t orig = oldShape[idx];
      int64_t outer = orig / ratio;
      if (control.innerParallel) {
        expandedShape.push_back(ratio); // reduction.
        expandedShape.push_back(outer); // parallel.
      } else {
        expandedShape.push_back(outer); // parallel.
        expandedShape.push_back(ratio); // reduction.
      }
    }

    auto expandedType = RankedTensorType::get(expandedShape, elementType);
    Value expanded = expandValue(rewriter, loc, operand->get(), expandedType);
    newInputs.push_back(expanded);
  }

  // The total number of output elements along this new dimension is
  // reductionDimSize / ratio.
  int64_t outputDimSize = reductionDimSize / ratio;
  OpOperand *valueInit = genericOp.getDpsInitOperand(0);
  OpOperand *indexInit = genericOp.getDpsInitOperand(1);
  // Value identity.
  Value identityValue =
      getSplitReductionInit(rewriter, loc, valueInit->get(), *identity,
                            outputDimSize, insertSplitIndex);
  // Index identity.
  Type indexElemType = genericOp.getRegionOutputArgs()[1].getType();
  Value identityIndex = getSplitReductionInit(
      rewriter, loc, indexInit->get(), rewriter.getZeroAttr(indexElemType),
      outputDimSize, insertSplitIndex);

  SmallVector<utils::IteratorType> newIteratorTypes =
      genericOp.getIteratorTypesArray();
  newIteratorTypes.insert(newIteratorTypes.begin() + insertSplitDimension,
                          utils::IteratorType::parallel);

  SmallVector<AffineMap> newMaps;
  unsigned numInputs = genericOp.getNumDpsInputs();
  for (OpOperand &operand : genericOp->getOpOperands()) {
    AffineMap map = genericOp.getMatchingIndexingMap(&operand);
    if (operand.getOperandNumber() < numInputs) {
      newMaps.push_back(getExpandedInputIndexingMap(
          map, reductionDim, insertSplitDimension, control.innerParallel,
          rewriter.getContext()));
    } else {
      newMaps.push_back(getExpandedOutputIndexingMap(map, insertSplitDimension,
                                                     rewriter.getContext()));
    }
  }

  // Step 1: Create a a structurally strict argmax that performs a partial
  // reduction over the split (tile) dimension. The argmax matches the pattern
  // expected by isArgmaxOp (maximumf, cmpf, select with index from
  // linalg.index). The result yields the local maximum values and their
  // corresponding local indices within each tile. These local indices will be
  // adjusted to global indices in step 2.
  auto partialArgmax = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{identityValue.getType(), identityIndex.getType()},
      newInputs, ValueRange{identityValue, identityIndex}, newMaps,
      newIteratorTypes,
      [reductionDim](OpBuilder &b, Location loc, ValueRange args) {
        Value in = args[0];
        Value outVal = args[1];
        Value outIdx = args[2];
        Value reductionIdx = b.create<linalg::IndexOp>(loc, reductionDim + 1);
        if (outIdx.getType() != reductionIdx.getType())
          reductionIdx =
              b.create<arith::IndexCastOp>(loc, outIdx.getType(), reductionIdx);
        Value maxVal = b.create<arith::MaximumFOp>(loc, in, outVal);
        Value cmp =
            b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, in, outVal);
        Value selIdx =
            b.create<arith::SelectOp>(loc, cmp, reductionIdx, outIdx);
        b.create<linalg::YieldOp>(loc, ValueRange{maxVal, selIdx});
      });

  // Step 2: Final reduction that computes global indices and selects the one
  // corresponding to the maximum value.
  Value tileSize = rewriter.create<arith::ConstantIndexOp>(loc, ratio);
  unsigned intermRank =
      cast<RankedTensorType>(identityValue.getType()).getRank();
  AffineMap valueMap = rewriter.getMultiDimIdentityMap(intermRank);
  AffineMap indexMap = valueMap;

  SmallVector<AffineExpr> resultExprs;
  SmallVector<utils::IteratorType> reductionIteratorTypes(
      intermRank, utils::IteratorType::parallel);
  reductionIteratorTypes[insertSplitIndex] = utils::IteratorType::reduction;
  for (auto i : llvm::seq<unsigned>(0, intermRank)) {
    if (i != insertSplitIndex) {
      resultExprs.push_back(rewriter.getAffineDimExpr(i));
    }
  }

  AffineMap outputMap =
      AffineMap::get(intermRank, 0, resultExprs, rewriter.getContext());
  SmallVector<AffineMap> finalReductionMaps = {valueMap, indexMap, outputMap,
                                               outputMap};
  auto finalReduction = rewriter.create<linalg::GenericOp>(
      loc, genericOp.getResultTypes(),
      ValueRange{partialArgmax.getResult(0), partialArgmax.getResult(1)},
      genericOp.getDpsInits(), finalReductionMaps, reductionIteratorTypes,
      [combinerOps, tileSize, insertSplitDimension](OpBuilder &b, Location loc,
                                                    ValueRange inputs) {
        Value val = inputs[0];
        Value local = inputs[1];
        Value outVal = inputs[2];
        Value outIdx = inputs[3];
        Value outer = b.create<linalg::IndexOp>(loc, insertSplitDimension);
        Value offset = b.create<arith::MulIOp>(loc, outer, tileSize);
        if (offset.getType() != local.getType())
          offset = b.create<arith::IndexCastOp>(loc, local.getType(), offset);
        // gidx = outer * ratio + local.
        Value gidx = b.create<arith::AddIOp>(loc, offset, local);
        Operation *clonedMax = b.clone(*combinerOps.maxOp);
        clonedMax->setOperands({val, outVal});
        Operation *clonedCmp = b.clone(*combinerOps.cmpOp);
        clonedCmp->setOperands({val, outVal});
        Operation *clonedSel = b.clone(*combinerOps.selectOp);
        clonedSel->setOperands({clonedCmp->getResult(0), gidx, outIdx});
        b.create<linalg::YieldOp>(
            loc, ValueRange{clonedMax->getResult(0), clonedSel->getResult(0)});
      });

  rewriter.replaceOp(genericOp, finalReduction.getResults());
  // Init or alloc and fillOp are not applicable for argmax op; set to nullptr.
  return linalg::SplitReductionResult{
      /*initOrAlloc=*/nullptr, /*fillOp=*/nullptr,
      cast<linalg::LinalgOp>(partialArgmax.getOperation()), finalReduction};
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
