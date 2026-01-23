// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/ShardToFlow/Patterns.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Utils/Permutation.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Shard/Transforms/Simplifications.h"
#include "mlir/Dialect/Shard/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::Flow {

static CollectiveReductionOp
convertReductionKind(shard::ReductionKind reduction) {
  switch (reduction) {
  case shard::ReductionKind::Max:
    return CollectiveReductionOp::ReductionMaximum;
  case shard::ReductionKind::Min:
    return CollectiveReductionOp::ReductionMinimum;
  case shard::ReductionKind::Sum:
    return CollectiveReductionOp::ReductionSum;
  default:
    assert(false);
    return CollectiveReductionOp::None;
  }
}

static CollectiveReductionOpAttr
convertReductionKind(shard::ReductionKindAttr reduction) {
  return CollectiveReductionOpAttr::get(
      reduction.getContext(), convertReductionKind(reduction.getValue()));
}

static TypedValue<RankedTensorType>
buildTranspose(Value v, ArrayRef<int64_t> transposeVector,
               ImplicitLocOpBuilder &builder) {
  RankedTensorType type = cast<RankedTensorType>(v.getType());
  SmallVector<int64_t> transposedShape =
      permute(type.getShape(), transposeVector);
  Value target =
      tensor::EmptyOp::create(builder, transposedShape, type.getElementType());
  return cast<TypedValue<RankedTensorType>>(
      linalg::TransposeOp::create(builder, v, target, transposeVector)
          ->getResult(0));
}

static SmallVector<int64_t> transpose(ArrayRef<int64_t> shape, int64_t axisA,
                                      int64_t axisB) {
  SmallVector<int64_t> res = llvm::to_vector(shape);
  std::swap(res[axisA], res[axisB]);
  return res;
}

static RankedTensorType transpose(RankedTensorType type, int64_t axisA,
                                  int64_t axisB) {
  SmallVector<int64_t> newShape = transpose(type.getShape(), axisA, axisB);
  return type.clone(newShape);
}

static TypedValue<RankedTensorType>
buildTranspose(Value v, int64_t axisA, int64_t axisB,
               ImplicitLocOpBuilder &builder) {
  int64_t rank = cast<RankedTensorType>(v.getType()).getRank();
  SmallVector<int64_t> transposeVector(rank);
  std::iota(transposeVector.begin(), transposeVector.end(), 0);
  std::swap(transposeVector[axisA], transposeVector[axisB]);
  return buildTranspose(v, transposeVector, builder);
}

// (..., splitAxisSize, ...) ->
// (..., splitCount, splitAxisSize / splitCount, ...)
static TypedValue<RankedTensorType>
splitAxis(TypedValue<RankedTensorType> tensor, int64_t splitAxis,
          int64_t splitCount, ImplicitLocOpBuilder &builder) {
  ArrayRef<int64_t> shape = tensor.getType().getShape();
  SmallVector<int64_t> newShape;
  newShape.reserve(shape.size() + 1);
  for (int64_t i = 0; i < tensor.getType().getRank(); ++i) {
    if (i != splitAxis) {
      newShape.push_back(shape[i]);
      continue;
    }
    newShape.push_back(splitCount);
    if (ShapedType::isDynamic(shape[i])) {
      newShape.push_back(ShapedType::kDynamic);
    } else {
      assert(shape[i] % splitCount == 0);
      newShape.push_back(shape[i] / splitCount);
    }
  }

  RankedTensorType resultType = tensor.getType().clone(newShape);
  std::optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForReshape(tensor.getType(), resultType);
  return cast<TypedValue<RankedTensorType>>(
      tensor::ExpandShapeOp::create(builder, resultType, tensor,
                                    reassociation.value())
          .getResult());
}

// Transposes the input tensor by moving an axis to a new position by inserting
// it there.
static TypedValue<RankedTensorType>
moveAxis(TypedValue<RankedTensorType> tensor, int64_t axis, int64_t destination,
         ImplicitLocOpBuilder &builder) {
  SmallVector<int64_t> permutation =
      makeMovePermutation(tensor.getType().getRank(), axis, destination);
  return buildTranspose(tensor, permutation, builder);
}

static SmallVector<int64_t> collapseAxesN(ArrayRef<int64_t> shape,
                                          size_t firstAxis, size_t n) {
  assert(firstAxis + n <= shape.size());
  assert(n > 1);
  auto res = llvm::to_vector_of<int64_t>(shape.take_front(firstAxis));
  int64_t collapsedAxisSize = llvm::product_of(shape.slice(firstAxis, n));
  res.push_back(collapsedAxisSize);
  llvm::append_range(res, shape.drop_front(firstAxis + n));
  return res;
}

// Collapses `n` axes starting with axis `firstAxis`.
// Example:
// tensor shape = (1, 2, 3, 4), firstAxis = 1, n = 2
// The resulting tensor is with shape (1, 6, 4).
static TypedValue<RankedTensorType>
collapseAxesN(TypedValue<RankedTensorType> tensor, int64_t firstAxis, int64_t n,
              ImplicitLocOpBuilder &builder) {
  ArrayRef<int64_t> shape = tensor.getType().getShape();
  SmallVector<int64_t> newShape = collapseAxesN(shape, firstAxis, n);
  std::optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForCollapse(shape, newShape);
  return cast<TypedValue<RankedTensorType>>(
      tensor::CollapseShapeOp::create(builder, tensor, reassociation.value())
          .getResult());
}

// Splits an axis into 2 new dimensions and then move the new splitCount axis
// and collapse it into collapseAxis.
// The shape of the tensor and its transformations:
// (..., splitAxisSize, ..., collapseAxisSize, ...)
// -> split ->
// (..., splitCount, splitAxisSize / splitCount, ..., collapseAxisSize, ...)
// -> move ->
// (..., splitAxisSize / splitCount, ..., splitCount, collapseAxisSize, ...)
// -> concat ->
// (..., splitAxisSize / splitCount, ..., splitCount * collapseAxisSize, ...)
static TypedValue<RankedTensorType>
splitMoveCollapse(TypedValue<RankedTensorType> tensor, int64_t splitAxis,
                  int64_t collapseAxis, int64_t splitCount,
                  ImplicitLocOpBuilder &builder) {
  TypedValue<RankedTensorType> v =
      IREE::Flow::splitAxis(tensor, splitAxis, splitCount, builder);
  v = moveAxis(v, splitAxis, collapseAxis, builder);
  return collapseAxesN(v, collapseAxis, 2, builder);
}

namespace {

template <typename Op>
struct ShardToFlowCollectiveRewritePatternBase : OpRewritePattern<Op> {
  template <typename... OpRewritePatternArgs>
  ShardToFlowCollectiveRewritePatternBase(
      SymbolTableCollection &symbolTableCollection,
      LookupShardChannelFn lookupChannel,
      OpRewritePatternArgs &&...opRewritePatternArgs)
      : OpRewritePattern<Op>(
            std::forward<OpRewritePatternArgs...>(opRewritePatternArgs)...),
        symbolTableCollection(symbolTableCollection),
        lookupChannel(lookupChannel) {}

protected:
  // The !flow.channel corresponding to the shard and shard axes used in the op.
  template <typename ShardCollectiveOp>
  Value buildCachedChannelLoading(ShardCollectiveOp op,
                                  ImplicitLocOpBuilder &builder) const {
    shard::GridOp shard = shard::getGrid(op, symbolTableCollection);
    return lookupChannel(builder.getLoc(), shard, op.getGridAxes(), builder);
  }

  Value buildCachedChannelLoading(shard::ProcessLinearIndexOp op,
                                  ImplicitLocOpBuilder &builder) const {
    shard::GridOp shard = shard::getGrid(op, symbolTableCollection);
    return lookupChannel(builder.getLoc(), shard, std::nullopt, builder);
  }

  SymbolTableCollection &symbolTableCollection;
  LookupShardChannelFn lookupChannel;
};

struct ShardAllReduceToFlow
    : ShardToFlowCollectiveRewritePatternBase<shard::AllReduceOp> {
  using ShardToFlowCollectiveRewritePatternBase::
      ShardToFlowCollectiveRewritePatternBase;
  LogicalResult matchAndRewrite(shard::AllReduceOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());
    Value channel = buildCachedChannelLoading(op, builder);
    RankedTensorType resultType =
        cast<RankedTensorType>(op.getOperand().getType());
    Value target = tensor::EmptyOp::create(builder, resultType.getShape(),
                                           resultType.getElementType());
    auto flowAllReduce = IREE::Flow::CollectiveAllReduceOp::create(
        builder, convertReductionKind(op.getReductionAttr()),
        getCollectiveElementTypeAttr(resultType), target, op.getOperand(),
        channel);
    rewriter.replaceAllUsesWith(op.getResult(), flowAllReduce.getResult());
    rewriter.eraseOp(op.getOperation());
    return success();
  }
};

struct ShardAllGatherToFlow
    : ShardToFlowCollectiveRewritePatternBase<shard::AllGatherOp> {
  using ShardToFlowCollectiveRewritePatternBase::
      ShardToFlowCollectiveRewritePatternBase;
  LogicalResult matchAndRewrite(shard::AllGatherOp op,
                                PatternRewriter &rewriter) const override {
    if (!cast<RankedTensorType>(op.getOperand().getType()).hasStaticShape() ||
        !cast<ShapedType>(op.getResult().getType()).hasStaticShape()) {
      // TODO: add dynamic support.
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "Dynamic tensor case is unsupported.");
    }

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());
    Value channel = buildCachedChannelLoading(op, builder);

    int64_t gatherAxis = op.getGatherAxis().getSExtValue();

    // When gather axis != 0, we need to transpose between 0 and
    // gather axis before and after the flow all-gather op.
    Value flowAllGatherOperand =
        buildTranspose(op.getOperand(), 0, gatherAxis, builder);

    RankedTensorType flowAllGatherResultType = transpose(
        cast<RankedTensorType>(op.getResult().getType()), 0, gatherAxis);
    Value target = tensor::EmptyOp::create(
        builder, flowAllGatherResultType.getShape(),
        cast<ShapedType>(op.getResult().getType()).getElementType());
    auto flowAllGather = IREE::Flow::CollectiveAllGatherOp::create(
        builder, getCollectiveElementTypeAttr(flowAllGatherResultType), target,
        flowAllGatherOperand, channel);

    Value res = buildTranspose(flowAllGather, 0, gatherAxis, builder);

    rewriter.replaceAllUsesWith(op.getResult(), res);
    rewriter.eraseOp(op.getOperation());
    return success();
  }
};

struct ShardAllToAllToFlow
    : ShardToFlowCollectiveRewritePatternBase<shard::AllToAllOp> {
  using ShardToFlowCollectiveRewritePatternBase::
      ShardToFlowCollectiveRewritePatternBase;
  LogicalResult matchAndRewrite(shard::AllToAllOp op,
                                PatternRewriter &rewriter) const override {
    if (!cast<RankedTensorType>(op.getOperand().getType()).hasStaticShape() ||
        !op.getResult().getType().hasStaticShape()) {
      // TODO: add dynamic support.
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "Dynamic tensor case is unsupported.");
    }

    shard::GridOp shard = shard::getGrid(op, symbolTableCollection);
    assert(ShapedType::isStaticShape(shard.getShape()));
    int64_t splitCount =
        shard::collectiveProcessGroupSize(op.getGridAxes(), shard.getShape());
    // TODO: handle dynamic case.
    if (ShapedType::isDynamic(splitCount)) {
      // TODO: add dynamic support.
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "Dynamic split count induced by a dynamic shard is unsupported.");
    }

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());

    Value channel = buildCachedChannelLoading(op, builder);

    int64_t splitAxis = op.getSplitAxis().getSExtValue();

    TypedValue<RankedTensorType> splitAxisAsMostOuter =
        buildTranspose(op.getOperand(), 0, splitAxis, builder);

    Value target = tensor::EmptyOp::create(
        builder, splitAxisAsMostOuter.getType().getShape(),
        splitAxisAsMostOuter.getType().getElementType());
    auto flowAllToAll = IREE::Flow::CollectiveAllToAllOp::create(
        builder, getCollectiveElementTypeAttr(splitAxisAsMostOuter.getType()),
        target, splitAxisAsMostOuter, channel);

    TypedValue<RankedTensorType> splitAxisBackInItsPlace =
        buildTranspose(flowAllToAll, 0, splitAxis, builder);

    int64_t concatAxis = op.getConcatAxis().getSExtValue();
    Value res = splitMoveCollapse(splitAxisBackInItsPlace, splitAxis,
                                  concatAxis, splitCount, builder);

    rewriter.replaceAllUsesWith(op.getResult(), res);
    rewriter.eraseOp(op.getOperation());
    return success();
  }
};

struct ShardProcessLinearIndexToFlow
    : ShardToFlowCollectiveRewritePatternBase<shard::ProcessLinearIndexOp> {
  using ShardToFlowCollectiveRewritePatternBase::
      ShardToFlowCollectiveRewritePatternBase;
  LogicalResult matchAndRewrite(shard::ProcessLinearIndexOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());
    Value channel = buildCachedChannelLoading(op, builder);
    Value newIndex = IREE::Flow::ChannelRankOp::create(
        builder, builder.getIndexType(), channel);
    rewriter.replaceAllUsesWith(op.getResult(), newIndex);
    return success();
  }
};

struct ShardReduceScatterToFlow
    : ShardToFlowCollectiveRewritePatternBase<shard::ReduceScatterOp> {
  using ShardToFlowCollectiveRewritePatternBase::
      ShardToFlowCollectiveRewritePatternBase;
  LogicalResult matchAndRewrite(shard::ReduceScatterOp op,
                                PatternRewriter &rewriter) const override {
    if (!cast<RankedTensorType>(op.getOperand().getType()).hasStaticShape() ||
        !op.getResult().getType().hasStaticShape()) {
      // TODO: add dynamic support.
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "Dynamic tensor case is unsupported.");
    }

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());
    Value channel = buildCachedChannelLoading(op, builder);

    int64_t scatterAxis = op.getScatterAxis().getSExtValue();

    // When scatter axis != 0, we need to transpose between 0 and
    // scatter axis before and after the flow reduce-scatter op.
    Value flowReduceScatterOperand =
        buildTranspose(op.getOperand(), 0, scatterAxis, builder);
    RankedTensorType flowReduceScatterResultType = transpose(
        cast<RankedTensorType>(op.getResult().getType()), 0, scatterAxis);

    Value target =
        tensor::EmptyOp::create(builder, flowReduceScatterResultType.getShape(),
                                op.getResult().getType().getElementType());
    auto flowReduceScatter = IREE::Flow::CollectiveReduceScatterOp::create(
        builder, convertReductionKind(op.getReductionAttr()),
        getCollectiveElementTypeAttr(flowReduceScatterResultType), target,
        flowReduceScatterOperand, channel);

    Value res = buildTranspose(flowReduceScatter, 0, scatterAxis, builder);

    rewriter.replaceAllUsesWith(op.getResult(), res);
    rewriter.eraseOp(op.getOperation());
    return success();
  }
};

} // namespace

void populateShardToFlowCollectivesPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection,
    LookupShardChannelFn lookupChannel) {
  patterns.add<ShardAllGatherToFlow, ShardAllReduceToFlow, ShardAllToAllToFlow,
               ShardReduceScatterToFlow, ShardProcessLinearIndexToFlow>(
      symbolTableCollection, lookupChannel, patterns.getContext());
  shard::populateFoldingPatterns(patterns, symbolTableCollection);
  shard::populateProcessMultiIndexOpLoweringPatterns(patterns,
                                                     symbolTableCollection);
}

} // namespace mlir::iree_compiler::IREE::Flow
