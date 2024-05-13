// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/MeshToFlow/Patterns.h"

#include <algorithm>
#include <numeric>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/Permutation.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Mesh/Transforms/Simplifications.h"
#include "mlir/Dialect/Mesh/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
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
convertReductionKind(mesh::ReductionKind reduction) {
  switch (reduction) {
  case mesh::ReductionKind::Max:
    return CollectiveReductionOp::ReductionMaximum;
  case mesh::ReductionKind::Min:
    return CollectiveReductionOp::ReductionMinimum;
  case mesh::ReductionKind::Sum:
    return CollectiveReductionOp::ReductionSum;
  default:
    assert(false);
    return CollectiveReductionOp::None;
  }
}

static CollectiveReductionOpAttr
convertReductionKind(mesh::ReductionKindAttr reduction) {
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
      builder.create<tensor::EmptyOp>(transposedShape, type.getElementType());
  return cast<TypedValue<RankedTensorType>>(
      builder.create<linalg::TransposeOp>(v, target, transposeVector)
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
      builder
          .create<tensor::ExpandShapeOp>(resultType, tensor,
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
  SmallVector<int64_t> res;
  std::copy(shape.begin(), shape.begin() + firstAxis, std::back_inserter(res));
  size_t collapsedAxisSize = std::accumulate(
      shape.begin() + firstAxis + 1, shape.begin() + firstAxis + n,
      shape[firstAxis], [](size_t a, size_t b) { return a * b; });
  res.push_back(collapsedAxisSize);
  std::copy(shape.begin() + firstAxis + n, shape.end(),
            std::back_inserter(res));
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
      builder.create<tensor::CollapseShapeOp>(tensor, reassociation.value())
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
struct MeshToFlowCollectiveRewritePatternBase : OpRewritePattern<Op> {
  template <typename... OpRewritePatternArgs>
  MeshToFlowCollectiveRewritePatternBase(
      SymbolTableCollection &symbolTableCollection,
      LookupMeshChannelFn lookupChannel,
      OpRewritePatternArgs &&...opRewritePatternArgs)
      : OpRewritePattern<Op>(
            std::forward<OpRewritePatternArgs...>(opRewritePatternArgs)...),
        symbolTableCollection(symbolTableCollection),
        lookupChannel(lookupChannel) {}

protected:
  // The !flow.channel corresponding to the mesh and mesh axes used in the op.
  template <typename MeshCollectiveOp>
  Value buildCachedChannelLoading(MeshCollectiveOp op,
                                  ImplicitLocOpBuilder &builder) const {
    mesh::MeshOp mesh = mesh::getMesh(op, symbolTableCollection);
    return lookupChannel(builder.getLoc(), mesh, op.getMeshAxes(), builder);
  }

  Value buildCachedChannelLoading(mesh::ProcessLinearIndexOp op,
                                  ImplicitLocOpBuilder &builder) const {
    mesh::MeshOp mesh = mesh::getMesh(op, symbolTableCollection);
    return lookupChannel(builder.getLoc(), mesh, std::nullopt, builder);
  }

  SymbolTableCollection &symbolTableCollection;
  LookupMeshChannelFn lookupChannel;
};

struct MeshAllReduceToFlow
    : MeshToFlowCollectiveRewritePatternBase<mesh::AllReduceOp> {
  using MeshToFlowCollectiveRewritePatternBase::
      MeshToFlowCollectiveRewritePatternBase;
  LogicalResult matchAndRewrite(mesh::AllReduceOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());
    Value channel = buildCachedChannelLoading(op, builder);
    Value target = builder.create<tensor::EmptyOp>(
        op.getResult().getType().getShape(),
        op.getResult().getType().getElementType());
    auto flowAllReduce = builder.create<IREE::Flow::CollectiveAllReduceOp>(
        convertReductionKind(op.getReductionAttr()),
        getCollectiveElementTypeAttr(op.getResult().getType()), target,
        op.getOperand(), channel);
    rewriter.replaceAllUsesWith(op.getResult(), flowAllReduce.getResult());
    rewriter.eraseOp(op.getOperation());
    return success();
  }
};

struct MeshAllGatherToFlow
    : MeshToFlowCollectiveRewritePatternBase<mesh::AllGatherOp> {
  using MeshToFlowCollectiveRewritePatternBase::
      MeshToFlowCollectiveRewritePatternBase;
  LogicalResult matchAndRewrite(mesh::AllGatherOp op,
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

    int64_t gatherAxis = op.getGatherAxis().getSExtValue();

    // When gather axis != 0, we need to transpose between 0 and
    // gather axis before and after the flow all-gather op.
    Value flowAllGatherOperand =
        buildTranspose(op.getOperand(), 0, gatherAxis, builder);

    RankedTensorType flowAllGatherResultType = transpose(
        cast<RankedTensorType>(op.getResult().getType()), 0, gatherAxis);
    Value target = builder.create<tensor::EmptyOp>(
        flowAllGatherResultType.getShape(),
        op.getResult().getType().getElementType());
    auto flowAllGather = builder.create<IREE::Flow::CollectiveAllGatherOp>(
        getCollectiveElementTypeAttr(flowAllGatherResultType), target,
        flowAllGatherOperand, channel);

    Value res = buildTranspose(flowAllGather, 0, gatherAxis, builder);

    rewriter.replaceAllUsesWith(op.getResult(), res);
    rewriter.eraseOp(op.getOperation());
    return success();
  }
};

struct MeshAllToAllToFlow
    : MeshToFlowCollectiveRewritePatternBase<mesh::AllToAllOp> {
  using MeshToFlowCollectiveRewritePatternBase::
      MeshToFlowCollectiveRewritePatternBase;
  LogicalResult matchAndRewrite(mesh::AllToAllOp op,
                                PatternRewriter &rewriter) const override {
    if (!cast<RankedTensorType>(op.getOperand().getType()).hasStaticShape() ||
        !op.getResult().getType().hasStaticShape()) {
      // TODO: add dynamic support.
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "Dynamic tensor case is unsupported.");
    }

    mesh::MeshOp mesh = mesh::getMesh(op, symbolTableCollection);
    assert(!ShapedType::isDynamicShape(mesh.getShape()));
    int64_t splitCount =
        mesh::collectiveProcessGroupSize(op.getMeshAxes(), mesh.getShape());
    // TODO: handle dynamic case.
    if (ShapedType::isDynamic(splitCount)) {
      // TODO: add dynamic support.
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "Dynamic split count induced by a dynamic mesh is unsupported.");
    }

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());

    Value channel = buildCachedChannelLoading(op, builder);

    int64_t splitAxis = op.getSplitAxis().getSExtValue();

    TypedValue<RankedTensorType> splitAxisAsMostOuter =
        buildTranspose(op.getOperand(), 0, splitAxis, builder);

    Value target = builder.create<tensor::EmptyOp>(
        splitAxisAsMostOuter.getType().getShape(),
        splitAxisAsMostOuter.getType().getElementType());
    auto flowAllToAll = builder.create<IREE::Flow::CollectiveAllToAllOp>(
        getCollectiveElementTypeAttr(splitAxisAsMostOuter.getType()), target,
        splitAxisAsMostOuter, channel);

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

struct MeshProcessLinearIndexToFlow
    : MeshToFlowCollectiveRewritePatternBase<mesh::ProcessLinearIndexOp> {
  using MeshToFlowCollectiveRewritePatternBase::
      MeshToFlowCollectiveRewritePatternBase;
  LogicalResult matchAndRewrite(mesh::ProcessLinearIndexOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());
    Value channel = buildCachedChannelLoading(op, builder);
    Value newIndex = builder.create<IREE::Flow::ChannelRankOp>(
        builder.getIndexType(), channel);
    rewriter.replaceAllUsesWith(op.getResult(), newIndex);
    return success();
  }
};

struct MeshReduceScatterToFlow
    : MeshToFlowCollectiveRewritePatternBase<mesh::ReduceScatterOp> {
  using MeshToFlowCollectiveRewritePatternBase::
      MeshToFlowCollectiveRewritePatternBase;
  LogicalResult matchAndRewrite(mesh::ReduceScatterOp op,
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

    Value target = builder.create<tensor::EmptyOp>(
        flowReduceScatterResultType.getShape(),
        op.getResult().getType().getElementType());
    auto flowReduceScatter =
        builder.create<IREE::Flow::CollectiveReduceScatterOp>(
            convertReductionKind(op.getReductionAttr()),
            getCollectiveElementTypeAttr(flowReduceScatterResultType), target,
            flowReduceScatterOperand, channel);

    Value res = buildTranspose(flowReduceScatter, 0, scatterAxis, builder);

    rewriter.replaceAllUsesWith(op.getResult(), res);
    rewriter.eraseOp(op.getOperation());
    return success();
  }
};

} // namespace

void populateMeshToFlowCollectivesPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection,
    LookupMeshChannelFn lookupChannel) {
  patterns.add<MeshAllGatherToFlow, MeshAllReduceToFlow, MeshAllToAllToFlow,
               MeshReduceScatterToFlow, MeshProcessLinearIndexToFlow>(
      symbolTableCollection, lookupChannel, patterns.getContext());
  mesh::populateFoldingPatterns(patterns, symbolTableCollection);
  mesh::populateProcessMultiIndexOpLoweringPatterns(patterns,
                                                    symbolTableCollection);
}

} // namespace mlir::iree_compiler::IREE::Flow
