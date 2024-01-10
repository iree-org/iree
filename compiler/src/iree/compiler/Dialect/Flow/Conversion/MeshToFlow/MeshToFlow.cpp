// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/MeshToFlow/MeshToFlow.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/Folding.h"
#include "iree/compiler/Utils/Indexing.h"
#include "iree/compiler/Utils/OpVisitor.h"
#include "iree/compiler/Utils/Permutation.h"
#include "iree/compiler/Utils/SmallVectorDenseMapInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
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
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-mesh-to-flow"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/Flow/Conversion/MeshToFlow/Passes.h.inc" // IWYU pragma: keep

// TODO: Use from MLIR when exposed.
static mesh::ClusterOp getMesh(Operation *op, FlatSymbolRefAttr meshSymbol,
                               SymbolTableCollection &symbolTableCollection) {
  return symbolTableCollection.lookupNearestSymbolFrom<mesh::ClusterOp>(
      op, meshSymbol);
}

template <typename MeshCollectiveOp>
static mesh::ClusterOp getMesh(MeshCollectiveOp op,
                               SymbolTableCollection &symbolTableCollection) {
  return getMesh(op.getOperation(), op.getMeshAttr(), symbolTableCollection);
}

static bool hasMoreThanOneMesh(Operation *op) {
  int meshCount = 0;
  op->walk([&meshCount](mesh::ClusterOp mesh) {
    ++meshCount;
    return meshCount > 1 ? WalkResult::interrupt() : WalkResult::advance();
  });
  return meshCount > 1;
}

static bool isDefaultChannel(mesh::ClusterOp mesh,
                             ArrayRef<mesh::MeshAxis> meshAxes) {
  if (mesh.getRank() != static_cast<int64_t>(meshAxes.size())) {
    return false;
  }
  return isIdentityPermutation(meshAxes);
}

static Value getDefaultChannel(mesh::ClusterOp mesh,
                               bool useNamedDefaultChannels,
                               ImplicitLocOpBuilder &builder) {
  if (useNamedDefaultChannels)
    return builder.create<IREE::Flow::ChannelDefaultOp>(mesh.getSymName());
  else
    return builder.create<IREE::Flow::ChannelDefaultOp>();
}

// Remove from `values` elements that have indices present in filter.
static SmallVector<Value> filterOutByIndex(ArrayRef<Value> values,
                                           ArrayRef<mesh::MeshAxis> filter) {
  SmallVector<Value> res;
  for (size_t i = 0; i < values.size(); ++i) {
    if (!llvm::is_contained(filter, i)) {
      res.push_back(values[i]);
    }
  }
  return res;
}

static CollectiveReductionOp convertReductionKind(mesh::Partial reduction) {
  switch (reduction) {
  case mesh::Partial::Max:
    return CollectiveReductionOp::ReductionMaximum;
  case mesh::Partial::Min:
    return CollectiveReductionOp::ReductionMinimum;
  case mesh::Partial::Sum:
    return CollectiveReductionOp::ReductionSum;
  default:
    assert(false);
    return CollectiveReductionOp::None;
  }
}

static CollectiveReductionOpAttr
convertReductionKind(mesh::PartialAttr reduction) {
  return CollectiveReductionOpAttr::get(
      reduction.getContext(), convertReductionKind(reduction.getValue()));
}

static Value buildChannelCreation(mesh::ClusterOp mesh,
                                  ArrayRef<mesh::MeshAxis> meshAxes,
                                  bool useNamedDefaultChannels,
                                  ImplicitLocOpBuilder &builder) {
  assert(mesh);
  Value meshChannel = getDefaultChannel(mesh, useNamedDefaultChannels, builder);
  SmallVector<Value> meshProcessMultiIndex =
      builder.create<mesh::ProcessMultiIndexOp>(mesh).getResults();
  SmallVector<Value> meshShape =
      builder.create<mesh::ClusterShapeOp>(mesh).getResults();
  SmallVector<Value> reorderedMeshIndex =
      permute(ArrayRef<Value>(meshProcessMultiIndex), meshAxes);
  SmallVector<Value> reorderedMeshShape =
      permute(ArrayRef<Value>(meshShape), meshAxes);
  SmallVector<Value> groupIndex =
      filterOutByIndex(meshProcessMultiIndex, meshAxes);
  SmallVector<Value> groupsShape = filterOutByIndex(meshShape, meshAxes);
  OpFoldResult reorderedProcessLinearIndex =
      linearIndexFromShape(toOpFoldResults(reorderedMeshIndex),
                           toOpFoldResults(reorderedMeshShape), builder);
  OpFoldResult color = linearIndexFromShape(
      toOpFoldResults(groupIndex), toOpFoldResults(groupsShape), builder);
  return builder.create<ChannelSplitOp>(
      meshChannel,
      getValueOrCreateConstantIndexOp(builder, builder.getLoc(), color),
      getValueOrCreateConstantIndexOp(builder, builder.getLoc(),
                                      reorderedProcessLinearIndex));
}

static SmallString<64> getChannelName(mesh::ClusterOp mesh,
                                      ArrayRef<mesh::MeshAxis> axes) {
  SmallString<64> res;
  llvm::raw_svector_ostream stream(res);
  stream << "_mesh_" << mesh.getSymName();
  if (axes.empty()) {
    return res;
  }

  stream << "_axes";
  for (mesh::MeshAxis axis : axes) {
    stream << "_" << axis;
  }

  return res;
}

static void buildChannelInitializer(mesh::ClusterOp mesh,
                                    ArrayRef<mesh::MeshAxis> meshAxes,
                                    bool useNamedDefaultChannels,
                                    ImplicitLocOpBuilder &builder) {
  Util::InitializerOp initOp = builder.create<Util::InitializerOp>();
  Block *block = builder.createBlock(&initOp.getBody());
  ImplicitLocOpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToStart(block);
  Value channel =
      buildChannelCreation(mesh, meshAxes, useNamedDefaultChannels, builder);
  builder.create<Util::GlobalStoreOp>(channel, getChannelName(mesh, meshAxes));
  builder.create<Util::ReturnOp>();
}

// Construct a Flow channel inside `module` using
// util.global and util.initializer.
static void buildGlobalChannelCreation(mesh::ClusterOp mesh,
                                       ArrayRef<mesh::MeshAxis> meshAxes,
                                       bool useNamedDefaultChannels,
                                       ModuleOp module, OpBuilder &opBuilder) {
  if (isDefaultChannel(mesh, meshAxes)) {
    return;
  }

  ImplicitLocOpBuilder builder(mesh.getLoc(), opBuilder);
  ImplicitLocOpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToStart(&module.getBodyRegion().getBlocks().front());

  auto channelName = getChannelName(mesh, meshAxes);
  builder.create<Util::GlobalOp>(
      builder.getStringAttr("private"), channelName,
      IREE::Flow::ChannelType::get(builder.getContext()), false, TypedAttr(),
      IREE::Util::InlineNeverAttr::get(builder.getContext()));
  buildChannelInitializer(mesh, meshAxes, useNamedDefaultChannels, builder);
}

static Value buildCachedChannelLoading(mesh::ClusterOp mesh,
                                       ArrayRef<mesh::MeshAxis> meshAxes,
                                       bool useNamedDefaultChannels,
                                       ImplicitLocOpBuilder &builder) {
  if (isDefaultChannel(mesh, meshAxes)) {
    return getDefaultChannel(mesh, useNamedDefaultChannels, builder);
  }
  return builder.create<Util::GlobalLoadOp>(
      ChannelType::get(builder.getContext()), getChannelName(mesh, meshAxes));
}

// The !flow.channel corresponding to the mesh and mesh axes used in the op.
template <typename MeshCollectiveOp>
static Value buildCachedChannelLoading(
    MeshCollectiveOp op, SymbolTableCollection &symbolTableCollection,
    bool useNamedDefaultChannels, ImplicitLocOpBuilder &builder) {
  ImplicitLocOpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointAfter(op);

  mesh::ClusterOp mesh = getMesh(op, symbolTableCollection);
  return buildCachedChannelLoading(mesh, op.getMeshAxes(),
                                   useNamedDefaultChannels, builder);
}

SmallVector<mesh::MeshAxis> getAllMeshAxes(mesh::ClusterOp mesh) {
  SmallVector<mesh::MeshAxis> res(mesh.getRank());
  std::iota(res.begin(), res.end(), 0);
  return res;
}

static Value buildCachedChannelLoading(
    mesh::ProcessLinearIndexOp op, SymbolTableCollection &symbolTableCollection,
    bool useNamedDefaultChannels, ImplicitLocOpBuilder &builder) {
  ImplicitLocOpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointAfter(op);

  mesh::ClusterOp mesh = getMesh(op, symbolTableCollection);
  return buildCachedChannelLoading(mesh, getAllMeshAxes(mesh),
                                   useNamedDefaultChannels, builder);
}

static TypedValue<RankedTensorType>
buildTranspose(Value v, ArrayRef<int64_t> transposeVector,
               ImplicitLocOpBuilder &builder) {
  RankedTensorType type = v.getType().cast<RankedTensorType>();
  SmallVector<int64_t> transposedShape =
      permute(type.getShape(), transposeVector);
  Value target =
      builder.create<tensor::EmptyOp>(transposedShape, type.getElementType());
  return builder.create<linalg::TransposeOp>(v, target, transposeVector)
      ->getResult(0)
      .cast<TypedValue<RankedTensorType>>();
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
  int64_t rank = v.getType().cast<RankedTensorType>().getRank();
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
  return builder.create<tensor::ExpandShapeOp>(resultType, tensor,
                                               reassociation.value());
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
  return builder.create<tensor::CollapseShapeOp>(tensor, reassociation.value());
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

// TODO: Use this function from MLIR when it is exposed in Mesh utils.
static int64_t collectiveDeviceGroupSize(ArrayRef<mesh::MeshAxis> meshAxes,
                                         ArrayRef<int64_t> meshShape) {
  int64_t res = 1;

  for (mesh::MeshAxis axis : meshAxes) {
    if (ShapedType::isDynamic(meshShape[axis])) {
      return ShapedType::kDynamic;
    }
    assert(size_t(axis) < meshShape.size());
    res *= meshShape[axis];
  }

  return res;
}

namespace {

template <typename Op>
struct MeshToFlowCollectiveRewritePatternBase : OpRewritePattern<Op> {
  template <typename... OpRewritePatternArgs>
  MeshToFlowCollectiveRewritePatternBase(
      SymbolTableCollection &symbolTableCollection,
      bool useNamedDefaultChannels,
      OpRewritePatternArgs &&...opRewritePatternArgs)
      : OpRewritePattern<Op>(
            std::forward<OpRewritePatternArgs...>(opRewritePatternArgs)...),
        symbolTableCollection(symbolTableCollection),
        useNamedDefaultChannels(useNamedDefaultChannels) {}

protected:
  SymbolTableCollection &symbolTableCollection;
  bool useNamedDefaultChannels;
};

struct MeshAllReduceToFlow
    : MeshToFlowCollectiveRewritePatternBase<mesh::AllReduceOp> {
  using MeshToFlowCollectiveRewritePatternBase::
      MeshToFlowCollectiveRewritePatternBase;

  LogicalResult matchAndRewrite(mesh::AllReduceOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());
    Value channel = buildCachedChannelLoading(op, symbolTableCollection,
                                              useNamedDefaultChannels, builder);
    Value target = builder.create<tensor::EmptyOp>(
        op.getResult().getType().getShape(),
        op.getResult().getType().getElementType());
    auto flowAllReduce = builder.create<CollectiveAllReduceOp>(
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
    if (ShapedType::isDynamicShape(
            op.getOperand().getType().cast<RankedTensorType>().getShape()) ||
        ShapedType::isDynamicShape(op.getResult().getType().getShape())) {
      // TODO: add dynamic support.
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "Dynamic tensor case is unsupported.");
    }

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());
    Value channel = buildCachedChannelLoading(op, symbolTableCollection,
                                              useNamedDefaultChannels, builder);

    int64_t gatherAxis = op.getGatherAxis().getSExtValue();

    // When gather axis != 0, we need to transpose between 0 and
    // gather axis before and after the flow all-gather op.
    Value flowAllGatherOperand =
        buildTranspose(op.getOperand(), 0, gatherAxis, builder);

    RankedTensorType flowAllGatherResultType = transpose(
        op.getResult().getType().cast<RankedTensorType>(), 0, gatherAxis);
    Value target = builder.create<tensor::EmptyOp>(
        flowAllGatherResultType.getShape(),
        op.getResult().getType().getElementType());
    auto flowAllGather = builder.create<CollectiveAllGatherOp>(
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
    if (ShapedType::isDynamicShape(
            op.getOperand().getType().cast<RankedTensorType>().getShape()) ||
        ShapedType::isDynamicShape(op.getResult().getType().getShape())) {
      // TODO: add dynamic support.
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "Dynamic tensor case is unsupported.");
    }

    mesh::ClusterOp mesh = getMesh(op, symbolTableCollection);
    assert(!ShapedType::isDynamicShape(mesh.getShape()));
    int64_t splitCount =
        collectiveDeviceGroupSize(op.getMeshAxes(), mesh.getShape());
    // TODO: handle dynamic case.
    if (ShapedType::isDynamic(splitCount)) {
      // TODO: add dynamic support.
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "Dynamic split count induced by a dynamic mesh is unsupported.");
    }

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());

    Value channel = buildCachedChannelLoading(op, symbolTableCollection,
                                              useNamedDefaultChannels, builder);

    int64_t splitAxis = op.getSplitAxis().getSExtValue();

    TypedValue<RankedTensorType> splitAxisAsMostOuter =
        buildTranspose(op.getOperand(), 0, splitAxis, builder);

    Value target = builder.create<tensor::EmptyOp>(
        splitAxisAsMostOuter.getType().getShape(),
        splitAxisAsMostOuter.getType().getElementType());
    auto flowAllToAll = builder.create<CollectiveAllToAllOp>(
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
    Value channel = buildCachedChannelLoading(op, symbolTableCollection,
                                              useNamedDefaultChannels, builder);
    Value newIndex =
        builder.create<ChannelRankOp>(builder.getIndexType(), channel);
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
    if (ShapedType::isDynamicShape(
            op.getOperand().getType().cast<RankedTensorType>().getShape()) ||
        ShapedType::isDynamicShape(op.getResult().getType().getShape())) {
      // TODO: add dynamic support.
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "Dynamic tensor case is unsupported.");
    }

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());
    Value channel = buildCachedChannelLoading(op, symbolTableCollection,
                                              useNamedDefaultChannels, builder);

    int64_t scatterAxis = op.getScatterAxis().getSExtValue();

    // When scatter axis != 0, we need to transpose between 0 and
    // scatter axis before and after the flow reduce-scatter op.
    Value flowReduceScatterOperand =
        buildTranspose(op.getOperand(), 0, scatterAxis, builder);
    RankedTensorType flowReduceScatterResultType = transpose(
        op.getResult().getType().cast<RankedTensorType>(), 0, scatterAxis);

    Value target = builder.create<tensor::EmptyOp>(
        flowReduceScatterResultType.getShape(),
        op.getResult().getType().getElementType());
    auto flowReduceScatter = builder.create<CollectiveReduceScatterOp>(
        convertReductionKind(op.getReductionAttr()),
        getCollectiveElementTypeAttr(flowReduceScatterResultType), target,
        flowReduceScatterOperand, channel);

    Value res = buildTranspose(flowReduceScatter, 0, scatterAxis, builder);

    rewriter.replaceAllUsesWith(op.getResult(), res);
    rewriter.eraseOp(op.getOperation());
    return success();
  }
};

using MeshAndAxesSet =
    DenseSet<std::tuple<mesh::ClusterOp, SmallVector<mesh::MeshAxis>>>;

template <typename Op>
struct CollectiveOpVisitor {
  CollectiveOpVisitor(MeshAndAxesSet &meshAndAxesSet,
                      SymbolTableCollection &symbolTableCollection)
      : meshAndAxesSet(meshAndAxesSet),
        symbolTableCollection(symbolTableCollection) {}
  void operator()(Op op) {
    meshAndAxesSet.insert(std::make_tuple(
        symbolTableCollection.lookupNearestSymbolFrom<mesh::ClusterOp>(
            op, op.getMeshAttr()),
        llvm::to_vector(op.getMeshAxes())));
  }

private:
  MeshAndAxesSet &meshAndAxesSet;
  SymbolTableCollection &symbolTableCollection;
};

template <typename Op>
struct CollectiveOpWithoutMeshAxesVisitor {
  CollectiveOpWithoutMeshAxesVisitor(
      MeshAndAxesSet &meshAndAxesSet,
      SymbolTableCollection &symbolTableCollection)
      : meshAndAxesSet(meshAndAxesSet),
        symbolTableCollection(symbolTableCollection) {}
  void operator()(Op op) {
    mesh::ClusterOp mesh =
        symbolTableCollection.lookupNearestSymbolFrom<mesh::ClusterOp>(
            op, op.getMeshAttr());
    meshAndAxesSet.insert(std::make_tuple(mesh, getAllMeshAxes(mesh)));
  }

private:
  MeshAndAxesSet &meshAndAxesSet;
  SymbolTableCollection &symbolTableCollection;
};

void populateMeshAndAxes(Operation *op, MeshAndAxesSet &meshAndAxesSet,
                         SymbolTableCollection &symbolTableCollection) {
  OpVisitorCollection opVisitors;
  opVisitors.emplaceVisitors<
      CollectiveOpVisitor<mesh::AllGatherOp>,
      CollectiveOpVisitor<mesh::AllReduceOp>,
      CollectiveOpVisitor<mesh::AllToAllOp>,
      CollectiveOpVisitor<mesh::ReduceScatterOp>,
      CollectiveOpWithoutMeshAxesVisitor<mesh::ProcessLinearIndexOp>>(
      meshAndAxesSet, symbolTableCollection);

  op->walk([&opVisitors](Operation *op) {
    opVisitors(op);
    return WalkResult::advance();
  });
}

static void createChannels(ModuleOp moduleOp,
                           SymbolTableCollection &symbolTableCollection,
                           MeshAndAxesSet &meshAndAxesSet,
                           bool useNamedDefaultChannels) {
  populateMeshAndAxes(moduleOp, meshAndAxesSet, symbolTableCollection);

  OpBuilder builder(moduleOp->getContext());

  // Sort for deterministic testing with FileCheck.
  auto meshAndAxesSetSorted = llvm::to_vector(meshAndAxesSet);
  llvm::sort(meshAndAxesSetSorted, [](auto &a, auto &b) {
    int nameCompareRes =
        std::get<0>(a).getSymName().compare(std::get<0>(b).getSymName());
    if (nameCompareRes == 0)
      return std::get<1>(a) < std::get<1>(b);
    return nameCompareRes < 0;
  });
  for (auto &[mesh, meshAxes] : llvm::make_range(meshAndAxesSetSorted.rbegin(),
                                                 meshAndAxesSetSorted.rend())) {
    buildGlobalChannelCreation(mesh, meshAxes, useNamedDefaultChannels,
                               moduleOp, builder);
  }
}

static LogicalResult
convertCollectives(ModuleOp moduleOp,
                   SymbolTableCollection &symbolTableCollection,
                   bool useNamedDefaultChannels) {
  RewritePatternSet patterns(moduleOp->getContext());
  IREE::Flow::populateMeshToFlowCollectivesPatterns(
      patterns, symbolTableCollection, useNamedDefaultChannels);
  return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

static void removeMeshClusterOps(MeshAndAxesSet &meshAndAxesSet) {
  auto meshRange =
      llvm::map_range(meshAndAxesSet, [](auto &v) { return std::get<0>(v); });
  DenseSet<mesh::ClusterOp> clusterOpsSet(std::begin(meshRange),
                                          std::end(meshRange));
  for (mesh::ClusterOp op : clusterOpsSet) {
    if (op)
      op.erase();
  }
}

struct ConvertMeshToFlowPass
    : public IREE::Flow::ConvertMeshToFlowBase<ConvertMeshToFlowPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registerMeshToFlowDependencies(registry);
  }

  void runOnOperation() override {
    // Run only on the top module.
    if (getOperation()->getParentOp() != nullptr) {
      return;
    }

    MeshAndAxesSet meshAndAxesSet;
    SymbolTableCollection symbolTableCollection;
    bool useNamedDefaultChannels = hasMoreThanOneMesh(getOperation());

    createChannels(getOperation(), symbolTableCollection, meshAndAxesSet,
                   useNamedDefaultChannels);
    if (failed(convertCollectives(getOperation(), symbolTableCollection,
                                  useNamedDefaultChannels))) {
      return signalPassFailure();
    }

    // Cleanup cluster definition ops that are no longer referenced.
    removeMeshClusterOps(meshAndAxesSet);
  }
};

} // namespace

void populateMeshToFlowCollectivesPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection,
    bool useNamedDefaultChannels) {
  patterns.add<MeshAllGatherToFlow, MeshAllReduceToFlow, MeshAllToAllToFlow,
               MeshReduceScatterToFlow, MeshProcessLinearIndexToFlow>(
      symbolTableCollection, useNamedDefaultChannels, patterns.getContext());
  mesh::populateFoldingPatterns(patterns, symbolTableCollection);
  mesh::processMultiIndexOpLoweringPopulatePatterns(patterns,
                                                    symbolTableCollection);
}

std::unique_ptr<Pass> createConvertMeshToFlowPass() {
  return std::make_unique<ConvertMeshToFlowPass>();
}

void registerMeshToFlowDependencies(DialectRegistry &registry) {
  registry.insert<affine::AffineDialect, FlowDialect, linalg::LinalgDialect,
                  mesh::MeshDialect, tensor::TensorDialect>();
}

void registerMeshToFlowPasses() { registerPasses(); }

} // namespace mlir::iree_compiler::IREE::Flow
