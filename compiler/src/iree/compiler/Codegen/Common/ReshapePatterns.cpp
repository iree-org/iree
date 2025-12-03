// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-codegen-reshape-patterns"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FOLDRESHAPEINTOINTERFACETENSORPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

//===---------------------------------------------------------------------===//
// Utility functions
//===---------------------------------------------------------------------===//

static int64_t getProductExcludingDynamic(ArrayRef<int64_t> sizes) {
  return llvm::accumulate(sizes, int64_t(1), [](int64_t res, int64_t size) {
    return ShapedType::isDynamic(size) ? res : res * size;
  });
}

//===---------------------------------------------------------------------===//
// Patterns to fold tensor.expand/collapse_shape into
// `hal.interface.binding.subspan`
//===---------------------------------------------------------------------===//

static SmallVector<OpFoldResult>
inferCollapsedShape(RewriterBase &rewriter, Location loc,
                    RankedTensorType expandedType,
                    ArrayRef<ReassociationIndices> reassociations,
                    ValueRange expandedDynamicDims) {
  ArrayRef<int64_t> expandedStaticShape = expandedType.getShape();
  SmallVector<OpFoldResult> expandedMixedShape =
      mlir::getMixedValues(expandedStaticShape, expandedDynamicDims, rewriter);
  SmallVector<OpFoldResult> collapsedShape;
  unsigned expandedShapeDim = 0;
  for (auto reassociation : reassociations) {
    AffineExpr mulExpr = rewriter.getAffineSymbolExpr(0);
    for (auto i : llvm::seq<unsigned>(1, reassociation.size())) {
      mulExpr = mulExpr * rewriter.getAffineSymbolExpr(i);
    }
    auto collapsedDim = affine::makeComposedFoldedAffineApply(
        rewriter, loc, mulExpr,
        ArrayRef(expandedMixedShape)
            .slice(expandedShapeDim, reassociation.size()));
    collapsedShape.push_back(collapsedDim);
    expandedShapeDim += reassociation.size();
  }
  return collapsedShape;
}

/// Folds tensor.expand/collapse_shape into the source
/// hal.interface.binding.subspan.
///
/// For example, this matches the following pattern:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1x96xf32>>
///   %tensor = iree_tensor_ext.dispatch.tensor.load %subspan :
///       !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1x96xf32>> ->
///       tensor<3x3x1x96xf32>
///   %0 = linalg.tensor_reshape %tensor [
///         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///       ] : tensor<3x3x1x96xf32> into tensor<864xf32>
///
/// And turns it into:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !iree_tensor_ext.dispatch.tensor<readonly:tensor<864xf32>>
///   %0 = iree_tensor_ext.dispatch.tensor.load %subspan :
///       !iree_tensor_ext.dispatch.tensor<readonly:tensor<864xf32>> ->
///       tensor<864xf32>
struct FoldCollapseShapeIntoInterfaceTensorLoad
    : OpRewritePattern<tensor::CollapseShapeOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    Value reshapeSrc = reshapeOp.getSrc();
    auto reshapeSrcType = cast<RankedTensorType>(reshapeSrc.getType());
    auto loadOp =
        reshapeSrc.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    if (!loadOp)
      return failure();

    // Make sure we are loading the full incoming subspan. Otherwise we cannot
    // simply adjust the subspan's resultant type later.
    if (!isFullSlice(loadOp, loadOp.getSourceType(), loadOp.getSourceDims())) {
      return failure();
    }

    auto subspanOp = loadOp.getSource()
                         .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp)
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(subspanOp);
    SmallVector<OpFoldResult> collapsedShape = inferCollapsedShape(
        rewriter, subspanOp.getLoc(), reshapeSrcType,
        reshapeOp.getReassociationIndices(), subspanOp.getDynamicDims());
    SmallVector<int64_t> collapsedStaticShape;
    SmallVector<Value> collapsedDynamicShape;
    dispatchIndexOpFoldResults(collapsedShape, collapsedDynamicShape,
                               collapsedStaticShape);

    auto tensorAccess =
        cast<IREE::TensorExt::DispatchTensorType>(subspanOp.getType())
            .getAccess();
    auto newSubspanType = IREE::TensorExt::DispatchTensorType::get(
        tensorAccess, reshapeOp.getResultType());

    Value newSubspanOp = IREE::HAL::InterfaceBindingSubspanOp::create(
        rewriter, subspanOp.getLoc(), newSubspanType, subspanOp.getLayout(),
        subspanOp.getBinding(), subspanOp.getByteOffset(),
        collapsedDynamicShape, subspanOp.getAlignmentAttr(),
        subspanOp.getDescriptorFlagsAttr());

    rewriter.setInsertionPoint(reshapeOp);
    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorLoadOp>(
        reshapeOp, reshapeOp.getResultType(), newSubspanOp,
        collapsedDynamicShape);

    return success();
  }
};

/// Folds tensor.expand_shape into the source
/// hal.interface.binding.subspan.
///
/// For example, this matches the following pattern:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1x96xf32>>
///   %tensor = iree_tensor_ext.dispatch.tensor.load %subspan :
///       !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1x96xf32>> ->
///       tensor<3x3x1x96xf32>
///   %0 = linalg.expand_reshape %tensor [
///         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///       ] : tensor<3x3x1x96xf32> into tensor<864xf32>
///
/// And turns it into:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !iree_tensor_ext.dispatch.tensor<readonly:tensor<864xf32>>
///   %0 = iree_tensor_ext.dispatch.tensor.load %subspan :
///       !iree_tensor_ext.dispatch.tensor<readonly:tensor<864xf32>> ->
///       tensor<864xf32>
struct FoldExpandShapeIntoInterfaceTensorLoad
    : OpRewritePattern<tensor::ExpandShapeOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    Value reshapeSrc = reshapeOp.getSrc();
    auto loadOp =
        reshapeSrc.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    if (!loadOp) {
      return failure();
    }

    // Make sure we are loading the full incoming subspan. Otherwise we cannot
    // simply adjust the subspan's resultant type later.
    if (!isFullSlice(loadOp, loadOp.getSourceType(), loadOp.getSourceDims())) {
      return failure();
    }

    // In the corner case where the expand_shape is the source of a store, dont
    // fold with the load. Instead fold with the store to reduce the
    // dimensionality
    if (reshapeOp->hasOneUse()) {
      if (auto storeOp = dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(
              *reshapeOp->getUsers().begin())) {
        if (isFullSlice(storeOp, storeOp.getTargetType(),
                        storeOp.getTargetDims())) {
          return rewriter.notifyMatchFailure(reshapeOp,
                                             "fold with store instead");
        }
      }
    }

    auto subspanOp = loadOp.getSource()
                         .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp)
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(subspanOp);

    auto currDynamicDims = subspanOp.getDynamicDims();
    auto currStaticDims = loadOp.getType().getShape();
    auto currOfrDynamicDims =
        mlir::getMixedValues(currStaticDims, currDynamicDims, rewriter);

    // Try to infer the expanded shape. This only works if each reassociation
    // has <=1 dyn dim.
    std::optional<SmallVector<OpFoldResult>> expandedDims =
        mlir::inferExpandShapeOutputShape(
            rewriter, subspanOp.getLoc(), reshapeOp.getType(),
            reshapeOp.getReassociationIndices(), currOfrDynamicDims);
    if (!expandedDims) {
      // If inference fails, try to use the reshape's SSA values.
      if (failed(mlir::moveValueDefinitions(
              rewriter, reshapeOp.getOutputShape(), subspanOp))) {
        return rewriter.notifyMatchFailure(reshapeOp,
                                           "could not infer output shape or "
                                           "move SSA values before subspan op");
      }
      expandedDims = reshapeOp.getMixedOutputShape();
    }

    auto tensorAccess =
        cast<IREE::TensorExt::DispatchTensorType>(subspanOp.getType())
            .getAccess();
    auto newSubspanType = IREE::TensorExt::DispatchTensorType::get(
        tensorAccess, reshapeOp.getResultType());

    SmallVector<Value> expandedDynamicDims;
    SmallVector<int64_t> expandedStaticDims;
    dispatchIndexOpFoldResults(expandedDims.value(), expandedDynamicDims,
                               expandedStaticDims);

    Value newSubspanOp;
    newSubspanOp = IREE::HAL::InterfaceBindingSubspanOp::create(
        rewriter, subspanOp.getLoc(), newSubspanType, subspanOp.getLayout(),
        subspanOp.getBinding(), subspanOp.getByteOffset(), expandedDynamicDims,
        subspanOp.getAlignmentAttr(), subspanOp.getDescriptorFlagsAttr());

    rewriter.setInsertionPoint(reshapeOp);
    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorLoadOp>(
        reshapeOp, reshapeOp.getResultType(), newSubspanOp,
        expandedDynamicDims);

    return success();
  }
};

/// Folds tensor.expand into the source hal.interface.binding.subspan.
///
/// For example, this matches the following pattern:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x3x1x96xf32>>
///   %0 = tensor.expand_shape %tensor [[0, 1, 2, 3]]
///       : tensor<864xf32> into tensor<3x3x1x96xf32>
///   %tensor = iree_tensor_ext.dispatch.tensor.store %0, %subspan :
///       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x3x1x96xf32>> ->
///       tensor<3x3x1x96xf32>
///
/// And turns it into:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<864xf32>>
///   %0 = iree_tensor_ext.dispatch.tensor.store %tensor, %subspan :
///       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<864xf32>> ->
///       tensor<864xf32>
struct FoldExpandShapeIntoInterfaceTensorStore
    : OpRewritePattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using OpRewritePattern<
      IREE::TensorExt::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Make sure we are storing the full incoming subspan. Otherwise we cannot
    // simply adjust the subspan's resultant type later.
    if (!isFullSlice(storeOp, storeOp.getTargetType(),
                     storeOp.getTargetDims())) {
      return failure();
    }

    auto reshapeOp = storeOp.getValue().getDefiningOp<tensor::ExpandShapeOp>();
    if (!reshapeOp) {
      return failure();
    }

    Value reshapeSrc = reshapeOp.getSrc();
    // If the source is a `iree_tensor_ext.dispatch.tensor.load`, fold with the
    // load instead to reduce dimensionality of the problem
    if (auto loadOp =
            reshapeSrc.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>()) {
      if (isFullSlice(loadOp, loadOp.getSourceType(), loadOp.getSourceDims())) {
        return rewriter.notifyMatchFailure(
            storeOp, "fold expand_shape with load instead");
      }
    }

    auto subspanOp = storeOp.getTarget()
                         .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp)
      return failure();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(subspanOp);
    SmallVector<OpFoldResult> collapsedShape = inferCollapsedShape(
        rewriter, subspanOp.getLoc(), reshapeOp.getResultType(),
        reshapeOp.getReassociationIndices(), subspanOp.getDynamicDims());
    SmallVector<int64_t> collapsedStaticShape;
    SmallVector<Value> collapsedDynamicShape;
    dispatchIndexOpFoldResults(collapsedShape, collapsedDynamicShape,
                               collapsedStaticShape);

    auto tensorAccess =
        cast<IREE::TensorExt::DispatchTensorType>(subspanOp.getType())
            .getAccess();
    auto newSubspanType = IREE::TensorExt::DispatchTensorType::get(
        tensorAccess, reshapeSrc.getType());

    Value newSubspanOp = IREE::HAL::InterfaceBindingSubspanOp::create(
        rewriter, subspanOp.getLoc(), newSubspanType, subspanOp.getLayout(),
        subspanOp.getBinding(), subspanOp.getByteOffset(),
        collapsedDynamicShape, subspanOp.getAlignmentAttr(),
        subspanOp.getDescriptorFlagsAttr());

    rewriter.setInsertionPoint(storeOp);
    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorStoreOp>(
        storeOp, reshapeSrc, newSubspanOp, collapsedDynamicShape);

    return success();
  }
};

// Helper function to fix-up expanded values using the original (collapsed)
// index and the reassociation indices.
template <typename T>
static void transformOverReassociation(
    MutableArrayRef<T> expandedValues,
    ArrayRef<ReassociationIndices> reassocInfo,
    llvm::function_ref<void(size_t /*collapsedIdx*/,
                            ReassociationIndicesRef /*reassoc*/,
                            MutableArrayRef<T> /*expandedValues*/)>
        transformFn) {
  for (auto [idx, reassoc] : llvm::enumerate(reassocInfo)) {
    size_t reassocSize = reassoc.size();
    SmallVector<T> collapsedValues;
    collapsedValues.reserve(reassocSize);
    for (int64_t expandedIdx : reassoc) {
      collapsedValues.push_back(std::move(expandedValues[expandedIdx]));
    }

    transformFn(idx, reassoc, collapsedValues);

    for (auto [newValue, expandedIdx] :
         llvm::zip_equal(collapsedValues, reassoc)) {
      expandedValues[expandedIdx] = std::move(newValue);
    }
  }
}

/// Folds tensor.collapse_shape into the source hal.interface.binding.subspan.
///
/// For example, this matches the following pattern:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x3x1x96xf32>>
///   %0 = tensor.collapse_shape %tensor [[0, 1, 2, 3]]
///       : tensor<3x?x?x96xf32> into tensor<?xf32>
///   %tensor = iree_tensor_ext.dispatch.tensor.store %0, %subspan :
///       tensor<?xf32> ->
///       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%dim}
///
/// And turns it into:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x?x?x96xf32>>
///   %0 = iree_tensor_ext.dispatch.tensor.store %tensor, %subspan :
///       tensor<3x?x?x96xf32> ->
///       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x?x?x96xf32>>{%d0,
///       %d1}
///
/// TODO: This handles full slices (along collapsed dims). The pattern below
/// (`FoldCollapseShapeIntoTensorInsertSlice`) handles cases where the slice is
/// not a full slice, but requires the shapes to be static. This pattern handles
/// dynamic shapes as well. Combine the two (if possible, it isn't clear that it
/// is possible)
struct FoldCollapseShapeIntoInterfaceTensorStoreFullSlice
    : OpRewritePattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using OpRewritePattern<
      IREE::TensorExt::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto reshapeOp =
        storeOp.getValue().getDefiningOp<tensor::CollapseShapeOp>();
    if (!reshapeOp) {
      return failure();
    }

    auto subspanOp = storeOp.getTarget()
                         .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp) {
      return failure();
    }

    if (!areAllConstantIntValue(storeOp.getMixedStrides(), 1)) {
      return rewriter.notifyMatchFailure(storeOp, "found a non-1 stride");
    }

    const llvm::SmallBitVector droppedDims = storeOp.getDroppedDims();
    int64_t firstStoreDim = 0;
    while (firstStoreDim < droppedDims.size() &&
           droppedDims.test(firstStoreDim)) {
      ++firstStoreDim;
    }
    const int64_t lastStoreDim =
        firstStoreDim + reshapeOp.getResultType().getRank();

    SmallVector<ReassociationIndices> reassocInfo =
        reshapeOp.getReassociationIndices();

    // To support partial stores, keep track of collapsed and non-collapsed
    // dimensions. We will need these to ensure that the collapse does not
    // happen along partial store dimension and to update the store type.
    SmallVector<int64_t> collapsedDstDims;
    for (auto [index, reassocIndices] : llvm::enumerate(reassocInfo)) {
      if (reassocIndices.size() > 1) {
        collapsedDstDims.push_back(index + firstStoreDim);
      }
    }

    // Make sure we are storing the full incoming subspan slice along the
    // collapsed indices. Otherwise we cannot simply adjust the subspan's
    // resultant type later.
    const SmallVector<OpFoldResult, 4> origOffsets = storeOp.getMixedOffsets();
    const SmallVector<OpFoldResult, 4> origSizes = storeOp.getMixedSizes();
    const SmallVector<OpFoldResult> mixedTensorShape = getMixedValues(
        storeOp.getTargetType().getShape(), storeOp.getTargetDims(), rewriter);

    for (int64_t collapsedDim : collapsedDstDims) {
      if (origSizes[collapsedDim] != mixedTensorShape[collapsedDim] ||
          !isZeroInteger(origOffsets[collapsedDim])) {
        return rewriter.notifyMatchFailure(
            storeOp,
            llvm::formatv(
                "found a partial store along the collapsed dimension {}",
                collapsedDim));
      }
    }

    Value reshapeSrc = reshapeOp.getSrc();
    auto reshapeSrcType = cast<RankedTensorType>(reshapeSrc.getType());

    // Compute the type and dynamic dims of the interface binding.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(subspanOp);
    auto dynamicDims = subspanOp.getDynamicDims();
    ArrayRef<int64_t> staticShape = reshapeOp.getType().getShape();
    SmallVector<OpFoldResult> mixedShape =
        mlir::getMixedValues(staticShape, dynamicDims, rewriter);
    std::optional<SmallVector<OpFoldResult>> expandedShape =
        mlir::inferExpandShapeOutputShape(
            rewriter, subspanOp.getLoc(),
            cast<ShapedType>(reshapeSrc.getType()), reassocInfo, mixedShape);
    if (!expandedShape) {
      return rewriter.notifyMatchFailure(
          storeOp, "failed to compute expand shape for interface binding");
    }

    transformOverReassociation<OpFoldResult>(
        *expandedShape, reassocInfo,
        [&mixedTensorShape](size_t collapsedIdx, ReassociationIndicesRef,
                            MutableArrayRef<OpFoldResult> expandedValues) {
          if (expandedValues.size() == 1) {
            expandedValues[0] = mixedTensorShape[collapsedIdx];
          }
        });

    SmallVector<OpFoldResult> dispatchTensorMixedShape = mlir::getMixedValues(
        storeOp.getTargetType().getShape(), storeOp.getTargetDims(), rewriter);
    if (firstStoreDim != 0) {
      SmallVector<OpFoldResult> tmp(
          ArrayRef<OpFoldResult>(dispatchTensorMixedShape)
              .take_front(firstStoreDim));
      llvm::append_range(tmp, *expandedShape);
      *expandedShape = std::move(tmp);
    }
    llvm::append_range(*expandedShape,
                       ArrayRef<OpFoldResult>(dispatchTensorMixedShape)
                           .drop_front(lastStoreDim));

    SmallVector<int64_t> expandedStaticShape;
    SmallVector<Value> expandedDynamicShape;
    dispatchIndexOpFoldResults(*expandedShape, expandedDynamicShape,
                               expandedStaticShape);

    auto tensorAccess =
        cast<IREE::TensorExt::DispatchTensorType>(subspanOp.getType())
            .getAccess();
    auto newSubspanShape = llvm::to_vector_of<int64_t>(expandedStaticShape);
    auto newSubspanType = IREE::TensorExt::DispatchTensorType::get(
        tensorAccess, reshapeSrcType.cloneWith(
                          newSubspanShape, reshapeSrcType.getElementType()));
    auto newSubspanOp = IREE::HAL::InterfaceBindingSubspanOp::create(
        rewriter, subspanOp.getLoc(), newSubspanType, subspanOp.getLayout(),
        subspanOp.getBinding(), subspanOp.getByteOffset(), expandedDynamicShape,
        subspanOp.getAlignmentAttr(), subspanOp.getDescriptorFlagsAttr());

    rewriter.setInsertionPoint(storeOp);

    int64_t reshapeDstRank = reshapeOp.getType().getRank();
    int64_t reshapeSrcRank = reshapeOp.getSrcType().getRank();
    int64_t targetRank = storeOp.getTargetType().getRank();

    // Create a new reassociation that represents the change in shape of the
    // entire target tensor (not just the inserted section).
    SmallVector<ReassociationIndices> newReassocInfo;
    for (int64_t i = 0; i < firstStoreDim; ++i) {
      newReassocInfo.push_back(ReassociationIndices{i});
    }
    for (ReassociationIndices reassoc : reassocInfo) {
      for (int64_t &elem : reassoc) {
        elem += firstStoreDim;
      }
      newReassocInfo.push_back(std::move(reassoc));
    }
    for (int64_t i = lastStoreDim; i < targetRank; ++i) {
      newReassocInfo.push_back(
          ReassociationIndices{i + reshapeSrcRank - reshapeDstRank});
    }

    const size_t expandedSize = newSubspanShape.size();
    SmallVector<OpFoldResult> newOffsets(expandedSize,
                                         rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> newSizes(expandedSize);
    const SmallVector<OpFoldResult> newStrides(expandedSize,
                                               rewriter.getIndexAttr(1));
    transformOverReassociation<OpFoldResult>(
        newOffsets, newReassocInfo,
        [&origOffsets](size_t collapseIdx, ReassociationIndicesRef reassoc,
                       MutableArrayRef<OpFoldResult> expandedValues) {
          // Restore the original offsets of non-collapsed dimensions.
          if (reassoc.size() == 1) {
            expandedValues[0] = origOffsets[collapseIdx];
          }
        });

    transformOverReassociation<OpFoldResult>(
        newSizes, newReassocInfo,
        [&expandedShape,
         &origSizes](size_t collapseIdx, ReassociationIndicesRef reassoc,
                     MutableArrayRef<OpFoldResult> expandedValues) {
          // Restore the original sizes of non-collapsed dimensions.
          if (reassoc.size() == 1) {
            expandedValues[0] = origSizes[collapseIdx];
            return;
          }

          // Otherwise, use the shape dims for full slices along the collapsed
          // dims.
          for (auto [idx, expandedValue] :
               llvm::zip_equal(reassoc, expandedValues)) {
            expandedValue = (*expandedShape)[idx];
          }
        });

    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorStoreOp>(
        storeOp, reshapeSrc, newSubspanOp, expandedDynamicShape, newOffsets,
        newSizes, newStrides);
    return success();
  }
};

/// Folds tensor.collapse_shape with static or partially dynamic shape into the
/// source hal.interface.binding.subspan.
///
/// For example, this matches the following pattern:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2592xf32>>
///   %0 = tensor.collapse_shape %tensor [[0, 1, 2, 3]]
///       : tensor<3x3x1x96xf32> into tensor<864xf32>
///   %tensor = iree_tensor_ext.dispatch.tensor.store %0, %subspan,
///       offsets = [%x], sizes = [864], strides = [1]
///       : tensor<864xf32> ->
///       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2592xf32>>
///
/// And turns it into:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<9x3x1x96xf32>>
///   %0 = iree_tensor_ext.dispatch.tensor.store %tensor, %subspan :
///       offsets = [%x * 286, 0, 0, 0], sizes = [3, 3, 1, 96]
///       strides = [1, 1, 1, 1] : tensor<3x3x1x96xf32> ->
///       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<9x3x1x96xf32>>
///
/// Dynamic offsets are supported if they come from an affine.apply operation
/// where the result is provably divisible by the inner dimension product. For
/// example, if the inner dimensions are [4, 128] (product = 512), an offset of
/// `affine_map<()[s0] -> (s0 * 512)>` can be transformed to `s0`.
///
/// The divisibility check handles Add and Mul expressions recursively:
/// - For Mul: divisible if either operand is divisible (pessimistic)
/// - For Add: divisible if both operands are divisible
/// - For constants: divisible if value % innerDimProduct == 0
struct FoldCollapseShapeIntoInterfaceTensorStore
    : OpRewritePattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using OpRewritePattern<
      IREE::TensorExt::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Bail out if the strides aren't unit.
    if (!llvm::all_of(storeOp.getMixedStrides(), isOneInteger)) {
      return rewriter.notifyMatchFailure(storeOp, "expected unit strides");
    }

    auto collapseShape =
        storeOp.getValue().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseShape) {
      return rewriter.notifyMatchFailure(
          storeOp, "expected `tensor.collapse_shape` source");
    }
    if (collapseShape.getType().getRank() !=
        storeOp.getTargetType().getRank()) {
      return rewriter.notifyMatchFailure(
          storeOp, "expected `tensor.collapse_shape` rank to be the same as "
                   "the `iree_tensor_ext.dispatch.tensor.store` rank.");
    }
    RankedTensorType reshapeSrcType = collapseShape.getSrcType();
    ArrayRef<int64_t> reshapeSrcShape = reshapeSrcType.getShape();

    auto subspanOp =
        storeOp.getTarget()
            .template getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp) {
      return rewriter.notifyMatchFailure(
          storeOp, "expected `hal.interface.binding.subspan` target");
    }

    auto subspanType =
        cast<IREE::TensorExt::DispatchTensorType>(subspanOp.getType());
    SmallVector<Value> dynamicDims = subspanOp.getDynamicDims();

    // Verify the subspan shape against the shape of the slice being inserted.
    for (auto &&[i, subspanSize, group, size, offset] : llvm::enumerate(
             subspanType.getShape(), collapseShape.getReassociationIndices(),
             storeOp.getMixedSizes(), storeOp.getMixedOffsets())) {
      // No limitations on a single dimension.
      if (group.size() == 1) {
        continue;
      }

      // Check that each reassociation group contains at most a single dynamic
      // dimension.
      unsigned numDynamicDims =
          llvm::count_if(group, [reshapeSrcShape](int64_t dim) {
            return ShapedType::isDynamic(reshapeSrcShape[dim]);
          });
      if (numDynamicDims > 1) {
        return rewriter.notifyMatchFailure(
            collapseShape,
            "got multiple dynamic dimensions in group: " + std::to_string(i));
      }

      // Compute the inner dimension product. This requires all inner dims to
      // be static for the offset divisibility and subspan size checks below.
      int64_t innerDimSize = 1;
      bool hasStaticInnerDims = true;
      for (int64_t j : llvm::drop_begin(group)) {
        if (ShapedType::isDynamic(reshapeSrcShape[j])) {
          hasStaticInnerDims = false;
          break;
        }
        innerDimSize *= reshapeSrcShape[j];
      }

      // Check that the offset is divisible by the inner dimension product.
      std::optional<int64_t> maybeStaticOffset = getConstantIntValue(offset);
      if (maybeStaticOffset.has_value()) {
        // Static offset case: verify it's a multiple of the inner dim product.
        int64_t staticOffset = maybeStaticOffset.value();
        if (staticOffset != 0) {
          if (!hasStaticInnerDims) {
            return rewriter.notifyMatchFailure(
                storeOp, "has non-zero static offset but dynamic inner dims");
          }
          if (staticOffset % innerDimSize != 0) {
            return rewriter.notifyMatchFailure(
                storeOp, "has a static offset that is not a multiple of the "
                         "inner dimension product");
          }
        }
      } else {
        // Dynamic offset case: check if it comes from an affine.apply where
        // the inner dim product is a multiplicand.
        if (!hasStaticInnerDims) {
          return rewriter.notifyMatchFailure(
              storeOp, "has dynamic offset with dynamic inner dimensions");
        }

        Value offsetValue = cast<Value>(offset);
        auto applyOp = offsetValue.getDefiningOp<affine::AffineApplyOp>();
        if (!applyOp) {
          return rewriter.notifyMatchFailure(
              storeOp, "has dynamic offset in dimension to be expanded that is "
                       "not from affine.apply");
        }

        AffineMap map = applyOp.getAffineMap();
        AffineExpr expr = map.getResult(0);

        // Helper to check if an affine expression is provably divisible by
        // innerDimSize.
        std::function<bool(AffineExpr)> isDivisible =
            [&](AffineExpr e) -> bool {
          if (auto constExpr = dyn_cast<AffineConstantExpr>(e)) {
            return constExpr.getValue() % innerDimSize == 0;
          }
          if (auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(e)) {
            if (binaryExpr.getKind() == AffineExprKind::Mul) {
              // For multiplication, check if either operand makes it divisible.
              return isDivisible(binaryExpr.getLHS()) ||
                     isDivisible(binaryExpr.getRHS());
            }
            if (binaryExpr.getKind() == AffineExprKind::Add) {
              // For addition, both operands must be divisible.
              return isDivisible(binaryExpr.getLHS()) &&
                     isDivisible(binaryExpr.getRHS());
            }
          }
          return false;
        };

        if (!isDivisible(expr)) {
          return rewriter.notifyMatchFailure(
              storeOp, "dynamic offset from affine.apply is not provably a "
                       "multiple of inner dimension product");
        }
      }

      // In case of a dynamic dimension, check that the subspan size is dynamic
      // as well. Dynamic -> static is unlikely and eases the implementation
      // below.
      if (numDynamicDims >= 1) {
        if (ShapedType::isStatic(subspanSize)) {
          return rewriter.notifyMatchFailure(
              subspanOp, "collapsing and storing with dynamic dimension into a "
                         "static dimension is not supported");
        }
        continue;
      }

      // Handle static sizes - use innerDimSize computed above.
      if (subspanSize % innerDimSize != 0) {
        return rewriter.notifyMatchFailure(
            storeOp, "subspan type indivisible by expanded shape");
      }
    }

    AffineExpr d0, d1;
    bindDims(rewriter.getContext(), d0, d1);
    AffineExpr div = d0.ceilDiv(d1);

    rewriter.setInsertionPoint(subspanOp);
    Location loc = collapseShape.getLoc();
    SmallVector<int64_t> expandedSubspanShape;
    SmallVector<OpFoldResult> expandedSizes;
    SmallVector<Value> newSubspanDynamicDims;
    // Store offset info for deferred computation. The offset value may be
    // defined in a nested region that doesn't dominate the subspan op, so we
    // defer creating affine.apply ops until after moving the insertion point.
    struct OffsetInfo {
      OpFoldResult offset;
      int64_t innerDimSize;
      bool needsComputation;
    };
    SmallVector<OffsetInfo> offsetInfos;
    OpFoldResult zero = rewriter.getIndexAttr(0);
    size_t dynIndex = 0;
    for (auto [subspanSize, group, offset, size] : llvm::zip_equal(
             subspanType.getShape(), collapseShape.getReassociationIndices(),
             storeOp.getMixedOffsets(), storeOp.getMixedSizes())) {
      if (group.size() == 1) {
        expandedSizes.push_back(size);
        offsetInfos.push_back({offset, 1, false});
        expandedSubspanShape.push_back(subspanSize);
        if (ShapedType::isDynamic(subspanSize)) {
          newSubspanDynamicDims.push_back(dynamicDims[dynIndex++]);
        }
        continue;
      }

      SmallVector<int64_t> innerDimSizes =
          llvm::map_to_vector(group, [reshapeSrcShape](int64_t dim) {
            return reshapeSrcShape[dim];
          });
      int64_t totalSize = getProductExcludingDynamic(innerDimSizes);
      int64_t innerDimSize =
          getProductExcludingDynamic(llvm::drop_begin(innerDimSizes));

      // The first size can be calculated in the following way:
      // 1. In case of a dynamic expanded dimension, note that it's guaranteed
      // above that each group can only contain a single dynamic dimension, so
      // we can calculate the remainder inner dimension size and divide the
      // single dynamic dimension by it.
      // 2. In case of a static expanded dimension with a dynamic subspan size,
      // use the expanded dimension size directly.
      // 3. In case of a static expanded dimension with a static subspan size,
      // divide the subspan size by the inner dimension product size.
      const int64_t firstDimSize = innerDimSizes[0];
      if (ShapedType::isDynamic(firstDimSize)) {
        OpFoldResult totalSizeAttr = rewriter.getIndexAttr(totalSize);
        Value newDynamicDim =
            affine::makeComposedAffineApply(
                rewriter, loc, div, {dynamicDims[dynIndex++], totalSizeAttr})
                .getResult();
        expandedSizes.push_back(newDynamicDim);
        newSubspanDynamicDims.push_back(newDynamicDim);
        expandedSubspanShape.push_back(firstDimSize);
      } else if (ShapedType::isDynamic(subspanSize)) {
        expandedSizes.push_back(rewriter.getIndexAttr(firstDimSize));
        expandedSubspanShape.push_back(firstDimSize);
      } else {
        assert(subspanSize % innerDimSize == 0 &&
               "The subspan size should be a multiple of the inner dimension "
               "size of the reshape op source");
        expandedSizes.push_back(rewriter.getIndexAttr(firstDimSize));
        expandedSubspanShape.push_back(subspanSize / innerDimSize);
      }

      // Store offset info for deferred computation. We can't create the
      // affine.apply here because the offset value may not dominate the
      // current insertion point.
      offsetInfos.push_back({offset, innerDimSize, true});

      for (int64_t reshapeSrcSize : llvm::drop_begin(innerDimSizes)) {
        offsetInfos.push_back({zero, 1, false});
        expandedSubspanShape.push_back(reshapeSrcSize);
        if (ShapedType::isDynamic(reshapeSrcSize)) {
          OpFoldResult totalSizeAttr = rewriter.getIndexAttr(totalSize);
          Value newDynamicDim =
              affine::makeComposedAffineApply(
                  rewriter, loc, div, {dynamicDims[dynIndex++], totalSizeAttr})
                  .getResult();
          expandedSizes.push_back(newDynamicDim);
          newSubspanDynamicDims.push_back(newDynamicDim);
        } else {
          expandedSizes.push_back(rewriter.getIndexAttr(reshapeSrcSize));
        }
      }
    }

    auto newSubspanTensorType = RankedTensorType::get(
        expandedSubspanShape, reshapeSrcType.getElementType());
    auto newSubspanType = IREE::TensorExt::DispatchTensorType::get(
        subspanType.getAccess(), newSubspanTensorType);

    Value newSubspanOp;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(subspanOp);
      newSubspanOp = IREE::HAL::InterfaceBindingSubspanOp::create(
          rewriter, subspanOp.getLoc(), newSubspanType, subspanOp.getLayout(),
          subspanOp.getBinding(), subspanOp.getByteOffset(),
          newSubspanDynamicDims, subspanOp.getAlignmentAttr(),
          subspanOp.getDescriptorFlagsAttr());
    }

    SmallVector<OpFoldResult> expandedStrides(reshapeSrcShape.size(),
                                              rewriter.getIndexAttr(1));
    SmallVector<int64_t> expandedStaticDims;
    SmallVector<Value> expandedDynamicDims;
    dispatchIndexOpFoldResults(expandedSizes, expandedDynamicDims,
                               expandedStaticDims);
    rewriter.setInsertionPoint(storeOp);

    // Now compute the expanded offsets at the store's insertion point where the
    // offset values are guaranteed to dominate.
    SmallVector<OpFoldResult> expandedOffsets;
    for (const OffsetInfo &info : offsetInfos) {
      if (info.needsComputation) {
        OpFoldResult innerDimSizeAttr =
            rewriter.getIndexAttr(info.innerDimSize);
        expandedOffsets.push_back(affine::makeComposedFoldedAffineApply(
            rewriter, loc, div, {info.offset, innerDimSizeAttr}));
      } else {
        expandedOffsets.push_back(info.offset);
      }
    }

    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorStoreOp>(
        storeOp, collapseShape.getSrc(), newSubspanOp, expandedDynamicDims,
        expandedOffsets, expandedSizes, expandedStrides);
    return success();
  }
};

/// Folds iree_tensor_ext.bitcast that only changes the innermost dimension
/// into the source hal.interface.binding.subspan.
///
/// This pattern matches the following:
///   %subspan = hal.interface.binding.subspan ... :
///       !dispatch_tensor<readonly:tensor<MxNx(2K)xf4E2M1FN>>
///   %tensor = dispatch.tensor.load %subspan :
///       !dispatch_tensor<readonly:tensor<MxNx(2K)xf4E2M1FN>> ->
///       tensor<MxNx(2K)xf4E2M1FN>
///   %bitcast = iree_tensor_ext.bitcast %tensor :
///       tensor<MxNx(2K)xf4E2M1FN> -> tensor<MxNxKxi8>
///
/// And transforms it into:
///   %subspan = hal.interface.binding.subspan ... :
///       !dispatch_tensor<readonly:tensor<MxNxKxi8>>
///   %tensor = dispatch.tensor.load %subspan :
///       !dispatch_tensor<readonly:tensor<MxNxKxi8>> -> tensor<MxNxKxi8>
///
/// This is valid when:
/// - Only the innermost dimension size and element type change
/// - The bitcast represents a reinterpretation (e.g., f4E2M1FN -> i8)
/// - The total byte size of the innermost dimension remains constant
/// - The tensor has no encoding and the load is a full slice
struct FoldInnerBitcastIntoInterfaceTensorLoad
    : OpRewritePattern<IREE::TensorExt::BitCastOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::TensorExt::BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {
    Value bitcastSrc = bitcastOp.getSource();
    auto loadOp =
        bitcastSrc.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    if (!loadOp) {
      return failure();
    }

    auto subspanOp = loadOp.getSource()
                         .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp) {
      return failure();
    }

    auto bitcastSrcType = cast<RankedTensorType>(bitcastSrc.getType());
    auto bitcastResType = cast<RankedTensorType>(bitcastOp.getType());

    // Verify that only the inner most dimension is changed by the bitcast by
    // comparing dynamic and static sizes for equality.
    if (bitcastOp.getSourceDims() != bitcastOp.getResultDims() ||
        bitcastSrcType.getShape().drop_back() !=
            bitcastResType.getShape().drop_back() ||
        ShapedType::isDynamic(bitcastSrcType.getShape().back())) {
      return failure();
    }

    // Fail if the inner most dim is sliced by the load or if this is an encoded
    // tensor.
    auto subspanType =
        cast<IREE::TensorExt::DispatchTensorType>(subspanOp.getType());
    auto subspanTensorType = cast<RankedTensorType>(subspanType.getBoundType());
    if (subspanTensorType.getEncoding() ||
        subspanTensorType.getShape().back() !=
            bitcastSrcType.getShape().back()) {
      return failure();
    }

    int64_t newInnerSize = bitcastResType.getShape().back();
    SmallVector<int64_t> newSubspanShape(subspanType.getShape());
    newSubspanShape.back() = newInnerSize;

    auto newSubspanType = IREE::TensorExt::DispatchTensorType::get(
        subspanType.getAccess(),
        RankedTensorType::get(newSubspanShape,
                              bitcastResType.getElementType()));

    // Byte offset and byte alignment remain the same after folding the cast.
    // Simply create a new binding with the new type.
    rewriter.setInsertionPoint(subspanOp);
    Value newSubspanOp = IREE::HAL::InterfaceBindingSubspanOp::create(
        rewriter, subspanOp.getLoc(), newSubspanType, subspanOp.getLayout(),
        subspanOp.getBinding(), subspanOp.getByteOffset(),
        subspanOp.getDynamicDims(), subspanOp.getAlignmentAttr(),
        subspanOp.getDescriptorFlagsAttr());

    rewriter.setInsertionPoint(bitcastOp);
    SmallVector<int64_t> newSizes(loadOp.getStaticSizes());
    newSizes.back() = newInnerSize;
    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorLoadOp>(
        bitcastOp, bitcastResType, newSubspanOp, loadOp.getSourceDims(),
        loadOp.getOffsets(), loadOp.getSizes(), loadOp.getStrides(),
        loadOp.getStaticOffsets(), newSizes, loadOp.getStaticStrides());

    return success();
  }
};

struct FoldInnerBitcastIntoInterfaceTensorStore
    : OpRewritePattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using OpRewritePattern<
      IREE::TensorExt::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto bitcastOp =
        storeOp.getValue().getDefiningOp<IREE::TensorExt::BitCastOp>();
    if (!bitcastOp) {
      return failure();
    }

    Value bitcastSrc = bitcastOp.getSource();
    auto bitcastSrcType = cast<RankedTensorType>(bitcastSrc.getType());
    auto bitcastResType = cast<RankedTensorType>(bitcastOp.getType());

    // Verify that only the inner most dimension is changed by the bitcast by
    // comparing dynamic and static sizes for equality.
    if (bitcastOp.getSourceDims() != bitcastOp.getResultDims() ||
        bitcastSrcType.getShape().drop_back() !=
            bitcastResType.getShape().drop_back() ||
        ShapedType::isDynamic(bitcastSrcType.getShape().back())) {
      return failure();
    }

    auto subspanOp =
        storeOp.getTarget()
            .template getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp) {
      return failure();
    }

    auto subspanType =
        cast<IREE::TensorExt::DispatchTensorType>(subspanOp.getType());
    auto subspanTensorType = cast<RankedTensorType>(subspanType.getBoundType());
    if (subspanTensorType.getEncoding() ||
        subspanTensorType.getShape().back() !=
            bitcastResType.getShape().back()) {
      return failure();
    }

    int64_t newInnerSize = bitcastSrcType.getShape().back();
    SmallVector<int64_t> newSubspanShape(subspanType.getShape());
    newSubspanShape.back() = newInnerSize;

    auto newSubspanTensorType =
        RankedTensorType::get(newSubspanShape, bitcastSrcType.getElementType());
    auto newSubspanType = IREE::TensorExt::DispatchTensorType::get(
        subspanType.getAccess(), newSubspanTensorType);

    rewriter.setInsertionPointAfter(subspanOp);
    Value newSubspanOp = IREE::HAL::InterfaceBindingSubspanOp::create(
        rewriter, subspanOp.getLoc(), newSubspanType, subspanOp.getLayout(),
        subspanOp.getBinding(), subspanOp.getByteOffset(),
        subspanOp.getDynamicDims(), subspanOp.getAlignmentAttr(),
        subspanOp.getDescriptorFlagsAttr());

    SmallVector<int64_t> newSizes(storeOp.getStaticSizes());
    newSizes.back() = newInnerSize;
    rewriter.setInsertionPoint(storeOp);
    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorStoreOp>(
        storeOp, TypeRange{}, bitcastSrc, newSubspanOp, storeOp.getTargetDims(),
        storeOp.getOffsets(), storeOp.getSizes(), storeOp.getStrides(),
        storeOp.getStaticOffsets(), newSizes, storeOp.getStaticStrides());
    return success();
  }
};

/// Similar to FoldInnerBitcastIntoInterfaceTensorLoad, but handles the
/// bufferized load instead:
///   hal.interface.binding.subspan -> amdgpu.fat_raw_buffer_cast ->
///   iree_codegen.load_from_buffer -> iree_tensor_ext.bitcast
struct FoldInnerBitcastIntoLoadFromBuffer
    : OpRewritePattern<IREE::TensorExt::BitCastOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::TensorExt::BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {
    Value bitcastSrc = bitcastOp.getSource();

    // TODO(#22712): This pattern matching is fragile. Simplify this by using
    // a memref.bitcast or similar operation once available.

    // Step 1: Check for iree_codegen.load_from_buffer.
    auto loadOp = bitcastSrc.getDefiningOp<IREE::Codegen::LoadFromBufferOp>();
    if (!loadOp) {
      return rewriter.notifyMatchFailure(bitcastOp, "no load_from_buffer");
    }
    Value loadBuffer = loadOp.getBuffer();

    // Step 2: Check for amdgpu.fat_raw_buffer_cast.
    auto bufferCastOp = loadBuffer.getDefiningOp<amdgpu::FatRawBufferCastOp>();
    if (!bufferCastOp) {
      return rewriter.notifyMatchFailure(bitcastOp, "no fat_raw_buffer_cast");
    }
    Value bufferCastSource = bufferCastOp.getSource();

    // Step 3: Check for hal.interface.binding.subspan.
    auto subspanOp =
        bufferCastSource.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp) {
      return rewriter.notifyMatchFailure(bitcastOp, "no binding subspan");
    }

    // Now we have the full bufferized load chain verified.
    auto subspanType = cast<MemRefType>(subspanOp.getType());
    auto bitcastSrcType = cast<RankedTensorType>(bitcastSrc.getType());
    auto bitcastResType = cast<RankedTensorType>(bitcastOp.getType());

    if (bitcastOp.getSourceDims() != bitcastOp.getResultDims() ||
        bitcastSrcType.getShape().drop_back() !=
            bitcastResType.getShape().drop_back() ||
        ShapedType::isDynamic(bitcastSrcType.getShape().back())) {
      return rewriter.notifyMatchFailure(
          bitcastOp,
          "expected that only the innermost dimension is changed by the "
          "bitcast op");
    }

    if (subspanType.getShape().back() != bitcastSrcType.getShape().back()) {
      return rewriter.notifyMatchFailure(
          bitcastOp,
          "innermost dimension doesn't match between subspan and tensor");
    }

    int64_t newInnerSize = bitcastResType.getShape().back();
    int64_t oldInnerSize = bitcastSrcType.getShape().back();

    // Adjust the innermost dimension size.
    MemRefLayoutAttrInterface newSubspanLayout;
    SmallVector<int64_t> newSubspanShape(subspanType.getShape());
    newSubspanShape.back() = newInnerSize;
    // Scale offset and strides by (newInnerSize / oldInnerSize).
    if (auto stridedLayout =
            dyn_cast_if_present<StridedLayoutAttr>(subspanType.getLayout())) {
      int64_t newOffset = stridedLayout.getOffset();
      if (ShapedType::isStatic(newOffset)) {
        newOffset = (newOffset * newInnerSize) / oldInnerSize;
      }
      SmallVector<int64_t> newStrides(stridedLayout.getStrides());
      for (size_t i = 0; i + 1 < newStrides.size(); ++i) {
        if (ShapedType::isStatic(newStrides[i])) {
          newStrides[i] = (newStrides[i] * newInnerSize) / oldInnerSize;
        }
      }
      newSubspanLayout =
          StridedLayoutAttr::get(rewriter.getContext(), newOffset, newStrides);
    }
    // Use the bitcast's result element type.
    auto newSubspanType =
        MemRefType::get(newSubspanShape, bitcastResType.getElementType(),
                        newSubspanLayout, subspanType.getMemorySpace());

    rewriter.setInsertionPoint(subspanOp);
    Value newSubspanOp = IREE::HAL::InterfaceBindingSubspanOp::create(
        rewriter, subspanOp.getLoc(), newSubspanType, subspanOp.getLayout(),
        subspanOp.getBinding(), subspanOp.getByteOffset(),
        subspanOp.getDynamicDims(), subspanOp.getAlignmentAttr(),
        subspanOp.getDescriptorFlagsAttr());

    rewriter.setInsertionPoint(bufferCastOp);
    Value newBufferCastOp = amdgpu::FatRawBufferCastOp::create(
        rewriter, bufferCastOp.getLoc(), newSubspanOp,
        bufferCastOp.getValidBytes(), bufferCastOp.getCacheSwizzleStride(),
        bufferCastOp.getBoundsCheck(), bufferCastOp.getResetOffset());

    rewriter.replaceOpWithNewOp<IREE::Codegen::LoadFromBufferOp>(
        loadOp, bitcastResType, newBufferCastOp);
    return success();
  }
};

//===--------------------------------------------------------------------====//
// Patterns to fold ops into iree_codegen.store_to/load_from_buffer
//===--------------------------------------------------------------------====//

/// Given an iree_codegen.load_from_buffer op or iree_codegen.store_to_buffer
/// op, and a list of reassociation indices, replace the memref operand of the
/// load_from_buffer or store_to_buffer op with a collapsed memref according to
/// the reassociations. The `tensorToMemrefOp` will be modified in place without
/// updating users or producers. The caller of this function is responsible for
/// updating producers and consumers to maintain valid IR.
template <typename OpTy>
static LogicalResult
collapseMemrefOperand(RewriterBase &rewriter, OpTy tensorToMemrefOp,
                      ArrayRef<ReassociationIndices> reassociations) {
  Value memref = tensorToMemrefOp.getBuffer();
  auto memrefType = cast<MemRefType>(memref.getType());
  if (!memref::CollapseShapeOp::isGuaranteedCollapsible(memrefType,
                                                        reassociations)) {
    return failure();
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfterValue(memref);
  Location loc = tensorToMemrefOp.getLoc();
  Value collapsedMemref =
      memref::CollapseShapeOp::create(rewriter, loc, memref, reassociations);
  rewriter.modifyOpInPlace(tensorToMemrefOp, [&]() {
    tensorToMemrefOp.getBufferMutable().assign(collapsedMemref);
  });
  return success();
}

/// Given an iree_codegen.load_from_buffer op or iree_codegen.store_to_buffer
/// op, a list of reassociation indices, and an output shape, replace the memref
/// operand of the load_from_buffer or store_to_buffer op with an expanded
/// memref according to the reassociations. The `tensorToMemrefOp` will be
/// modified in place without updating users or producers. The caller of this
/// function is responsible for updating producers and consumers to maintain
/// valid IR.
template <typename OpTy>
static LogicalResult
expandMemrefOperand(RewriterBase &rewriter, OpTy tensorToMemrefOp,
                    ArrayRef<ReassociationIndices> reassociations,
                    SmallVector<OpFoldResult> &mixedOutputShape) {
  Value memref = tensorToMemrefOp.getBuffer();
  auto memrefType = cast<MemRefType>(memref.getType());
  SmallVector<int64_t> expandedShape;
  SmallVector<Value> dynamicValues;
  std::tie(expandedShape, dynamicValues) =
      decomposeMixedValues(mixedOutputShape);
  FailureOr<MemRefType> expandedMemrefType =
      memref::ExpandShapeOp::computeExpandedType(memrefType, expandedShape,
                                                 reassociations);
  if (failed(expandedMemrefType)) {
    return failure();
  }
  Location loc = tensorToMemrefOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  dynamicValues.push_back(memref);
  setInsertionPointAfterLastValue(rewriter, dynamicValues);
  Value expandedMemref =
      memref::ExpandShapeOp::create(rewriter, loc, *expandedMemrefType, memref,
                                    reassociations, mixedOutputShape);
  rewriter.modifyOpInPlace(tensorToMemrefOp, [&]() {
    tensorToMemrefOp.getBufferMutable().assign(expandedMemref);
  });
  return success();
}

/// Fold a tensor.expand_shape into a consumer iree_codegen.store_to_buffer op
/// by collapsing the memref operand of the store_to_buffer, and replacing the
/// tensor operand with the source of the expand_shape.
struct FoldExpandShapeIntoStoreToBuffer
    : OpRewritePattern<IREE::Codegen::StoreToBufferOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::Codegen::StoreToBufferOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto expandOp = storeOp.getTensor().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp) {
      return failure();
    }
    if (failed(collapseMemrefOperand(rewriter, storeOp,
                                     expandOp.getReassociationIndices()))) {
      return rewriter.notifyMatchFailure(storeOp, "memref is not collapsible");
    }
    rewriter.modifyOpInPlace(storeOp, [&]() {
      storeOp.getTensorMutable().assign(expandOp.getSrc());
    });
    return success();
  }
};

/// Fold a tensor.collapse_shape into a consumer iree_codegen.store_to_buffer op
/// by expanding the memref operand of the store_to_buffer, and replacing the
/// tensor operand with the source of the collapse_shape.
struct FoldCollapseShapeIntoStoreToBuffer
    : OpRewritePattern<IREE::Codegen::StoreToBufferOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::Codegen::StoreToBufferOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto collapseOp =
        storeOp.getTensor().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseOp) {
      return failure();
    }
    Value collapseSrc = collapseOp.getSrc();
    SmallVector<OpFoldResult> mixedOutputShape =
        tensor::getMixedSizes(rewriter, collapseSrc.getLoc(), collapseSrc);
    if (failed(expandMemrefOperand(rewriter, storeOp,
                                   collapseOp.getReassociationIndices(),
                                   mixedOutputShape))) {
      return rewriter.notifyMatchFailure(storeOp, "memref is not expandable");
    }
    rewriter.modifyOpInPlace(
        storeOp, [&]() { storeOp.getTensorMutable().assign(collapseSrc); });
    return success();
  }
};

/// Fold a tensor.collapse_shape into a producer iree_codegen.load_from_buffer
/// op by collapsing the memref operand of the load_from_buffer, and replacing
/// the collapse_shape with the collapsed load_from_buffer op.
struct FoldCollapseShapeIntoLoadFromBuffer
    : OpRewritePattern<IREE::Codegen::LoadFromBufferOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::Codegen::LoadFromBufferOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!loadOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(loadOp, "load op has multiple uses");
    }
    auto collapseOp =
        dyn_cast<tensor::CollapseShapeOp>(*loadOp->getUsers().begin());
    if (!collapseOp) {
      return failure();
    }
    if (failed(collapseMemrefOperand(rewriter, loadOp,
                                     collapseOp.getReassociationIndices()))) {
      return rewriter.notifyMatchFailure(loadOp, "memref is not collapsible");
    }
    rewriter.modifyOpInPlace(loadOp, [&]() {
      loadOp->getOpResult(0).setType(collapseOp.getResultType());
    });
    rewriter.replaceOp(collapseOp, loadOp);
    return success();
  }
};

/// Fold a tensor.expand_shape into a producer iree_codegen.load_from_buffer op
/// by expanding the memref operand of the load_from_buffer, and replacing the
/// expand_shape with the expanded load_from_buffer op.
struct FoldExpandShapeIntoLoadFromBuffer
    : OpRewritePattern<IREE::Codegen::LoadFromBufferOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::Codegen::LoadFromBufferOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!loadOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(loadOp, "load op has multiple uses");
    }
    auto expandOp =
        dyn_cast<tensor::ExpandShapeOp>(*loadOp->getUsers().begin());
    if (!expandOp) {
      return failure();
    }
    SmallVector<OpFoldResult> mixedOutputShape = expandOp.getMixedOutputShape();
    if (failed(expandMemrefOperand(rewriter, loadOp,
                                   expandOp.getReassociationIndices(),
                                   mixedOutputShape))) {
      return rewriter.notifyMatchFailure(loadOp, "memref is not expandable");
    }
    rewriter.modifyOpInPlace(loadOp, [&]() {
      loadOp->getOpResult(0).setType(expandOp.getResultType());
    });
    DominanceInfo domInfo;
    moveOpAfterLastOperand(rewriter, domInfo, loadOp);
    rewriter.replaceOp(expandOp, loadOp);
    return success();
  }
};

struct FoldReshapeIntoInterfaceTensorPass final
    : impl::FoldReshapeIntoInterfaceTensorPassBase<
          FoldReshapeIntoInterfaceTensorPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateReshapeToInterfaceTensorPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateReshapeToInterfaceTensorPatterns(RewritePatternSet &patterns) {
  patterns.insert<FoldCollapseShapeIntoInterfaceTensorLoad,
                  FoldCollapseShapeIntoInterfaceTensorStore,
                  FoldCollapseShapeIntoInterfaceTensorStoreFullSlice,
                  FoldExpandShapeIntoInterfaceTensorLoad,
                  FoldExpandShapeIntoInterfaceTensorStore,
                  FoldInnerBitcastIntoInterfaceTensorLoad,
                  FoldInnerBitcastIntoInterfaceTensorStore>(
      patterns.getContext());
}

void populateFoldTensorReshapeIntoBufferPatterns(RewritePatternSet &patterns) {
  patterns.insert<
      FoldCollapseShapeIntoLoadFromBuffer, FoldExpandShapeIntoLoadFromBuffer,
      FoldCollapseShapeIntoStoreToBuffer, FoldExpandShapeIntoStoreToBuffer,
      FoldInnerBitcastIntoLoadFromBuffer>(patterns.getContext());
}

} // namespace mlir::iree_compiler
