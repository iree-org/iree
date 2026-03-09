// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_PROPAGATERESHAPESBYEXPANSIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Calculate the expanded shape of `dest` if it can be expanded with the inner
/// expanded sizes of `sliceStaticSizes`. Returns failure if such expansion is
/// not possible.
static LogicalResult
getExpandedShape(SmallVector<ReassociationIndices> reIndices,
                 ArrayRef<int64_t> sliceStaticSizes, Value dest,
                 SmallVectorImpl<int64_t> &expandedShape,
                 SmallVectorImpl<int64_t> &totalInnerSizes) {
  auto destType = dyn_cast<ShapedType>(dest.getType());
  if (!destType) {
    return failure();
  }
  // TODO (nirvedhmeshram): Support rank reducing parallel_insert_slice.
  if (reIndices.size() != destType.getShape().size()) {
    return failure();
  }
  // Iterator to insert outer sizes.
  auto outerShapeIdx = 0;
  for (auto [reassociations, destSize] :
       llvm::zip_equal(reIndices, destType.getShape())) {
    // Dynamic destination dims that are not getting expanded are allowed.
    if (ShapedType::isDynamic(destSize) && reassociations.size() == 1) {
      expandedShape.insert(expandedShape.begin() + outerShapeIdx, destSize);
      outerShapeIdx++;
      totalInnerSizes.push_back(1);
      continue;
    }
    // Dynamic destination dims that are expanded are currently unsupported but
    // this support can be added if needed.
    if (ShapedType::isDynamic(destSize)) {
      return failure();
    }
    int64_t totalInnerSize = 1;
    for (int64_t reasociation : llvm::drop_begin(reassociations)) {
      int64_t expandedInnerSize = sliceStaticSizes[reasociation];
      // It is not safe to do this pattern if inner dimensions are dynamic.
      if (ShapedType::isDynamic(expandedInnerSize)) {
        return failure();
      }
      expandedShape.push_back(expandedInnerSize);
      totalInnerSize *= expandedInnerSize;
    }
    if (destSize % totalInnerSize != 0) {
      return failure();
    }
    totalInnerSizes.push_back(totalInnerSize);
    // insert the outer size in front of any inner sizes.
    expandedShape.insert(expandedShape.begin() + outerShapeIdx,
                         destSize / totalInnerSize);
    // set up the iterator for the next uncollapsed dimension.
    outerShapeIdx = expandedShape.size();
  }
  return success();
}

/// Calculate the collapsed shape of `dest` by collapsing dimensions according
/// to `reIndices`. Returns failure if such collapse is not possible.
/// `totalInnerSizes` stores the product of inner dimension sizes per
/// reassociation group (for stride computation). Converse of getExpandedShape.
static LogicalResult
getCollapsedShape(SmallVector<ReassociationIndices> reIndices, Value dest,
                  SmallVectorImpl<int64_t> &collapsedShape,
                  SmallVectorImpl<int64_t> &totalInnerSizes) {
  auto destType = dyn_cast<ShapedType>(dest.getType());
  if (!destType) {
    return failure();
  }
  // Verify total expanded rank matches dest rank.
  int64_t totalExpandedDims = 0;
  for (const auto &group : reIndices) {
    totalExpandedDims += group.size();
  }
  if (totalExpandedDims != destType.getRank()) {
    return failure();
  }
  ArrayRef<int64_t> destShape = destType.getShape();
  for (const auto &reassociations : reIndices) {
    int64_t collapsedSize = 1;
    int64_t totalInnerSize = 1;
    bool hasDynamic = false;
    for (auto [idx, dim] : llvm::enumerate(reassociations)) {
      int64_t dimSize = destShape[dim];
      if (ShapedType::isDynamic(dimSize)) {
        hasDynamic = true;
        break;
      }
      collapsedSize *= dimSize;
      if (idx > 0) {
        totalInnerSize *= dimSize;
      }
    }
    // Dynamic destination dims that are not being collapsed are allowed.
    if (hasDynamic && reassociations.size() == 1) {
      collapsedShape.push_back(destShape[reassociations[0]]);
      totalInnerSizes.push_back(1);
      continue;
    }
    // Dynamic destination dims that are collapsed are currently unsupported.
    if (hasDynamic) {
      return failure();
    }
    collapsedShape.push_back(collapsedSize);
    totalInnerSizes.push_back(totalInnerSize);
  }
  return success();
}

/// Check if the users of the expanded scf.forall destination can be updated to
/// account for the expand. If not we bail out. There are two supported users
/// which are extract_slice -> expand_shape with the same exact reassociation
/// map as the collapse op to be hoisted out or the root parallel_insert_slice.
static LogicalResult verifyAndCollectExpandableUsers(
    Value insertDest, SmallVector<ReassociationIndices> reIndices,
    tensor::ParallelInsertSliceOp parallelInsertOp,
    SmallVector<tensor::ExtractSliceOp> &expandableUsers) {
  for (Operation *user : insertDest.getUsers()) {
    if (user == parallelInsertOp) {
      continue;
    }
    auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!extractSliceOp) {
      return failure();
    }
    if (extractSliceOp.getMixedSizes() != parallelInsertOp.getMixedSizes()) {
      return failure();
    }
    if (extractSliceOp.getMixedOffsets() !=
        parallelInsertOp.getMixedOffsets()) {
      return failure();
    }
    for (Operation *user : extractSliceOp->getUsers()) {
      auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(user);
      if (!expandShapeOp) {
        return failure();
      }
      SmallVector<ReassociationIndices> expandReIndices =
          expandShapeOp.getReassociationIndices();
      if (reIndices != expandReIndices) {
        return failure();
      }
    }
    expandableUsers.push_back(extractSliceOp);
  }
  return success();
}

/// Check if the users of the collapsed scf.forall destination can be updated
/// to account for the collapse. Supported users are extract_slice ->
/// collapse_shape with the same reassociation map as the expand op to be
/// hoisted out, or the root parallel_insert_slice.
/// Converse of verifyAndCollectExpandableUsers.
static LogicalResult verifyAndCollectCollapsableUsers(
    Value insertDest, SmallVector<ReassociationIndices> reIndices,
    tensor::ParallelInsertSliceOp parallelInsertOp,
    SmallVector<tensor::ExtractSliceOp> &collapsableUsers) {
  for (Operation *user : insertDest.getUsers()) {
    if (user == parallelInsertOp) {
      continue;
    }
    auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!extractSliceOp) {
      return failure();
    }
    if (extractSliceOp.getMixedSizes() != parallelInsertOp.getMixedSizes()) {
      return failure();
    }
    if (extractSliceOp.getMixedOffsets() !=
        parallelInsertOp.getMixedOffsets()) {
      return failure();
    }
    for (Operation *user : extractSliceOp->getUsers()) {
      auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(user);
      if (!collapseShapeOp) {
        return failure();
      }
      SmallVector<ReassociationIndices> collapseReIndices =
          collapseShapeOp.getReassociationIndices();
      if (reIndices != collapseReIndices) {
        return failure();
      }
    }
    collapsableUsers.push_back(extractSliceOp);
  }
  return success();
}

/// Utility to expand the pre-verified expandable users of the scf.forall
/// output.
static void
expandVerifiedUsers(PatternRewriter &rewriter, Location loc, MLIRContext *ctx,
                    SmallVector<tensor::ExtractSliceOp> expandableUsers,
                    SmallVector<int64_t> totalInnerSizes,
                    SmallVector<ReassociationIndices> reIndices,
                    scf::ForallOp forallOp,
                    tensor::ParallelInsertSliceOp parallelInsertOp) {
  // compute the offsets,sizes,strides in the expanded dimensions.
  auto computeExpandedAccess = [&](ArrayRef<OpFoldResult> mixedOffsets,
                                   ShapedType resultType)
      -> std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
                    SmallVector<OpFoldResult>> {
    SmallVector<OpFoldResult> expandedOffsets;
    auto expandedOffsetsIdx = 0;

    for (auto [index, offset] : llvm::enumerate(mixedOffsets)) {
      // Add zero offsets for the extra dimensions from reIndices.
      for (size_t i = 1, e = reIndices[index].size(); i < e; ++i) {
        expandedOffsets.push_back(getAsIndexOpFoldResult(ctx, 0));
      }
      Value offsetVal = getValueOrCreateConstantIndexOp(rewriter, loc, offset);
      // Make sure we insert after offset.
      rewriter.setInsertionPointAfterValue(offsetVal);
      // Compute the outer dimension expression.
      AffineExpr s0, s1;
      bindSymbols(rewriter.getContext(), s0, s1);
      AffineExpr outerDimExpr = (s0).floorDiv(s1);
      // Insert computed offset using affine expression.
      expandedOffsets.insert(
          expandedOffsets.begin() + expandedOffsetsIdx,
          affine::makeComposedFoldedAffineApply(
              rewriter, loc, outerDimExpr,
              {offsetVal, rewriter.getIndexAttr(totalInnerSizes[index])}));

      expandedOffsetsIdx = expandedOffsets.size();
    }
    SmallVector<OpFoldResult> expandedSizes =
        getAsIndexOpFoldResult(ctx, resultType.getShape());
    SmallVector<OpFoldResult> expandedStrides(resultType.getRank(),
                                              rewriter.getIndexAttr(1));
    return {expandedOffsets, expandedSizes, expandedStrides};
  };
  auto collapseShapeOp =
      parallelInsertOp.getSource().getDefiningOp<tensor::CollapseShapeOp>();
  RankedTensorType resultType = collapseShapeOp.getSrcType();
  auto [expandedOffsets, expandedSizes, expandedStrides] =
      computeExpandedAccess(parallelInsertOp.getMixedOffsets(), resultType);
  rewriter.setInsertionPoint(parallelInsertOp);
  rewriter.replaceOpWithNewOp<tensor::ParallelInsertSliceOp>(
      parallelInsertOp, collapseShapeOp.getSrc(), parallelInsertOp.getDest(),
      expandedOffsets, expandedSizes, expandedStrides);
  for (tensor::ExtractSliceOp extractSliceOp : expandableUsers) {
    rewriter.setInsertionPoint(extractSliceOp);
    auto newExtractSliceOp =
        rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
            extractSliceOp, resultType, extractSliceOp.getSource(),
            expandedOffsets, expandedSizes, expandedStrides);
    for (Operation *user : newExtractSliceOp->getUsers()) {
      auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(user);
      expandShapeOp->replaceAllUsesWith(newExtractSliceOp);
    }
  }
  return;
}

/// Utility to collapse the pre-verified collapsable users of the scf.forall
/// output. Linearizes expanded offsets into collapsed offsets.
/// Converse of expandVerifiedUsers.
/// `collapsedInsertSizes` provides the sizes for the collapsed insert/extract
/// operations (handles rank-reducing inserts where expand source rank < dest
/// rank).
static void
collapseVerifiedUsers(PatternRewriter &rewriter, Location loc, MLIRContext *ctx,
                      SmallVector<tensor::ExtractSliceOp> collapsableUsers,
                      SmallVector<ReassociationIndices> reIndices,
                      ArrayRef<int64_t> destShape,
                      ArrayRef<int64_t> collapsedInsertSizes,
                      scf::ForallOp forallOp,
                      tensor::ParallelInsertSliceOp parallelInsertOp) {
  // Compute the offsets, sizes, strides in the collapsed dimensions.
  auto computeCollapsedAccess = [&](ArrayRef<OpFoldResult> mixedOffsets)
      -> std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
                    SmallVector<OpFoldResult>> {
    SmallVector<OpFoldResult> collapsedOffsets;

    for (const auto &reassociations : reIndices) {
      if (reassociations.size() == 1) {
        // No collapsing needed for this dimension, pass offset through.
        collapsedOffsets.push_back(mixedOffsets[reassociations[0]]);
        continue;
      }
      // Linearize: off_0 * stride_0 + off_1 * stride_1 + ...
      // where stride_i = product of sizes of dims after i in the group.
      SmallVector<OpFoldResult> operands;
      AffineExpr linearExpr = getAffineConstantExpr(0, ctx);
      for (auto [idx, dim] : llvm::enumerate(reassociations)) {
        int64_t stride = 1;
        for (size_t j = idx + 1; j < reassociations.size(); ++j) {
          stride *= destShape[reassociations[j]];
        }
        AffineExpr sym = getAffineSymbolExpr(operands.size(), ctx);
        linearExpr = linearExpr + sym * stride;
        operands.push_back(mixedOffsets[dim]);
      }
      // Set insertion point after offset values.
      for (auto operand : operands) {
        Value val = getValueOrCreateConstantIndexOp(rewriter, loc, operand);
        rewriter.setInsertionPointAfterValue(val);
      }
      collapsedOffsets.push_back(
          affine::makeComposedFoldedAffineApply(
              rewriter, loc, linearExpr, operands));
    }
    SmallVector<OpFoldResult> collapsedSizes =
        getAsIndexOpFoldResult(ctx, collapsedInsertSizes);
    SmallVector<OpFoldResult> collapsedStrides(collapsedInsertSizes.size(),
                                               rewriter.getIndexAttr(1));
    return {collapsedOffsets, collapsedSizes, collapsedStrides};
  };
  auto expandShapeOp =
      parallelInsertOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
  auto [collapsedOffsets, collapsedSizes, collapsedStrides] =
      computeCollapsedAccess(parallelInsertOp.getMixedOffsets());
  rewriter.setInsertionPoint(parallelInsertOp);
  rewriter.replaceOpWithNewOp<tensor::ParallelInsertSliceOp>(
      parallelInsertOp, expandShapeOp.getSrc(), parallelInsertOp.getDest(),
      collapsedOffsets, collapsedSizes, collapsedStrides);
  for (tensor::ExtractSliceOp extractSliceOp : collapsableUsers) {
    rewriter.setInsertionPoint(extractSliceOp);
    // Compute result type from collapsed insert sizes for the extract_slice.
    auto srcType = expandShapeOp.getSrcType();
    auto collapsedResultType =
        RankedTensorType::get(collapsedInsertSizes, srcType.getElementType());
    auto newExtractSliceOp =
        rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
            extractSliceOp, collapsedResultType, extractSliceOp.getSource(),
            collapsedOffsets, collapsedSizes, collapsedStrides);
    for (Operation *user : newExtractSliceOp->getUsers()) {
      auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(user);
      collapseShapeOp->replaceAllUsesWith(newExtractSliceOp);
    }
  }
  return;
}

/// This pattern expands destination of workgroup mapped scf.foralls by
/// hoisting out collapse_shape op consumed by its parallel.insert_slice op.
struct ExpandDestinationForallOp final
    : OpRewritePattern<tensor::ParallelInsertSliceOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::ParallelInsertSliceOp parallelInsertOp,
                                PatternRewriter &rewriter) const override {
    Location loc = parallelInsertOp.getLoc();
    MLIRContext *ctx = getContext();
    auto collapseOp =
        parallelInsertOp.getSource().getDefiningOp<tensor::CollapseShapeOp>();
    // No collapse op to hoist out.
    if (!collapseOp) {
      return failure();
    }

    // Ignore trivially foldable collapse ops.
    if (collapseOp.getSrcType().getRank() ==
        collapseOp.getResultType().getRank()) {
      return failure();
    }

    // Get the destination to expand.
    Value insertDest = parallelInsertOp.getDest();

    // Get the enclosing scf.forall op.
    OpResult tiedResult = parallelInsertOp.getTiedOpResult();
    int64_t tiedResultIdx = tiedResult.getResultNumber();

    auto forallOp = dyn_cast<scf::ForallOp>(tiedResult.getOwner());
    if (!forallOp) {
      return failure();
    }

    SmallVector<int64_t> expandedDestShape;
    SmallVector<int64_t> totalInnerSizes;
    // Get the shape of the outer expand which will be the new destination
    // of the scf.forall and the total size of inner dimensions per uncollapsed
    // dimension.
    SmallVector<ReassociationIndices> reIndices =
        collapseOp.getReassociationIndices();
    if (failed(getExpandedShape(reIndices, collapseOp.getSrcType().getShape(),
                                insertDest, expandedDestShape,
                                totalInnerSizes))) {
      return failure();
    }

    // We only want this pattern if the forall op result is being written to a
    // full slice, or an expandable buffer. Otherwise the hoisted collapse op is
    // not foldable.
    for (Operation *foralluser : tiedResult.getUsers()) {
      auto storeOp =
          dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(foralluser);
      if (storeOp && isFullSlice(storeOp, storeOp.getTargetType(),
                                 storeOp.getTargetDims())) {
        continue;
      }
      auto storeToBufferOp =
          dyn_cast<IREE::Codegen::StoreToBufferOp>(foralluser);
      if (!storeToBufferOp) {
        return failure();
      }
      MemRefType bufferType = storeToBufferOp.getBuffer().getType();
      if (failed(memref::ExpandShapeOp::computeExpandedType(
              bufferType, expandedDestShape, reIndices))) {
        return failure();
      }
    }

    // This allows us to assume that the extract/inserts in the loop are
    // disjoint and makes the application of this pattern safe.
    if (!forallOpHasMappingType<IREE::Codegen::WorkgroupMappingAttr>(
            forallOp)) {
      return failure();
    }

    // Verify that the users of destination are valid to expand and collect all
    // such users.
    SmallVector<tensor::ExtractSliceOp> expandableUsers;
    if (failed(verifyAndCollectExpandableUsers(
            insertDest, collapseOp.getReassociationIndices(), parallelInsertOp,
            expandableUsers))) {
      return failure();
    }

    // Expand the users of the destination.
    rewriter.setInsertionPointToStart(forallOp.getBody());
    expandVerifiedUsers(rewriter, loc, ctx, expandableUsers, totalInnerSizes,
                        reIndices, forallOp, parallelInsertOp);
    rewriter.setInsertionPoint(forallOp);

    // This pattern only supports forall ops with single
    // output.
    SmallVector<Value> forallOutputs(forallOp.getOutputs());
    // Create the expand -> new scf.forall -> collapse chain.
    auto expandedDestType =
        cast<RankedTensorType>(forallOutputs[tiedResultIdx].getType())
            .clone(expandedDestShape);
    auto expandedDest =
        tensor::ExpandShapeOp::create(rewriter, loc, expandedDestType,
                                      forallOutputs[tiedResultIdx], reIndices);

    forallOutputs[tiedResultIdx] = expandedDest;

    scf::ForallOp newForallOp = scf::ForallOp::create(
        rewriter, loc, forallOp.getMixedLowerBound(),
        forallOp.getMixedUpperBound(), forallOp.getMixedStep(), forallOutputs,
        forallOp.getMappingAttr());

    auto collapsedResultOp = tensor::CollapseShapeOp::create(
        rewriter, loc,
        cast<ShapedType>(forallOp->getResult(tiedResultIdx).getType()),
        newForallOp->getResult(tiedResultIdx), reIndices);

    // Merge the old scf.forall block which has the expanded users into the new
    // scf.forall which has the expanded destination.
    SmallVector<Value> argReplacements(newForallOp.getInductionVars());
    argReplacements.append(newForallOp.getRegionIterArgs().begin(),
                           newForallOp.getRegionIterArgs().end());
    scf::InParallelOp parallelTerminator = newForallOp.getTerminator();
    parallelTerminator->erase();
    rewriter.mergeBlocks(forallOp.getBody(), newForallOp.getBody(),
                         argReplacements);

    // Replaces the uses of the old scf.forall with the new scf.forall
    for (int idx = 0; idx < forallOp->getNumResults(); ++idx) {
      if (idx == tiedResultIdx) {
        forallOp->getResult(idx).replaceAllUsesWith(
            collapsedResultOp->getResult(0));
      } else {
        forallOp->getResult(idx).replaceAllUsesWith(
            newForallOp->getResult(idx));
      }
    }
    return success();
  }
};

/// This pattern collapses destination of scf.foralls by hoisting out
/// expand_shape op consumed by its parallel.insert_slice op.
/// Converse of ExpandDestinationForallOp.
///
/// Handles rank-reducing parallel_insert_slice: when the expand_shape output
/// rank is less than the dest rank (e.g. 4D source inserted into 7D dest),
/// the pattern builds a full dest reassociation by adding singleton groups
/// for the rank-reduced (dropped) dimensions.
struct CollapseDestinationForallOp final
    : OpRewritePattern<tensor::ParallelInsertSliceOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::ParallelInsertSliceOp parallelInsertOp,
                                PatternRewriter &rewriter) const override {
    Location loc = parallelInsertOp.getLoc();
    MLIRContext *ctx = getContext();
    auto expandOp =
        parallelInsertOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    // No expand op to hoist out.
    if (!expandOp) {
      return failure();
    }

    // Ignore trivially foldable expand ops.
    if (expandOp.getSrcType().getRank() ==
        expandOp.getResultType().getRank()) {
      return failure();
    }

    // Get the destination to collapse.
    Value insertDest = parallelInsertOp.getDest();

    // Get the enclosing scf.forall op.
    OpResult tiedResult = parallelInsertOp.getTiedOpResult();
    int64_t tiedResultIdx = tiedResult.getResultNumber();

    auto forallOp = dyn_cast<scf::ForallOp>(tiedResult.getOwner());
    if (!forallOp) {
      return failure();
    }

    // Build reassociation indices covering all dest dimensions.
    // For non-rank-reducing inserts, this is just the expand's reassociation.
    // For rank-reducing inserts, we add singleton groups for dropped dims.
    SmallVector<ReassociationIndices> expandReIndices =
        expandOp.getReassociationIndices();
    int64_t expandedRank = expandOp.getResultType().getRank();
    auto destType = cast<ShapedType>(insertDest.getType());
    int64_t destRank = destType.getRank();

    SmallVector<ReassociationIndices> fullReIndices;
    if (expandedRank == destRank) {
      // Non-rank-reducing: use expand's reassociation directly.
      fullReIndices = expandReIndices;
    } else if (expandedRank < destRank) {
      // Rank-reducing insert. Identify dropped dims by matching the expand
      // output shape against the insert sizes.
      ArrayRef<int64_t> expandedShape = expandOp.getResultType().getShape();
      SmallVector<OpFoldResult> insertMixedSizes =
          parallelInsertOp.getMixedSizes();
      llvm::SmallDenseSet<unsigned> droppedDims;
      unsigned srcIdx = 0;
      for (unsigned destIdx = 0;
           destIdx < static_cast<unsigned>(destRank); ++destIdx) {
        std::optional<int64_t> size = getConstantIntValue(insertMixedSizes[destIdx]);
        if (!size) {
          // Dynamic insert size: try to match to source dim.
          if (srcIdx < static_cast<unsigned>(expandedRank)) {
            srcIdx++;
          } else {
            return failure();
          }
          continue;
        }
        if (srcIdx < static_cast<unsigned>(expandedRank) &&
            *size == expandedShape[srcIdx]) {
          srcIdx++;
        } else {
          if (*size != 1)
            return failure();
          droppedDims.insert(destIdx);
        }
      }
      if (srcIdx != static_cast<unsigned>(expandedRank))
        return failure();

      // Map expanded dims to dest dims (skipping dropped dims).
      SmallVector<unsigned> expandedToDestDim;
      for (unsigned destIdx = 0;
           destIdx < static_cast<unsigned>(destRank); ++destIdx) {
        if (!droppedDims.count(destIdx))
          expandedToDestDim.push_back(destIdx);
      }

      // Build full reassociation: map expand groups to dest dims, add
      // singleton groups for dropped dims.
      for (const auto &group : expandReIndices) {
        ReassociationIndices destGroup;
        for (int64_t expandDim : group) {
          destGroup.push_back(expandedToDestDim[expandDim]);
        }
        fullReIndices.push_back(destGroup);
      }
      for (unsigned droppedDim : droppedDims) {
        fullReIndices.push_back({static_cast<int64_t>(droppedDim)});
      }
      // Sort groups by first element to maintain dim order.
      llvm::sort(fullReIndices,
                 [](const auto &a, const auto &b) {
                   return a.front() < b.front();
                 });

      // Verify each group has contiguous indices.
      for (const auto &group : fullReIndices) {
        for (size_t i = 1; i < group.size(); ++i) {
          if (group[i] != group[i - 1] + 1)
            return failure();
        }
      }
    } else {
      return failure();
    }

    SmallVector<int64_t> collapsedDestShape;
    SmallVector<int64_t> totalInnerSizes;
    // Get the collapsed shape which will be the new destination of the
    // scf.forall.
    if (failed(getCollapsedShape(fullReIndices, insertDest, collapsedDestShape,
                                 totalInnerSizes))) {
      return failure();
    }

    // Compute collapsed insert sizes from original insert sizes.
    SmallVector<int64_t> collapsedInsertSizes;
    ArrayRef<int64_t> originalInsertStaticSizes =
        parallelInsertOp.getStaticSizes();
    for (const auto &group : fullReIndices) {
      int64_t product = 1;
      for (int64_t dim : group) {
        int64_t dimSize = originalInsertStaticSizes[dim];
        if (ShapedType::isDynamic(dimSize))
          return failure();
        product *= dimSize;
      }
      collapsedInsertSizes.push_back(product);
    }

    // We only want this pattern if the forall op result is being written to a
    // full slice, or a collapsable buffer. Otherwise the hoisted expand op is
    // not foldable.
    for (Operation *foralluser : tiedResult.getUsers()) {
      auto storeOp =
          dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(foralluser);
      if (storeOp && isFullSlice(storeOp, storeOp.getTargetType(),
                                 storeOp.getTargetDims())) {
        continue;
      }
      auto storeToBufferOp =
          dyn_cast<IREE::Codegen::StoreToBufferOp>(foralluser);
      if (!storeToBufferOp) {
        return failure();
      }
    }

    // Verify that the users of destination are valid to collapse and collect
    // all such users.
    SmallVector<tensor::ExtractSliceOp> collapsableUsers;
    if (failed(verifyAndCollectCollapsableUsers(
            insertDest, fullReIndices, parallelInsertOp,
            collapsableUsers))) {
      return failure();
    }

    // Collapse the users of the destination.
    rewriter.setInsertionPointToStart(forallOp.getBody());
    collapseVerifiedUsers(rewriter, loc, ctx, collapsableUsers, fullReIndices,
                          destType.getShape(), collapsedInsertSizes, forallOp,
                          parallelInsertOp);
    rewriter.setInsertionPoint(forallOp);

    // Create the collapse -> new scf.forall -> expand chain.
    SmallVector<Value> forallOutputs(forallOp.getOutputs());
    auto collapsedDestType =
        cast<RankedTensorType>(forallOutputs[tiedResultIdx].getType())
            .clone(collapsedDestShape);
    auto collapsedDest =
        tensor::CollapseShapeOp::create(rewriter, loc, collapsedDestType,
                                        forallOutputs[tiedResultIdx],
                                        fullReIndices);

    forallOutputs[tiedResultIdx] = collapsedDest;

    scf::ForallOp newForallOp = scf::ForallOp::create(
        rewriter, loc, forallOp.getMixedLowerBound(),
        forallOp.getMixedUpperBound(), forallOp.getMixedStep(), forallOutputs,
        forallOp.getMappingAttr());

    auto expandedResultOp = tensor::ExpandShapeOp::create(
        rewriter, loc,
        cast<ShapedType>(forallOp->getResult(tiedResultIdx).getType()),
        newForallOp->getResult(tiedResultIdx), fullReIndices);

    // Merge the old scf.forall block which has the collapsed users into the
    // new scf.forall which has the collapsed destination.
    SmallVector<Value> argReplacements(newForallOp.getInductionVars());
    argReplacements.append(newForallOp.getRegionIterArgs().begin(),
                           newForallOp.getRegionIterArgs().end());
    scf::InParallelOp parallelTerminator = newForallOp.getTerminator();
    parallelTerminator->erase();
    rewriter.mergeBlocks(forallOp.getBody(), newForallOp.getBody(),
                         argReplacements);

    // Replace the uses of the old scf.forall with the new scf.forall.
    for (int idx = 0; idx < forallOp->getNumResults(); ++idx) {
      if (idx == tiedResultIdx) {
        forallOp->getResult(idx).replaceAllUsesWith(
            expandedResultOp->getResult(0));
      } else {
        forallOp->getResult(idx).replaceAllUsesWith(
            newForallOp->getResult(idx));
      }
    }
    return success();
  }
};

/// This pattern hoists expand_shape & collapse_shape ops out of scf.for loops.
struct ExpandDestinationForOp final : OpRewritePattern<scf::YieldOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(scf::YieldOp yieldOp,
                                PatternRewriter &rewriter) const override {
    Location loc = yieldOp.getLoc();
    auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
    if (!forOp) {
      return failure();
    }
    tensor::CollapseShapeOp collapseOp;
    tensor::ExpandShapeOp expandOp;
    int64_t tiedResultIdx = 0;

    for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
      collapseOp = operand.getDefiningOp<tensor::CollapseShapeOp>();
      if (!collapseOp) {
        continue;
      }
      if (collapseOp.getSrcType().getRank() ==
          collapseOp.getResultType().getRank()) {
        continue;
      }

      // Get the corresponding expandOp.
      auto iterArg = forOp.getRegionIterArgs()[idx];
      for (auto user : iterArg.getUsers()) {
        expandOp = dyn_cast<tensor::ExpandShapeOp>(user);
        if (expandOp &&
            (expandOp.getReassociationIndices() ==
             collapseOp.getReassociationIndices()) &&
            (expandOp.getResultType() == collapseOp.getSrcType())) {
          break;
        } else {
          expandOp = nullptr;
        }
      }

      if (expandOp && collapseOp) {
        bool hasOtherUsers = false;
        for (auto user : iterArg.getUsers()) {
          if (user != expandOp) {
            hasOtherUsers = true;
            expandOp = nullptr;
            collapseOp = nullptr;
            break;
          }
        }
        if (!hasOtherUsers) {
          tiedResultIdx = idx;
          break;
        }
      }
    }
    if (!expandOp || !collapseOp) {
      return failure();
    }

    // Create the expand -> new scf.for -> collapse chain.
    rewriter.setInsertionPoint(forOp);

    Value initArg = forOp.getInitArgs()[tiedResultIdx];
    auto expandedDest = tensor::ExpandShapeOp::create(
        rewriter, loc, expandOp.getResultType(), initArg,
        expandOp.getReassociationIndices());

    auto expandedInitArgs = llvm::to_vector_of<Value>(forOp.getInitArgs());
    expandedInitArgs[tiedResultIdx] = expandedDest.getResult();

    scf::ForOp newForOp = scf::ForOp::create(
        rewriter, loc, forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), expandedInitArgs);

    auto collapsedOutput = tensor::CollapseShapeOp::create(
        rewriter, loc, collapseOp.getResultType(),
        newForOp.getResults()[tiedResultIdx],
        collapseOp.getReassociationIndices());

    // Users of the result of collapseOp must use the input to the collapseOp.
    collapseOp->getResult(0).replaceAllUsesWith(collapseOp.getOperand());

    // Users of the result of expandOp must use the iter_arg of the new forOp.
    for (auto user : forOp.getRegionIterArgs()[tiedResultIdx].getUsers()) {
      if (user->getNumResults() > 0) {
        user->getResult(0).replaceAllUsesWith(
            newForOp.getRegionIterArgs()[tiedResultIdx]);
      }
    }

    // Merge the old scf.for block with the new scf.for block.
    SmallVector<Value> ivs = {newForOp.getInductionVar()};
    SmallVector<Value> argReplacements(ivs);
    argReplacements.append(newForOp.getRegionIterArgs().begin(),
                           newForOp.getRegionIterArgs().end());
    rewriter.mergeBlocks(forOp.getBody(), newForOp.getBody(), argReplacements);

    // Replace the uses of the old scf.for with the new scf.for.
    for (int idx = 0; idx < forOp->getNumResults(); ++idx) {
      if (idx == tiedResultIdx) {
        forOp->getResult(idx).replaceAllUsesWith(collapsedOutput->getResult(0));
      } else {
        forOp->getResult(idx).replaceAllUsesWith(newForOp->getResult(idx));
      }
    }
    return success();
  }
};

/// This pattern exchanges bitcast(extract_slice) to extract_slice(bitcast) in
/// an attempt to move the bitcast closer to the loads. There is a related
/// pattern that does the reverse when folding the bitcast is not possible and
/// should be applied later.
struct SwapInnerBitcastWithExtractSlice
    : OpRewritePattern<IREE::TensorExt::BitCastOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::TensorExt::BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {
    Value bitcastSrc = bitcastOp.getSource();
    auto sliceOp = bitcastSrc.getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp) {
      return rewriter.notifyMatchFailure(bitcastOp, "non-slice producer");
    }

    auto bitcastSrcType = cast<RankedTensorType>(bitcastSrc.getType());
    auto bitcastResType = cast<RankedTensorType>(bitcastOp.getType());

    // Verify that only the inner most dimension is changed by the bitcast by
    // comparing dynamic and static sizes for equality.
    if (bitcastOp.getSourceDims() != bitcastOp.getResultDims() ||
        bitcastSrcType.getShape().drop_back() !=
            bitcastResType.getShape().drop_back() ||
        ShapedType::isDynamic(bitcastSrcType.getShape().back())) {
      return rewriter.notifyMatchFailure(
          bitcastOp, "bitcast affects more than inner most dim");
    }

    // Fail if the inner most dim is sliced or if this is an encoded tensor.
    RankedTensorType sliceInputType = sliceOp.getSource().getType();
    if (sliceInputType.getEncoding() ||
        sliceInputType.getRank() != bitcastSrcType.getRank() ||
        sliceInputType.getShape().back() != bitcastSrcType.getShape().back()) {
      return rewriter.notifyMatchFailure(
          bitcastOp,
          "inner dimension is sliced or rank reducing or tensor is encoded");
    }

    int64_t newInnerSize = bitcastResType.getShape().back();
    SmallVector<int64_t> newBitcastShape(sliceInputType.getShape());
    newBitcastShape.back() = newInnerSize;

    auto newBitcastType =
        RankedTensorType::get(newBitcastShape, bitcastResType.getElementType());

    // Get the dynamic sizes of the slice source. Extracting a slice can remove
    // dynamic dimensions or introduce new ones, so a new list of sizes is
    // needed.
    SmallVector<OpFoldResult> newMixedSizes =
        tensor::getMixedSizes(rewriter, sliceOp.getLoc(), sliceOp.getSource());
    SmallVector<Value> sliceSourceDynamicSizes;
    SmallVector<int64_t> sliceSourceStaticSizes;
    dispatchIndexOpFoldResults(newMixedSizes, sliceSourceDynamicSizes,
                               sliceSourceStaticSizes);

    Value newBitcast = IREE::TensorExt::BitCastOp::create(
        rewriter, bitcastOp.getLoc(), newBitcastType, sliceOp.getSource(),
        sliceSourceDynamicSizes, sliceSourceDynamicSizes);
    SmallVector<int64_t> newSizes(sliceOp.getStaticSizes());
    newSizes.back() = newInnerSize;
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        bitcastOp, bitcastResType, newBitcast, sliceOp.getOffsets(),
        sliceOp.getSizes(), sliceOp.getStrides(), sliceOp.getStaticOffsets(),
        newSizes, sliceOp.getStaticStrides());

    return success();
  }
};

struct PropagateReshapesByExpansionPass final
    : impl::PropagateReshapesByExpansionPassBase<
          PropagateReshapesByExpansionPass> {
  void runOnOperation() override;
};
} // namespace

void populateCollapseDestinationForallPatterns(RewritePatternSet &patterns) {
  patterns.add<CollapseDestinationForallOp>(patterns.getContext());
}

void PropagateReshapesByExpansionPass::runOnOperation() {
  MLIRContext *context = &getContext();

  {
    RewritePatternSet patterns(context);
    // Preemptively attempt to fold any reshapes into interface bindings if
    // possible to simplify subsequent reshape propagation.
    populateReshapeToInterfaceTensorPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  RewritePatternSet bubbleExpandShapePatterns(context);
  linalg::ControlFusionFn bubbleUpExpansionControlFn =
      [](OpOperand *fusedOperand) {
        Operation *producer = fusedOperand->get().getDefiningOp();
        Operation *consumer = fusedOperand->getOwner();

        // Block only if one of the operations has a lowering configuration
        // which means it likely expects tiling specific to its original shape.
        if (getLoweringConfig(producer) || getLoweringConfig(consumer)) {
          return false;
        }
        return true;
      };
  linalg::populateFoldReshapeOpsByExpansionPatterns(bubbleExpandShapePatterns,
                                                    bubbleUpExpansionControlFn);
  IREE::Codegen::populateFoldReshapeOpsByExpansionPatterns(
      bubbleExpandShapePatterns, bubbleUpExpansionControlFn);
  // Add patterns to do some additional cleanup (on top of canonicalizations
  // that can be done later) of reshape ops.
  tensor::populateFoldTensorEmptyPatterns(bubbleExpandShapePatterns);
  linalg::FillOp::getCanonicalizationPatterns(bubbleExpandShapePatterns,
                                              context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(
      bubbleExpandShapePatterns, context);
  tensor::EmptyOp::getCanonicalizationPatterns(bubbleExpandShapePatterns,
                                               context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(bubbleExpandShapePatterns,
                                                     context);
  populateReshapeToInterfaceTensorPatterns(bubbleExpandShapePatterns);
  populateFoldTensorReshapeIntoBufferPatterns(bubbleExpandShapePatterns);
  bubbleExpandShapePatterns
      .add<ExpandDestinationForallOp, CollapseDestinationForallOp,
           ExpandDestinationForOp,
           SwapInnerBitcastWithExtractSlice>(context);

  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(bubbleExpandShapePatterns)))) {
    getOperation()->emitOpError("Failed to propagate reshapes");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
