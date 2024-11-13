// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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
  if (!destType)
    return failure();
  // TODO (nirvedhmeshram): Support rank reducing parallel_insert_slice.
  if (reIndices.size() != destType.getShape().size())
    return failure();
  // Iterator to insert outer sizes.
  auto outerShapeIter = expandedShape.begin();
  for (auto [reassociations, destSize] :
       llvm::zip_equal(reIndices, destType.getShape())) {
    // Dynamic destination dims that are not getting expanded are allowed.
    if (ShapedType::isDynamic(destSize) && reassociations.size() == 1) {
      expandedShape.insert(outerShapeIter++, destSize);
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
      if (ShapedType::isDynamic(expandedInnerSize))
        return failure();
      expandedShape.push_back(expandedInnerSize);
      totalInnerSize *= expandedInnerSize;
    }
    if (destSize % totalInnerSize != 0)
      return failure();
    totalInnerSizes.push_back(totalInnerSize);
    // insert the outer size in front of any inner sizes.
    expandedShape.insert(outerShapeIter, destSize / totalInnerSize);
    // set up the iterator for the next uncollapsed dimension.
    outerShapeIter = expandedShape.end();
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
    if (!extractSliceOp)
      return failure();
    if (extractSliceOp.getMixedSizes() != parallelInsertOp.getMixedSizes())
      return failure();
    if (extractSliceOp.getMixedOffsets() != parallelInsertOp.getMixedOffsets())
      return failure();
    auto expandShapeOp =
        dyn_cast<tensor::ExpandShapeOp>(*extractSliceOp->getUsers().begin());
    if (!expandShapeOp)
      return failure();
    SmallVector<ReassociationIndices> expandReIndices =
        expandShapeOp.getReassociationIndices();
    if (reIndices != expandReIndices)
      return failure();
    expandableUsers.push_back(extractSliceOp);
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
    auto expandedOffsetsIter = expandedOffsets.begin();

    for (auto [index, offset] : llvm::enumerate(mixedOffsets)) {
      // Add zero offsets for the extra dimensions from reIndices.
      for (size_t i = 1, e = reIndices[index].size(); i < e; ++i) {
        expandedOffsets.push_back(getAsIndexOpFoldResult(ctx, 0));
      }
      rewriter.setInsertionPointToStart(forallOp.getBody());
      // Compute the outer dimension expression.
      AffineExpr s0, s1;
      bindSymbols(rewriter.getContext(), s0, s1);
      AffineExpr outerDimExpr = (s0).floorDiv(s1);
      // Insert computed offset using affine expression.
      expandedOffsets.insert(
          expandedOffsetsIter,
          affine::makeComposedFoldedAffineApply(
              rewriter, loc, outerDimExpr,
              {getValueOrCreateConstantIndexOp(rewriter, loc, offset),
               rewriter.getIndexAttr(totalInnerSizes[index])}));

      expandedOffsetsIter = expandedOffsets.end();
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
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        extractSliceOp, resultType, extractSliceOp.getSource(), expandedOffsets,
        expandedSizes, expandedStrides);
  }
  return;
}

/// This pattern expands destination of workgroup mapped scf.foralls by
/// hoisting out collapse_shape op consumed by its parallel.insert_slice op.
struct ExpandDestinationForallOp final
    : OpRewritePattern<tensor::ParallelInsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ParallelInsertSliceOp parallelInsertOp,
                                PatternRewriter &rewriter) const override {
    Location loc = parallelInsertOp.getLoc();
    MLIRContext *ctx = getContext();
    auto collapseOp =
        parallelInsertOp.getSource().getDefiningOp<tensor::CollapseShapeOp>();
    // No collapse op to hoist out.
    if (!collapseOp)
      return failure();

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
    if (!forallOp)
      return failure();

    // We only want this pattern if the forall op result is being written to a
    // full slice. Otherwise the hoisted collapse op is not foldable.
    for (Operation *foralluser : tiedResult.getUsers()) {
      auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(foralluser);
      if (!storeOp)
        return failure();
      if (!isFullSlice(storeOp, storeOp.getTargetType(),
                       storeOp.getTargetDims())) {
        return failure();
      }
    }

    // This allows us to assume that the extract/inserts in the loop are
    // disjoint and makes the application of this pattern safe.
    if (!forallOpHasMappingType<IREE::Codegen::WorkgroupMappingAttr>(
            forallOp)) {
      return failure();
    }
    // This pattern only supports forall ops with single
    // output.
    SmallVector<Value> forallOutputs(forallOp.getOutputs());

    SmallVector<ReassociationIndices> reIndices =
        collapseOp.getReassociationIndices();
    SmallVector<int64_t> expandedDestShape;
    SmallVector<int64_t> totalInnerSizes;
    // Get the shape of the outer expand which will be the new destination
    // of the scf.forall and the total size of inner dimensions per uncollapsed
    // dimension.
    if (failed(getExpandedShape(reIndices, collapseOp.getSrcType().getShape(),
                                insertDest, expandedDestShape,
                                totalInnerSizes))) {
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

    // Create the expand -> new scf.forall -> collapse chain.
    auto expandedDestType =
        cast<RankedTensorType>(forallOutputs[tiedResultIdx].getType())
            .clone(expandedDestShape);
    auto expandedDest = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandedDestType, forallOutputs[tiedResultIdx], reIndices);

    forallOutputs[tiedResultIdx] = expandedDest;

    scf::ForallOp newForallOp = rewriter.create<scf::ForallOp>(
        loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
        forallOp.getMixedStep(), forallOutputs, forallOp.getMappingAttr());

    auto collapsedResultOp = rewriter.create<tensor::CollapseShapeOp>(
        loc, cast<ShapedType>(forallOp->getResult(tiedResultIdx).getType()),
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

struct PropagateReshapesByExpansionPass final
    : impl::PropagateReshapesByExpansionPassBase<
          PropagateReshapesByExpansionPass> {
  void runOnOperation() override;
};
} // namespace

void PropagateReshapesByExpansionPass::runOnOperation() {
  MLIRContext *context = &getContext();

  {
    RewritePatternSet patterns(context);
    // Preemptively attempt to fold any reshapes into interface bindings if
    // possible to simplify subsequent reshape propagation.
    populateReshapeToInterfaceTensorPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
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
  bubbleExpandShapePatterns.add<ExpandDestinationForallOp>(context);

  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), std::move(bubbleExpandShapePatterns)))) {
    getOperation()->emitOpError("Failed to propagate reshapes");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
