// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_PROPAGATERESHAPESBYEXPANSIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

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
  populateExpandDestinationForallPatterns(bubbleExpandShapePatterns);
  bubbleExpandShapePatterns
      .add<ExpandDestinationForOp, SwapInnerBitcastWithExtractSlice>(context);

  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(bubbleExpandShapePatterns)))) {
    getOperation()->emitOpError("Failed to propagate reshapes");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
