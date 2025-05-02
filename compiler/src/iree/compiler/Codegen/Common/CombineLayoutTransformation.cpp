// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-combine-layout-transformation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_COMBINELAYOUTTRANSFORMATIONPASS
#define GEN_PASS_DEF_FOLDLAYOUTTRANSFORMATIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Preprocessing Utilities
//===----------------------------------------------------------------------===//

/// Convert complex ops into simpler ops by decomposing or raising to a named
/// op.
///  - `PackOp`s and `UnPackOp`s are decomposed.
///  - Transpose `linalg::GenericOp`s are raised to `linalg::TransposeOp`s.
static void simplifyComplexRelayoutOps(RewriterBase &rewriter,
                                       FunctionOpInterface funcOp) {
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<linalg::PackOp> packOps(
      funcOp.getFunctionBody().getOps<linalg::PackOp>());
  for (auto packOp : packOps) {
    rewriter.setInsertionPoint(packOp);
    (void)linalg::lowerPack(rewriter, packOp,
                            /*lowerPadLikeWithInsertSlice=*/false);
  }
  SmallVector<linalg::UnPackOp> unPackOps(
      funcOp.getFunctionBody().getOps<linalg::UnPackOp>());
  for (auto unPackOp : unPackOps) {
    rewriter.setInsertionPoint(unPackOp);
    (void)linalg::lowerUnPack(rewriter, unPackOp,
                              /*lowerUnpadLikeWithExtractSlice=*/false);
  }
  SmallVector<linalg::GenericOp> genericOps(
      funcOp.getFunctionBody().getOps<linalg::GenericOp>());
  for (auto genericOp : genericOps) {
    if (linalg::isaTransposeOpInterface(genericOp)) {
      rewriter.setInsertionPoint(genericOp);
      (void)linalg::specializeGenericOp(rewriter, genericOp);
    }
  }
}

//===----------------------------------------------------------------------===//
// Combining Layout Transformation Ops
//===----------------------------------------------------------------------===//

/// Folds an `op` that does not affect index computation into a `mapScatterOp`.
/// This is used for ops like `linalg::CopyOp`.
static IREE::LinalgExt::MapScatterOp
foldIdentityLikeOpIntoMapScatter(RewriterBase &rewriter, Operation *op,
                                 IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == op->getResult(0) &&
         "expected op to be the producer of mapScatterOp");
  rewriter.modifyOpInPlace(
      mapScatterOp, [&]() { mapScatterOp->setOperand(0, op->getOperand(0)); });
  return mapScatterOp;
}

/// Fold a `transposeOp` into a consumer `mapScatterOp`, by transposing the
/// uses of the `mapScatterOp`s transformation_region block arguments.
static IREE::LinalgExt::MapScatterOp
foldTransposeIntoMapScatter(RewriterBase &rewriter,
                            linalg::TransposeOp transposeOp,
                            IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == transposeOp->getResult(0) &&
         "expected transposeOp to be the producer of mapScatterOp");

  ArrayRef<int64_t> perm = transposeOp.getPermutation();
  auto indexTransformBuilder =
      [&](ArrayRef<BlockArgument> srcIndices) -> SmallVector<Value> {
    SmallVector<Value> indexValues(srcIndices);
    return applyPermutation(indexValues, perm);
  };
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.insertTransformationAtStart(rewriter, indexTransformBuilder,
                                             perm.size());
    mapScatterOp->setOperand(0, transposeOp.getInput());
  });
  return mapScatterOp;
}

/// Fold a tensor::ExpandShapeOp or tensor::CollapseShapeOp into a consumer
/// `mapScatterOp`, by linearizing and then delinearizing the source indices
/// of the `mapScatterOp`s index transformation.
static IREE::LinalgExt::MapScatterOp
foldReshapeIntoMapScatter(RewriterBase &rewriter, Operation *reshapeOp,
                          SmallVector<OpFoldResult> srcDims,
                          SmallVector<OpFoldResult> resultDims,
                          IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == reshapeOp->getResult(0) &&
         "expected reshapeOp to be the producer of mapScatterOp");

  auto indexTransformBuilder =
      [&](ArrayRef<BlockArgument> srcIndices) -> SmallVector<Value> {
    auto linearizeIndexOp = rewriter.create<affine::AffineLinearizeIndexOp>(
        mapScatterOp->getLoc(), srcIndices, srcDims);
    auto delinearizeIndexOp = rewriter.create<affine::AffineDelinearizeIndexOp>(
        mapScatterOp->getLoc(), linearizeIndexOp.getResult(), resultDims);
    return delinearizeIndexOp->getResults();
  };
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.insertTransformationAtStart(rewriter, indexTransformBuilder,
                                             srcDims.size());
    mapScatterOp->setOperand(0, reshapeOp->getOperand(0));
  });
  return mapScatterOp;
}

static IREE::LinalgExt::MapScatterOp
foldCollapseShapeIntoMapScatter(RewriterBase &rewriter,
                                tensor::CollapseShapeOp collapseShapeOp,
                                IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == collapseShapeOp->getResult(0) &&
         "expected collapseShapeOp to be the producer of mapScatterOp");
  Location loc = collapseShapeOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(collapseShapeOp);
  SmallVector<OpFoldResult> mixedSrcSizes =
      tensor::getMixedSizes(rewriter, loc, collapseShapeOp.getSrc());
  // There can be leftover tensor.dim ops consuming the result of the collpase,
  // but they will be folded into some affine.apply ops on the source sizes by
  // later cleanup patterns.
  SmallVector<OpFoldResult> mixedResultSizes =
      tensor::getMixedSizes(rewriter, loc, collapseShapeOp.getResult());
  return foldReshapeIntoMapScatter(rewriter, collapseShapeOp, mixedSrcSizes,
                                   mixedResultSizes, mapScatterOp);
}

static IREE::LinalgExt::MapScatterOp
foldExpandShapeIntoMapScatter(RewriterBase &rewriter,
                              tensor::ExpandShapeOp expandShapeOp,
                              IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == expandShapeOp->getResult(0) &&
         "expected expandShapeOp to be the producer of mapScatterOp");
  Location loc = expandShapeOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(expandShapeOp);
  SmallVector<OpFoldResult> srcMixedSizes =
      tensor::getMixedSizes(rewriter, loc, expandShapeOp.getSrc());
  return foldReshapeIntoMapScatter(rewriter, expandShapeOp, srcMixedSizes,
                                   expandShapeOp.getMixedOutputShape(),
                                   mapScatterOp);
}

/// Fold an `extractSliceOp` into a consumer `mapScatterOp` by applying a mask
/// based on the bounds of the extractSliceOp. Currently, only zero offsets and
/// unit strides are supported.
///
/// For a given `%mask` value, and slice sizes of `%size0`, `%size1`, ...,
/// `%sizeN`, the `%new_mask` becomes:
/// ```
/// %bound0 = arith.cmpi uge, %idx0, %size0 : index
/// %mask0 = arith.andi %mask, %bound0 : i1
/// %bound1 = arith.cmpi uge, %idx1, %size1 : index
/// %mask1 = arith.andi %mask0, %bound1 : i1
/// ...
/// %boundN = arith.cmpi uge, %idxN, %sizeN : index
/// %new_mask = arith.andi %maskN-1, %boundN : i1
/// ```
static FailureOr<IREE::LinalgExt::MapScatterOp>
foldExtractSliceIntoMapScatter(RewriterBase &rewriter,
                               tensor::ExtractSliceOp extractSliceOp,
                               IREE::LinalgExt::MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == extractSliceOp->getResult(0) &&
         "expected extractSliceOp to be the producer of mapScatterOp");
  // TODO(Max191): Support rank-reducing slices.
  if (extractSliceOp.getSourceType().getRank() !=
      extractSliceOp.getResultType().getRank()) {
    return rewriter.notifyMatchFailure(
        extractSliceOp, "rank reducing extract_slice op is not supported");
  }
  // TODO(Max191): Support non-zero offsets.
  if (!areAllConstantIntValue(extractSliceOp.getMixedOffsets(), 0)) {
    return rewriter.notifyMatchFailure(extractSliceOp,
                                       "non-zero offsets are not supported");
  }
  // TODO(Max191): Support non-unit strides.
  if (!areAllConstantIntValue(extractSliceOp.getMixedStrides(), 1)) {
    return rewriter.notifyMatchFailure(extractSliceOp,
                                       "non-unit strides are not supported");
  }
  SmallVector<OpFoldResult> bounds(extractSliceOp.getMixedSizes());
  Block &transformBody = mapScatterOp.getTransformationRegion().front();
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
  Value mask = yieldOp->getOperands().back();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(yieldOp);
  Location loc = mapScatterOp->getLoc();
  ArrayRef<BlockArgument> srcIndices = transformBody.getArguments();
  for (auto [bound, srcIdx] : llvm::zip_equal(bounds, srcIndices)) {
    Value boundValue = getValueOrCreateConstantIndexOp(rewriter, loc, bound);
    auto isOutOfBounds =
        rewriter
            .create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, srcIdx,
                                   boundValue)
            ->getResult(0);
    mask = rewriter.create<arith::AndIOp>(loc, mask, isOutOfBounds);
  }
  rewriter.modifyOpInPlace(yieldOp, [&]() {
    yieldOp->setOperand(yieldOp->getNumOperands() - 1, mask);
  });
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp->setOperand(0, extractSliceOp.getSource());
  });
  return mapScatterOp;
}

/// Fold the `op` into the `mapScatterOp`, if possible. The resulting
/// map_scatter op is returned, if the `op` was folded. Otherwise, return
/// failure.
static FailureOr<IREE::LinalgExt::MapScatterOp>
foldIntoMapScatter(RewriterBase &rewriter, Operation *op,
                   IREE::LinalgExt::MapScatterOp mapScatterOp) {
  if (isa<linalg::CopyOp>(op)) {
    return foldIdentityLikeOpIntoMapScatter(rewriter, op, mapScatterOp);
  }
  if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
    return foldTransposeIntoMapScatter(rewriter, transposeOp, mapScatterOp);
  }
  if (auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
    return foldExpandShapeIntoMapScatter(rewriter, expandShapeOp, mapScatterOp);
  }
  if (auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
    return foldCollapseShapeIntoMapScatter(rewriter, collapseShapeOp,
                                           mapScatterOp);
  }
  if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    return foldExtractSliceIntoMapScatter(rewriter, extractSliceOp,
                                          mapScatterOp);
  }
  return failure();
}

/// Return true if the mapScatterOp is an identity transformation, meaning it
/// is just doing a copy from input to output. This is true if the mask always
/// evaluates to true, and the mapping from input to output index is identity.
static bool isIdentityMapScatterOp(IREE::LinalgExt::MapScatterOp mapScatterOp) {
  if (mapScatterOp.getInputType() != mapScatterOp.getOutputType()) {
    return false;
  }

  // Check that the mask is always true.
  Block &transformBody =
      mapScatterOp.getTransformationRegion().getBlocks().front();
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
  Value mask = yieldOp->getOperands().back();
  std::optional<int64_t> constMask = getConstantIntValue(mask);
  if (!constMask.has_value() || constMask.value() == 0) {
    return false;
  }

  // Check that the block arguments are directly yielded in the order that they
  // are defined in the block.
  for (int i = 0; i < mapScatterOp.getOutputRank(); ++i) {
    auto yieldedBbArg = dyn_cast<BlockArgument>(yieldOp.getOperand(i));
    if (!yieldedBbArg || yieldedBbArg.getArgNumber() != i) {
      return false;
    }
  }
  return true;
}

/// Starting from the `root`, iteratively combine any relayout op producers
/// into a single iree_linalg_ext.map_scatter op. An identity map_scatter op
/// is inserted before the root, and then the producers of the map_scatter op
/// are folded into the map_scatter until an unsupported op is reached.
static void combineRelayoutOpChain(RewriterBase &rewriter,
                                   IREE::LinalgExt::MapScatterOp mapScatterOp) {
  Operation *relayoutOp = mapScatterOp.getInput().getDefiningOp();
  if (!relayoutOp) {
    return;
  }
  IREE::LinalgExt::MapScatterOp combinedRelayoutOp = mapScatterOp;
  while (relayoutOp) {
    LDBG("Attempting to fold " << relayoutOp->getName()
                               << " into map_scatter op:\n"
                               << *relayoutOp << "\n");
    FailureOr<IREE::LinalgExt::MapScatterOp> maybeCombinedRelayoutOp =
        foldIntoMapScatter(rewriter, relayoutOp, combinedRelayoutOp);
    if (failed(maybeCombinedRelayoutOp)) {
      LDBG("Failed to fold " << relayoutOp->getName()
                             << " into map_scatter op");
      break;
    }
    combinedRelayoutOp = maybeCombinedRelayoutOp.value();
    LDBG("Successfully folded " << relayoutOp->getName()
                                << " into map_scatter. New map_scatter op:\n"
                                << combinedRelayoutOp << "\n");
    relayoutOp = combinedRelayoutOp.getInput().getDefiningOp();
  }
  /*if (combinedRelayoutOp.isIdentity()) {
    rewriter.replaceOp(combinedRelayoutOp, combinedRelayoutOp.getInput());
  }*/
}

static IREE::LinalgExt::MapScatterOp
insertIdentityMapScatter(RewriterBase &rewriter,
                         IREE::Codegen::StoreToMemrefOp storeOp) {
  Location loc = storeOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(storeOp);
  auto mapScatterDest =
      rewriter
          .create<tensor::EmptyOp>(
              loc, memref::getMixedSizes(rewriter, loc, storeOp.getTarget()),
              storeOp.getValue().getType().getElementType())
          .getResult();
  auto mapScatterOp = rewriter.create<IREE::LinalgExt::MapScatterOp>(
      loc, storeOp.getValue(), mapScatterDest);
  rewriter.modifyOpInPlace(
      storeOp, [&]() { storeOp->setOperand(0, mapScatterOp.getResult(0)); });
  LDBG("Created identity map_scatter:\n" << mapScatterOp << "\n");
  return mapScatterOp;
}

namespace {

struct CombineLayoutTransformationPass final
    : impl::CombineLayoutTransformationPassBase<
          CombineLayoutTransformationPass> {
  using impl::CombineLayoutTransformationPassBase<
      CombineLayoutTransformationPass>::CombineLayoutTransformationPassBase;

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Apply some preprocessing to convert complex layout transformation
    // ops like pack and unpack into simpler supported ops.
    IRRewriter rewriter(&getContext());
    simplifyComplexRelayoutOps(rewriter, funcOp);

    // Start from iree_codegen.store_to_memref ops, and combine producer
    // relayout ops into a single map_scatter.
    SmallVector<IREE::Codegen::StoreToMemrefOp> dispatchResults(
        funcOp.getFunctionBody().getOps<IREE::Codegen::StoreToMemrefOp>());
    for (IREE::Codegen::StoreToMemrefOp dispatchResult : dispatchResults) {
      IREE::LinalgExt::MapScatterOp mapScatterOp =
          insertIdentityMapScatter(rewriter, dispatchResult);
      combineRelayoutOpChain(rewriter, mapScatterOp);
    }

    // Cleanup any tensor.dim ops that may be present after relayout
    // combination.
    RewritePatternSet cleanupPatterns(&getContext());
    memref::populateResolveRankedShapedTypeResultDimsPatterns(cleanupPatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(cleanupPatterns)))) {
      return signalPassFailure();
    }
  }
};

struct FoldLayoutTransformationPass final
    : impl::FoldLayoutTransformationPassBase<FoldLayoutTransformationPass> {
  using impl::FoldLayoutTransformationPassBase<
      FoldLayoutTransformationPass>::FoldLayoutTransformationPassBase;

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Apply some preprocessing to convert complex layout transformation
    // ops like pack and unpack into simpler supported ops.
    IRRewriter rewriter(&getContext());
    simplifyComplexRelayoutOps(rewriter, funcOp);

    // Start from iree_codegen.store_to_memref ops, and combine producer
    // relayout ops into a single map_scatter.
    SmallVector<IREE::LinalgExt::MapScatterOp> mapScatterOps;
    funcOp.walk([&](IREE::LinalgExt::MapScatterOp mapScatterOp) {
      mapScatterOps.push_back(mapScatterOp);
    });
    for (IREE::LinalgExt::MapScatterOp mapScatterOp : mapScatterOps) {
      combineRelayoutOpChain(rewriter, mapScatterOp);
    }

    // Cleanup any tensor.dim ops that may be present after relayout
    // combination.
    RewritePatternSet cleanupPatterns(&getContext());
    memref::populateResolveRankedShapedTypeResultDimsPatterns(cleanupPatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(cleanupPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
