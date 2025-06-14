// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CombineLayoutTransformation.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-combine-layout-transformation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_COMBINELAYOUTTRANSFORMATIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using IREE::LinalgExt::MapScatterOp;

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
    FailureOr<linalg::LowerPackResult> result = linalg::lowerPack(
        rewriter, packOp, /*lowerPadLikeWithInsertSlice=*/false);
    // For aligned pack ops, the pad will be a no-op, and can be folded away.
    // Fold it here so it does not complicate the index transformation folding
    // later on.
    if (failed(result) || !result->padOp) {
      continue;
    }
    if (areAllConstantIntValue(result->padOp.getMixedLowPad(), 0) &&
        areAllConstantIntValue(result->padOp.getMixedHighPad(), 0)) {
      rewriter.replaceOp(result->padOp, result->padOp.getSource());
    }
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
static MapScatterOp
foldIdentityLikeOpIntoMapScatter(RewriterBase &rewriter, Operation *op,
                                 MapScatterOp mapScatterOp) {
  assert(mapScatterOp.getInput() == op->getResult(0) &&
         "expected op to be the producer of mapScatterOp");
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.getInputMutable().assign(op->getOperand(0));
  });
  return mapScatterOp;
}

/// Fold a `transposeOp` into a consumer `mapScatterOp`, by transposing the
/// uses of the `mapScatterOp`s transformation_region block arguments.
static MapScatterOp foldTransposeIntoMapScatter(RewriterBase &rewriter,
                                                linalg::TransposeOp transposeOp,
                                                MapScatterOp mapScatterOp) {
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
    mapScatterOp.getInputMutable().assign(transposeOp.getInput());
  });
  return mapScatterOp;
}

/// Fold an `extractSliceOp` into a consumer `mapScatterOp` by applying a mask
/// based on the bounds of the extractSliceOp. Currently, only zero offsets and
/// unit strides are supported.
///
/// For a given `%mask` value, and slice sizes of `%size0`, `%size1`, ...,
/// `%sizeN`, the `%new_mask` becomes:
///
///   %bound0 = arith.cmpi ult, %idx0, %size0 : index
///   %mask0 = arith.andi %mask, %bound0 : i1
///   %bound1 = arith.cmpi ult, %idx1, %size1 : index
///   %mask1 = arith.andi %mask0, %bound1 : i1
///   ...
///   %boundN = arith.cmpi ult, %idxN, %sizeN : index
///   %new_mask = arith.andi %maskN-1, %boundN : i1
static FailureOr<MapScatterOp>
foldExtractSliceIntoMapScatter(RewriterBase &rewriter,
                               tensor::ExtractSliceOp extractSliceOp,
                               MapScatterOp mapScatterOp) {
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
            .create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, srcIdx,
                                   boundValue)
            ->getResult(0);
    mask = rewriter.create<arith::AndIOp>(loc, mask, isOutOfBounds);
  }
  rewriter.modifyOpInPlace(yieldOp, [&]() {
    yieldOp->setOperand(yieldOp->getNumOperands() - 1, mask);
  });
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.getInputMutable().assign(extractSliceOp.getSource());
  });
  return mapScatterOp;
}

static void buildNestedDistributionLoops(
    RewriterBase &rewriter, Location loc, int64_t distributionLevel,
    SmallVector<OpFoldResult> lbs, SmallVector<OpFoldResult> ubs,
    ArrayRef<DistributionConfig> distConfigs,
    function_ref<void(OpBuilder &, Location, ValueRange)> innerLoopBuilder) {
  DistributionConfig distConfig = distConfigs[distributionLevel];
  SmallVector<OpFoldResult> steps =
      getAsIndexOpFoldResult(rewriter.getContext(), distConfig.tileSizes);
  rewriter.create<scf::ForallOp>(
      loc, lbs, ubs, steps, /*outputs=*/ValueRange(),
      rewriter.getArrayAttr(distConfig.mapping),
      /*bodyBuilder=*/[&](OpBuilder &b, Location nestedLoc, ValueRange ivs) {
        SmallVector<OpFoldResult> nestedLbs(ivs);
        SmallVector<OpFoldResult> nestedUbs;
        for (auto [start, step, ub] : llvm::zip_equal(nestedLbs, steps, ubs)) {
          auto tileEnd = IREE::LinalgExt::addOfrs(b, nestedLoc, start, step);
          auto minMap = AffineMap::get(
              2, 0, {b.getAffineDimExpr(0), b.getAffineDimExpr(1)},
              rewriter.getContext());
          auto min = affine::makeComposedFoldedAffineMin(b, nestedLoc, minMap,
                                                         {tileEnd, ub});
          nestedUbs.push_back(min);
        }
        // Continue distribution if there are more distribution levels.
        if (distributionLevel + 1 < distConfigs.size()) {
          buildNestedDistributionLoops(
              rewriter, nestedLoc, distributionLevel + 1, nestedLbs, nestedUbs,
              distConfigs, innerLoopBuilder);
          b.create<scf::InParallelOp>(nestedLoc);
          return;
        }
        // Otherwise, tile to one, and generate the inner loop body.
        SmallVector<Value> nestedLbVals =
            getValueOrCreateConstantIndexOp(b, nestedLoc, nestedLbs);
        SmallVector<Value> nestedUbVals =
            getValueOrCreateConstantIndexOp(b, nestedLoc, nestedUbs);
        Value one = rewriter.create<arith::ConstantIndexOp>(nestedLoc, 1);
        SmallVector<Value> unitSteps(nestedLbs.size(), one);
        scf::buildLoopNest(rewriter, nestedLoc, nestedLbVals, nestedUbVals,
                           unitSteps, innerLoopBuilder);
        b.create<scf::InParallelOp>(nestedLoc);
      });
}

/// Fold a tensor.pad op into a iree_linalg_ext.map_scatter op, and separate
/// the writing of padding values into a separate operation on the buffer that
/// the map_scatter op is ultimately written into. The result buffer is taken
/// from the direct consumer of the `mapScatterOp`, which is expected to be an
/// `iree_codegen.store_to_buffer` op. Return failure if the result buffer is
/// not found.
static FailureOr<MapScatterOp>
foldPadIntoMapScatter(RewriterBase &rewriter, tensor::PadOp padOp,
                      MapScatterOp mapScatterOp,
                      PadDistributionConfigFn padDistributionConfigFn) {
  // Find the output buffer that the mapScatterOp is stored into.
  if (!mapScatterOp->hasOneUse()) {
    return rewriter.notifyMatchFailure(
        mapScatterOp, "map_scatter does not have a single user");
  }
  auto storeOp = dyn_cast<IREE::Codegen::StoreToBufferOp>(
      *mapScatterOp->getUsers().begin());
  if (!storeOp) {
    return rewriter.notifyMatchFailure(
        mapScatterOp,
        "map_scatter user is not an iree_codegen.store_to_buffer op");
  }

  rewriter.setInsertionPointAfter(storeOp);
  Location loc = padOp->getLoc();
  SmallVector<OpFoldResult> padSrcSizes =
      tensor::getMixedSizes(rewriter, loc, padOp.getSource());
  SmallVector<OpFoldResult> ubs =
      tensor::getMixedSizes(rewriter, loc, padOp.getResult());
  SmallVector<OpFoldResult> lbs(ubs.size(), rewriter.getIndexAttr(0));
  SmallVector<DistributionConfig> distConfigs = padDistributionConfigFn(
      padOp.getSourceType().getShape(), rewriter.getContext());

  // Write the padding values directly into the outputBuffer.
  Value outputBuffer = storeOp.getBuffer();
  auto innerLoopBuilder = [&](OpBuilder &b, Location loopLoc, ValueRange ivs) {
    // We need to scatter the padding values according to the existing
    // mapScatterOp transformation, so clone the transformation into the
    // loop nest.
    IRMapping mapping;
    Region &transformRegion = mapScatterOp.getTransformationRegion();
    Block *loopBody = rewriter.getInsertionBlock();
    transformRegion.cloneInto(loopBody->getParent(), mapping);
    Block *clonedTransformBody =
        mapping.lookup(&transformRegion.getBlocks().front());
    // Get a pointer to the YieldOp before inlining the Block.
    auto yieldOp =
        cast<IREE::LinalgExt::YieldOp>(clonedTransformBody->getTerminator());
    rewriter.inlineBlockBefore(clonedTransformBody, loopBody, loopBody->begin(),
                               ivs);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(yieldOp);
    // Compute the indices into the outputBuffer, and the if condition to
    // write the padding values. Padding values must obey the existing mask
    // of the current mapScatterOp, and also be in the low or high pad
    // range of the padOp.
    SmallVector<OpFoldResult> mixedLows = padOp.getMixedLowPad();
    Value writeCond;
    for (auto [low, srcSize, idx] :
         llvm::zip_equal(mixedLows, padSrcSizes, ivs)) {
      Value lowVal = getValueOrCreateConstantIndexOp(rewriter, loc, low);
      Value isLowPad = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, idx, lowVal);
      Value highPadStart = getValueOrCreateConstantIndexOp(
          rewriter, loc, IREE::LinalgExt::addOfrs(b, loopLoc, low, srcSize));
      Value isHighPad = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::uge, idx, highPadStart);
      Value isPad = rewriter.create<arith::OrIOp>(loc, isLowPad, isHighPad);
      writeCond = !writeCond
                      ? isPad
                      : rewriter.create<arith::OrIOp>(loc, writeCond, isPad);
    }
    SmallVector<Value> storeIndices(yieldOp.getOperands());
    rewriter.eraseOp(yieldOp);
    Value mask = storeIndices.pop_back_val();
    writeCond = rewriter.create<arith::AndIOp>(loc, writeCond, mask);
    // Create the store to the outputBuffer.
    auto thenBuilder = [&](OpBuilder &nestedBuilder, Location ifLoc) {
      nestedBuilder.create<memref::StoreOp>(
          ifLoc, padOp.getConstantPaddingValue(), outputBuffer, storeIndices);
      nestedBuilder.create<scf::YieldOp>(ifLoc);
    };
    b.create<scf::IfOp>(loopLoc, writeCond, thenBuilder);
  };

  buildNestedDistributionLoops(rewriter, loc, /*distributionLevel=*/0, lbs, ubs,
                               distConfigs, innerLoopBuilder);

  // Now that the padding values are being written to the outputBuffer, the
  // padOp becomes a no-op with respect to the index transformation on the
  // non-padded values.
  rewriter.modifyOpInPlace(mapScatterOp, [&]() {
    mapScatterOp.getInputMutable().assign(padOp.getSource());
  });
  return mapScatterOp;
}

/// Fold the `op` into the `mapScatterOp`, if possible. The resulting
/// map_scatter op is returned, if the `op` was folded. Otherwise, return
/// failure. For `PadOp`s, use the `padDistributionConfigFn` to distribute
/// the writing of padding values to the corresponding output buffer.
static FailureOr<MapScatterOp>
foldIntoMapScatter(RewriterBase &rewriter, Operation *op,
                   MapScatterOp mapScatterOp,
                   PadDistributionConfigFn padDistributionConfigFn) {
  return llvm::TypeSwitch<Operation *, FailureOr<MapScatterOp>>(op)
      .Case<linalg::CopyOp>([&](linalg::CopyOp copyOp) {
        return foldIdentityLikeOpIntoMapScatter(rewriter, copyOp, mapScatterOp);
      })
      .Case<linalg::TransposeOp>([&](linalg::TransposeOp transposeOp) {
        return foldTransposeIntoMapScatter(rewriter, transposeOp, mapScatterOp);
      })
      .Case<tensor::ExpandShapeOp>([&](tensor::ExpandShapeOp expandOp) {
        return foldExpandShapeIntoMapScatter(rewriter, expandOp, mapScatterOp);
      })
      .Case<tensor::CollapseShapeOp>([&](tensor::CollapseShapeOp collapseOp) {
        return foldCollapseShapeIntoMapScatter(rewriter, collapseOp,
                                               mapScatterOp);
      })
      .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp extractSliceOp) {
        return foldExtractSliceIntoMapScatter(rewriter, extractSliceOp,
                                              mapScatterOp);
      })
      .Case<tensor::PadOp>([&](tensor::PadOp padOp) {
        return foldPadIntoMapScatter(rewriter, padOp, mapScatterOp,
                                     padDistributionConfigFn);
      })
      .Default([](Operation *) { return failure(); });
}

/// Starting from the `root`, iteratively combine any relayout op producers
/// into a single iree_linalg_ext.map_scatter op. An identity map_scatter op
/// is inserted before the root, and then the producers of the map_scatter op
/// are folded into the map_scatter until an unsupported op is reached.
static void
combineRelayoutOpChain(RewriterBase &rewriter, MapScatterOp mapScatterOp,
                       PadDistributionConfigFn padDistributionConfigFn) {
  Operation *relayoutOp = mapScatterOp.getInput().getDefiningOp();
  if (!relayoutOp) {
    return;
  }
  MapScatterOp combinedRelayoutOp = mapScatterOp;
  while (relayoutOp) {
    LDBG("Attempting to fold " << relayoutOp->getName()
                               << " into map_scatter op:\n"
                               << *relayoutOp);
    FailureOr<MapScatterOp> maybeCombinedRelayoutOp = foldIntoMapScatter(
        rewriter, relayoutOp, combinedRelayoutOp, padDistributionConfigFn);
    if (failed(maybeCombinedRelayoutOp)) {
      LDBG("Failed to fold " << relayoutOp->getName()
                             << " into map_scatter op");
      break;
    }
    combinedRelayoutOp = maybeCombinedRelayoutOp.value();
    LDBG("Successfully folded " << relayoutOp->getName()
                                << " into map_scatter. New map_scatter op:\n"
                                << combinedRelayoutOp);
    relayoutOp = combinedRelayoutOp.getInput().getDefiningOp();
  }
  if (combinedRelayoutOp.isIdentity()) {
    rewriter.replaceOp(combinedRelayoutOp, combinedRelayoutOp.getInput());
  }
}

static MapScatterOp
insertIdentityMapScatter(RewriterBase &rewriter,
                         IREE::Codegen::StoreToBufferOp storeOp) {
  Location loc = storeOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(storeOp);
  auto mapScatterDest =
      rewriter
          .create<tensor::EmptyOp>(
              loc, memref::getMixedSizes(rewriter, loc, storeOp.getBuffer()),
              storeOp.getTensor().getType().getElementType())
          .getResult();
  auto mapScatterOp = MapScatterOp::createIdentityMapScatter(
      rewriter, loc, storeOp.getTensor(), mapScatterDest);
  rewriter.modifyOpInPlace(storeOp, [&]() {
    storeOp.getTensorMutable().assign(mapScatterOp.getResult(0));
  });
  LDBG("Created identity map_scatter:\n" << mapScatterOp);
  return mapScatterOp;
}

LogicalResult
combineLayoutTransformation(MLIRContext *ctx, FunctionOpInterface funcOp,
                            PadDistributionConfigFn padDistributionConfigFn) {
  // Sink relayout operations to the end of the funcOp.
  RewritePatternSet propagationPatterns(ctx);
  tensor::populateFoldTensorEmptyPatterns(propagationPatterns);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(propagationPatterns, ctx);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(propagationPatterns,
                                                       ctx);
  // Only sink reshape ops, so bail if the consumer operation is a reshape.
  auto controlSinkReshapesFn = [](OpOperand *operand) -> bool {
    Operation *consumer = operand->getOwner();
    return !llvm::isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(consumer);
  };
  linalg::populateFoldReshapeOpsByExpansionPatterns(propagationPatterns,
                                                    controlSinkReshapesFn);
  // Only sink unpack ops, so bail if the producer operation is not an unpack.
  // Also only sink unpack ops when new pack operations will not be created.
  // This means the consumer op must have at most one additional destination
  // operand, and it must come from an empty tensor.
  auto controlPropagationFn = [](OpOperand *operand) -> bool {
    Operation *producer = operand->get().getDefiningOp();
    Operation *consumer = operand->getOwner();
    if (!isa_and_nonnull<linalg::UnPackOp>(producer)) {
      return false;
    }
    // Pads and reshapes will not produce extra pack ops.
    if (isa<tensor::PadOp, tensor::ExpandShapeOp>(consumer)) {
      return true;
    }
    // Otherwise, the consumer must be a GenericOp with all of its `outs`
    // operands coming from tensor.empty ops, and the `operand` must be the
    // sole `ins` operand of the generic op. This ensures that no additional
    // linalg.pack ops will be created on other inputs of the generic op.
    // TODO(Max191): Remove the restriction of not creating new pack ops once
    // codegen can handle it.
    auto genericConsumer = dyn_cast<linalg::GenericOp>(consumer);
    if (!genericConsumer || genericConsumer.getNumDpsInputs() != 1 ||
        *genericConsumer.getDpsInputOperand(0) != *operand) {
      return false;
    }
    return llvm::all_of(
        genericConsumer.getDpsInits(), [&](Value consumerOperand) -> bool {
          return consumerOperand.getDefiningOp<tensor::EmptyOp>();
        });
  };
  linalg::populateDataLayoutPropagationPatterns(propagationPatterns,
                                                controlPropagationFn);
  // TODO(Max191): The propagation patterns could be applied at the same time as
  // relayout ops are folded into the map_scatter, which may enable even more
  // folding. This requires the relayout op folding to be done as pattern
  // rewrites, and also direct foldings for pack and unpack ops.
  if (failed(applyPatternsGreedily(funcOp, std::move(propagationPatterns)))) {
    return failure();
  }

  // Apply some preprocessing to convert complex layout transformation
  // ops like pack and unpack into simpler supported ops.
  IRRewriter rewriter(ctx);
  simplifyComplexRelayoutOps(rewriter, funcOp);

  // Start from iree_codegen.store_to_buffer ops, and combine producer
  // relayout ops into a single map_scatter.
  SmallVector<IREE::Codegen::StoreToBufferOp> dispatchResults(
      funcOp.getFunctionBody().getOps<IREE::Codegen::StoreToBufferOp>());
  for (IREE::Codegen::StoreToBufferOp dispatchResult : dispatchResults) {
    MapScatterOp mapScatterOp =
        insertIdentityMapScatter(rewriter, dispatchResult);
    combineRelayoutOpChain(rewriter, mapScatterOp, padDistributionConfigFn);
  }

  // Cleanup any tensor.dim ops that may be present after relayout
  // combination.
  RewritePatternSet cleanupPatterns(ctx);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(cleanupPatterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(cleanupPatterns)))) {
    return failure();
  }
  return success();
}

/// TODO(#20530): Improve heuristic for tile size selection.
static SmallVector<DistributionConfig>
defaultPadWorkgroupDistributionConfigFn(ArrayRef<int64_t> iterationBounds,
                                        MLIRContext *ctx) {
  DistributionConfig workgroupDistributionConfig;
  workgroupDistributionConfig.tileSizes.assign(iterationBounds.size(), 1);
  workgroupDistributionConfig.tileSizes.back() = 64;
  workgroupDistributionConfig.mapping = llvm::map_to_vector(
      llvm::seq<int64_t>(iterationBounds.size()),
      [&](int64_t dim) -> Attribute {
        switch (dim) {
        case 0:
        case 1:
        case 2:
          return IREE::Codegen::WorkgroupMappingAttr::get(
              ctx, IREE::Codegen::symbolizeWorkgroupId(dim).value());
        default:
          return IREE::Codegen::WorkgroupMappingAttr::get(
              ctx, IREE::Codegen::WorkgroupId::IdZ, dim - 2);
        }
      });
  std::reverse(workgroupDistributionConfig.mapping.begin(),
               workgroupDistributionConfig.mapping.end());
  return {workgroupDistributionConfig};
}

namespace {

struct CombineLayoutTransformationPass final
    : impl::CombineLayoutTransformationPassBase<
          CombineLayoutTransformationPass> {
  using impl::CombineLayoutTransformationPassBase<
      CombineLayoutTransformationPass>::CombineLayoutTransformationPassBase;

  void runOnOperation() override {
    if (failed(combineLayoutTransformation(
            &getContext(), getOperation(),
            defaultPadWorkgroupDistributionConfigFn))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
