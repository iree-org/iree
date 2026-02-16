// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CombineLayoutTransformation.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-combine-layout-transformation"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_COMBINERESULTLAYOUTTRANSFORMATIONPASS
#define GEN_PASS_DEF_COMBINESOURCELAYOUTTRANSFORMATIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using IREE::LinalgExt::MapLoadOp;
using IREE::LinalgExt::MapStoreOp;

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

/// Folds an `op` that does not affect index computation into a `mapStoreOp`.
/// This is used for ops like `linalg::CopyOp`.
static MapStoreOp foldIdentityLikeOpIntoMapStore(RewriterBase &rewriter,
                                                 Operation *op,
                                                 MapStoreOp mapStoreOp) {
  assert(mapStoreOp.getInput() == op->getResult(0) &&
         "expected op to be the producer of mapStoreOp");
  rewriter.modifyOpInPlace(mapStoreOp, [&]() {
    mapStoreOp.getInputMutable().assign(op->getOperand(0));
  });
  return mapStoreOp;
}

/// Fold a `transposeOp` into a consumer `mapStoreOp`, by transposing the
/// uses of the `mapStoreOp`s transformation_region block arguments.
static MapStoreOp foldTransposeIntoMapStore(RewriterBase &rewriter,
                                            linalg::TransposeOp transposeOp,
                                            MapStoreOp mapStoreOp) {
  assert(mapStoreOp.getInput() == transposeOp->getResult(0) &&
         "expected transposeOp to be the producer of mapStoreOp");

  SmallVector<int64_t> perm(transposeOp.getPermutation());
  rewriter.modifyOpInPlace(mapStoreOp, [&]() {
    mapStoreOp.insertTransformationAtStart(
        rewriter,
        [perm](ArrayRef<BlockArgument> indices) -> SmallVector<Value> {
          SmallVector<Value> indexValues(indices.begin(), indices.end());
          return applyPermutation(indexValues, perm);
        },
        perm.size());
    mapStoreOp.getInputMutable().assign(transposeOp.getInput());
  });
  return mapStoreOp;
}

/// Fold a tensor::ExpandShapeOp or tensor::CollapseShapeOp into a consumer
/// `mapStoreOp`, by linearizing and then delinearizing the source indices
/// of the `mapStoreOp`s index transformation.
template <typename ReshapeOpTy>
static IREE::LinalgExt::MapStoreOp
foldReshapeIntoMapStore(RewriterBase &rewriter, ReshapeOpTy reshapeOp,
                        IREE::LinalgExt::MapStoreOp mapStoreOp) {
  assert(mapStoreOp.getInput() == reshapeOp->getResult(0) &&
         "expected reshapeOp to be the producer of mapStoreOp");
  Location loc = reshapeOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(reshapeOp);
  SmallVector<OpFoldResult> srcDims =
      tensor::getMixedSizes(rewriter, loc, reshapeOp.getSrc());
  // There can be leftover tensor.dim ops consuming the result of the reshape,
  // but they are expected to be folded into some affine.apply ops on the source
  // sizes by later cleanup patterns.
  SmallVector<OpFoldResult> resultDims =
      tensor::getMixedSizes(rewriter, loc, reshapeOp.getResult());

  Location mapStoreLoc = mapStoreOp->getLoc();
  rewriter.modifyOpInPlace(mapStoreOp, [&]() {
    mapStoreOp.insertTransformationAtStart(
        rewriter,
        [&rewriter, mapStoreLoc, srcDims,
         resultDims](ArrayRef<BlockArgument> indices) -> SmallVector<Value> {
          SmallVector<Value> indexValues(indices.begin(), indices.end());
          auto linearizeIndexOp = affine::AffineLinearizeIndexOp::create(
              rewriter, mapStoreLoc, indexValues, srcDims, /*disjoint=*/true);
          auto delinearizeIndexOp = affine::AffineDelinearizeIndexOp::create(
              rewriter, mapStoreLoc, linearizeIndexOp.getResult(), resultDims,
              /*hasOuterBound=*/true);
          return delinearizeIndexOp->getResults();
        },
        srcDims.size());
    mapStoreOp.getInputMutable().assign(reshapeOp->getOperand(0));
  });
  return mapStoreOp;
}

/// Fold a tensor::ExpandShapeOp into a consumer `mapStoreOp`, by linearizing
/// and then delinearizing the source indices of the `mapStoreOp`s index
/// transformation.
static MapStoreOp
foldExpandShapeIntoMapStore(RewriterBase &rewriter,
                            tensor::ExpandShapeOp expandShapeOp,
                            MapStoreOp mapStoreOp) {
  return foldReshapeIntoMapStore(rewriter, expandShapeOp, mapStoreOp);
}

/// Fold a tensor::CollapseShapeOp into a consumer `mapStoreOp`, by
/// linearizing and then delinearizing the source indices of the
/// `mapStoreOp`s index transformation.
static MapStoreOp
foldCollapseShapeIntoMapStore(RewriterBase &rewriter,
                              tensor::CollapseShapeOp collapseShapeOp,
                              MapStoreOp mapStoreOp) {
  return foldReshapeIntoMapStore(rewriter, collapseShapeOp, mapStoreOp);
}

/// Fold an `extractSliceOp` into a consumer `mapStoreOp`. If the
/// extract_slice is an identity operation, the input is replaced directly.
/// Otherwise, a mask is applied based on the bounds of the extractSliceOp.
/// Currently, only zero offsets and unit strides are supported.
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
static FailureOr<MapStoreOp>
foldExtractSliceIntoMapStore(RewriterBase &rewriter,
                             tensor::ExtractSliceOp extractSliceOp,
                             MapStoreOp mapStoreOp) {
  assert(mapStoreOp.getInput() == extractSliceOp->getResult(0) &&
         "expected extractSliceOp to be the producer of mapStoreOp");
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

  // Check if this is an identity extract_slice (all sizes match the source).
  // If so, just replace the input without adding any masking.
  SmallVector<OpFoldResult> sliceSizes = extractSliceOp.getMixedSizes();
  Value source = extractSliceOp.getSource();
  bool isIdentity = true;
  for (auto [dim, sliceSize] : llvm::enumerate(sliceSizes)) {
    ValueBoundsConstraintSet::Variable sourceDimVar(source, dim);
    FailureOr<bool> areEqual =
        ValueBoundsConstraintSet::areEqual(sliceSize, sourceDimVar);
    if (!succeeded(areEqual) || !*areEqual) {
      isIdentity = false;
      break;
    }
  }

  if (isIdentity) {
    // Identity extract_slice: just replace the input without any masking.
    rewriter.modifyOpInPlace(
        mapStoreOp, [&]() { mapStoreOp.getInputMutable().assign(source); });
    return mapStoreOp;
  }

  SmallVector<OpFoldResult> bounds(extractSliceOp.getMixedSizes());
  Block &transformBody = mapStoreOp.getTransformationRegion().front();
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
  Value mask = yieldOp->getOperands().back();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(yieldOp);
  Location loc = mapStoreOp->getLoc();
  ArrayRef<BlockArgument> srcIndices = transformBody.getArguments();
  for (auto [bound, srcIdx] : llvm::zip_equal(bounds, srcIndices)) {
    Value boundValue = getValueOrCreateConstantIndexOp(rewriter, loc, bound);
    auto isOutOfBounds =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult, srcIdx,
                              boundValue)
            ->getResult(0);
    mask = arith::AndIOp::create(rewriter, loc, mask, isOutOfBounds);
  }
  rewriter.modifyOpInPlace(yieldOp, [&]() {
    yieldOp->setOperand(yieldOp->getNumOperands() - 1, mask);
  });
  rewriter.modifyOpInPlace(mapStoreOp, [&]() {
    mapStoreOp.getInputMutable().assign(extractSliceOp.getSource());
  });
  return mapStoreOp;
}

static void buildNestedDistributionLoops(
    RewriterBase &rewriter, Location loc, int64_t distributionLevel,
    SmallVector<OpFoldResult> lbs, SmallVector<OpFoldResult> ubs,
    ArrayRef<DistributionConfig> distConfigs,
    function_ref<void(OpBuilder &, Location, ValueRange)> innerLoopBuilder) {
  DistributionConfig distConfig = distConfigs[distributionLevel];
  SmallVector<OpFoldResult> steps =
      getAsIndexOpFoldResult(rewriter.getContext(), distConfig.tileSizes);
  scf::ForallOp::create(
      rewriter, loc, lbs, ubs, steps, /*outputs=*/ValueRange(),
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
          scf::InParallelOp::create(b, nestedLoc);
          return;
        }
        // Otherwise, tile to one, and generate the inner loop body.
        SmallVector<Value> nestedLbVals =
            getValueOrCreateConstantIndexOp(b, nestedLoc, nestedLbs);
        SmallVector<Value> nestedUbVals =
            getValueOrCreateConstantIndexOp(b, nestedLoc, nestedUbs);
        Value one = arith::ConstantIndexOp::create(rewriter, nestedLoc, 1);
        SmallVector<Value> unitSteps(nestedLbs.size(), one);
        scf::buildLoopNest(rewriter, nestedLoc, nestedLbVals, nestedUbVals,
                           unitSteps, innerLoopBuilder);
        scf::InParallelOp::create(b, nestedLoc);
      });
}

FailureOr<MapStoreOp>
foldPadIntoMapStore(RewriterBase &rewriter, tensor::PadOp padOp,
                    MapStoreOp mapStoreOp,
                    PadDistributionConfigFn padDistributionConfigFn) {
  // Find the output buffer that the mapStoreOp is stored into.
  if (!mapStoreOp->hasOneUse()) {
    return rewriter.notifyMatchFailure(mapStoreOp,
                                       "map_store does not have a single user");
  }
  auto storeOp =
      dyn_cast<IREE::Codegen::StoreToBufferOp>(*mapStoreOp->getUsers().begin());
  if (!storeOp) {
    return rewriter.notifyMatchFailure(
        mapStoreOp, "map_store user is not an iree_codegen.store_to_buffer op");
  }

  rewriter.setInsertionPointAfter(storeOp);
  Location loc = padOp->getLoc();
  SmallVector<OpFoldResult> padSrcSizes =
      tensor::getMixedSizes(rewriter, loc, padOp.getSource());
  SmallVector<OpFoldResult> padResultSizes =
      tensor::getMixedSizes(rewriter, loc, padOp.getResult());

  // Write the padding values directly into the outputBuffer.
  Value outputBuffer = storeOp.getBuffer();
  auto innerLoopBuilder = [&](OpBuilder &b, Location loopLoc, ValueRange ivs) {
    // We need to scatter the padding values according to the existing
    // mapStoreOp transformation, so clone the transformation into the
    // loop nest.
    IRMapping mapping;
    Region &transformRegion = mapStoreOp.getTransformationRegion();
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
    SmallVector<Value> storeIndices(yieldOp.getOperands());
    rewriter.eraseOp(yieldOp);
    Value mask = storeIndices.pop_back_val();
    // Create the store to the outputBuffer.
    auto thenBuilder = [&](OpBuilder &nestedBuilder, Location ifLoc) {
      memref::StoreOp::create(nestedBuilder, ifLoc,
                              padOp.getConstantPaddingValue(), outputBuffer,
                              storeIndices);
      scf::YieldOp::create(nestedBuilder, ifLoc);
    };
    scf::IfOp::create(b, loopLoc, mask, thenBuilder);
  };

  auto stopConditionFn = [](Value /*v*/, std::optional<int64_t> /*dim*/,
                            ValueBoundsConstraintSet & /*cstr*/) -> bool {
    return false;
  };

  // Prove that range [lb, ub) is empty (lb >= ub).
  //
  // For high padding, we have:
  //   lb = low + srcSize
  //   ub = resultSize
  // and for a well-formed pad, resultSize = srcSize + low + high, so:
  //   ub - lb = high
  // Therefore "lb >= ub" -> "high == 0". Expressing the check in terms of lb/ub
  // keeps it robust even when pad amount has been rewritten into a complex
  // form.
  auto proveEmptyRange = [&](RewriterBase &rewriter, Location loc, Value lb,
                             Value ub) -> bool {
    // Model diff = ub - lb  as an affine map (d0, d1) -> (d0 - d1)
    // and ask for a constant upper bound on diff.
    // If we can prove UB(diff) <= 0, then "diff <= 0" -> "ub - lb <= 0" ->
    // "lb >= ub".
    MLIRContext *ctx = rewriter.getContext();
    auto diffMap = AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0,
        {getAffineDimExpr(0, ctx) - getAffineDimExpr(1, ctx)}, ctx);
    auto ubConstantBound = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::UB,
        ValueBoundsConstraintSet::Variable(diffMap, ValueRange{ub, lb}),
        /*stopCondition=*/stopConditionFn,
        /*addConservativeSemiAffineBounds=*/true);
    return succeeded(ubConstantBound) && *ubConstantBound <= 0;
  };

  // Distribute the padding of each dimension separately. This causes some
  // overlap of the iteration spaces across the loops, but simplifies the
  // implementation. The trade-off is expected to be okay because we expect
  // the padding to be small relative to the size of the output buffer, which
  // makes the overlap small. The below example shows what the overlap looks
  // like.
  //
  // 1st loop (low pad d0)        2nd loop (high pad d0)
  //        |———|                       |———————|
  //          |                             |
  //          v                             v
  //        +———+———————————————————————+———————+     ———
  //        | . | . . . . . . . . . . . | . . . | <--  |  3rd loop (low pad d1)
  //        +———+———————————————————————+———————+     ———
  //        | . | . . . . . . . . . . . | . . . |
  //        | . | . . . . . . . . . . . | . . . |
  //        | . | . . . . . . . . . . . | . . . |
  //        | . | . . . . . . . . . . . | . . . |
  //        +———+———————————————————————+———————+     ———
  //        | . | . . . . . . . . . . . | . . . | <--  |  4th loop (high pad d1)
  //        +———+———————————————————————+———————+     ———
  //
  // Each distributed loop performs the padding for a hyperplane spanning the
  // low or high pad of the corresponding dimension, and fully spanning all
  // other dimensions of the tensor. This means that there is overlap on the
  // intersection of high and low padding for each dimension.
  //
  // In most cases today, the low padding is generally either 0, or very small,
  // and the high padding will be small relative to the size of the full tensor,
  // since padding is usually done to the nearest multiple of some tile size, or
  // for padding an image for a convolution. Because the padding is small, the
  // overlap should also be small, and we can live with the small redundant
  // computation in order to have a more simplified kernel.
  for (auto [idx, low, high] :
       llvm::enumerate(padOp.getMixedLowPad(), padOp.getMixedHighPad())) {
    // Create a distributed loop for the low padding.
    if (!isConstantIntValue(low, 0)) {
      SmallVector<OpFoldResult> ubs(padResultSizes);
      SmallVector<OpFoldResult> lbs(ubs.size(), rewriter.getIndexAttr(0));
      SmallVector<int64_t> shape(padOp.getSourceType().getShape());
      shape[idx] = padOp.getStaticLow()[idx];
      ubs[idx] = low;
      SmallVector<DistributionConfig> distConfigs = padDistributionConfigFn(
          padOp.getSourceType().getShape(), rewriter.getContext());
      buildNestedDistributionLoops(rewriter, loc, /*distributionLevel=*/0, lbs,
                                   ubs, distConfigs, innerLoopBuilder);
    }
    // Create a distributed loop for the high padding.
    if (isConstantIntValue(high, 0)) {
      continue;
    }
    SmallVector<OpFoldResult> ubs(padResultSizes);
    SmallVector<OpFoldResult> lbs(ubs.size(), rewriter.getIndexAttr(0));
    SmallVector<int64_t> shape(padOp.getSourceType().getShape());
    shape[idx] = padOp.getStaticHigh()[idx];
    lbs[idx] = IREE::LinalgExt::addOfrs(rewriter, loc, low, padSrcSizes[idx]);
    // ubs[idx] is padResultSizes[idx] already (since ubs initialized from
    // padResultSizes)

    // Fallback: Even if we couldn't prove the pad amount is exactly 0, the
    // derived half-open iteration space [lb, ub) may still be empty. For
    // well-formed pads this typically means lb == ub (i.e., the high pad is 0),
    // but we phrase it as the general condition lb >= ub for half-open ranges.
    // If it's empty, skip generating this loop nest.
    {
      Value lbV = getValueOrCreateConstantIndexOp(rewriter, loc, lbs[idx]);
      Value ubV = getValueOrCreateConstantIndexOp(rewriter, loc, ubs[idx]);

      if (lbV == ubV) {
        continue;
      }

      if (proveEmptyRange(rewriter, loc, lbV, ubV)) {
        continue;
      }
    }

    SmallVector<DistributionConfig> distConfigs =
        padDistributionConfigFn(shape, rewriter.getContext());
    buildNestedDistributionLoops(rewriter, loc, /*distributionLevel=*/0, lbs,
                                 ubs, distConfigs, innerLoopBuilder);
  }

  // Now that the padding values are being written to the outputBuffer, the
  // padOp becomes a no-op with respect to the index transformation on the
  // non-padded values.
  rewriter.modifyOpInPlace(mapStoreOp, [&]() {
    mapStoreOp.getInputMutable().assign(padOp.getSource());
  });
  return mapStoreOp;
}

FailureOr<MapStoreOp> foldIntoMapStore(RewriterBase &rewriter, Operation *op,
                                       MapStoreOp mapStoreOp) {
  return llvm::TypeSwitch<Operation *, FailureOr<MapStoreOp>>(op)
      .Case([&](linalg::CopyOp copyOp) {
        return foldIdentityLikeOpIntoMapStore(rewriter, copyOp, mapStoreOp);
      })
      .Case([&](linalg::TransposeOp transposeOp) {
        return foldTransposeIntoMapStore(rewriter, transposeOp, mapStoreOp);
      })
      .Case([&](tensor::ExpandShapeOp expandOp) {
        return foldExpandShapeIntoMapStore(rewriter, expandOp, mapStoreOp);
      })
      .Case([&](tensor::CollapseShapeOp collapseOp) {
        return foldCollapseShapeIntoMapStore(rewriter, collapseOp, mapStoreOp);
      })
      .Case([&](tensor::ExtractSliceOp extractSliceOp) {
        return foldExtractSliceIntoMapStore(rewriter, extractSliceOp,
                                            mapStoreOp);
      })
      .Default(failure());
}

// Insert identity map_store op after the root and replace all uses.
static MapStoreOp insertIdentityMapStore(RewriterBase &rewriter,
                                         OpResult root) {
  Location loc = root.getLoc();
  SetVector<OpOperand *> originalUses;
  for (OpOperand &use : root.getUses()) {
    originalUses.insert(&use);
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfterValue(root);
  Type elementType = cast<RankedTensorType>(root.getType()).getElementType();
  SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(rewriter, loc, root);
  Value mapStoreDest =
      tensor::EmptyOp::create(rewriter, loc, sizes, elementType);
  auto mapStoreOp =
      MapStoreOp::createIdentityMapStore(rewriter, loc, root, mapStoreDest);
  rewriter.replaceUsesWithIf(
      root, mapStoreOp.getResult(0),
      [&](OpOperand &use) { return originalUses.contains(&use); });
  LDBG() << "Created identity map_store:\n" << mapStoreOp;
  return mapStoreOp;
}

bool isSupportedSingleInputRelayoutOp(Operation *op) {
  return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
             tensor::ExtractSliceOp, tensor::PadOp, linalg::CopyOp,
             linalg::TransposeOp>(op);
}

// This is only desirable in the dispatch scope but not in the workgroup scope.
static bool
shouldDoReshapesByExpansion(IREE::Codegen::RelayoutCombinationScope scope) {
  if (scope == IREE::Codegen::RelayoutCombinationScope::Dispatch) {
    return true;
  }
  return false;
}

/// Insert identity map_store ops after the given operation if it is a valid
/// leaf op of a relayout op chain. A relayout op chain is a sequence of
/// relayout ops (defined by `isSupportedSingleInputRelayoutOp`) for which the
/// only users of the ops in the chain are relayout ops, except for the leaves
/// of the chain. The leaves are simply relayout ops that have non relayout op
/// users. The `controlFn` is a callback on the leaf OpResult that provides
/// control over whether or not to insert a map_store op.
struct InsertMapStoreOpPattern : public RewritePattern {
  InsertMapStoreOpPattern(MLIRContext *context,
                          CombineRelayoutOpsControlFnRef controlFn = nullptr,
                          PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        controlFn(controlFn) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isSupportedSingleInputRelayoutOp(op)) {
      return failure();
    }
    // Relayout ops with only relayout op users are not leaves.
    auto isDimOrSupportedRelayoutOp = [](Operation *op) {
      return isSupportedSingleInputRelayoutOp(op) || isa<tensor::DimOp>(op);
    };
    if (llvm::all_of(op->getUsers(), isDimOrSupportedRelayoutOp)) {
      return failure();
    }
    // All relayout ops have a single result.
    OpResult leaf = op->getResult(0);
    if (controlFn && !controlFn(leaf)) {
      return failure();
    }
    (void)insertIdentityMapStore(rewriter, leaf);
    return success();
  }

private:
  CombineRelayoutOpsControlFnRef controlFn;
};

LogicalResult
combineLayoutTransformation(MLIRContext *ctx, FunctionOpInterface funcOp,
                            PadDistributionConfigFn padDistributionConfigFn,
                            bool doReshapeByExpansion,
                            CombineRelayoutOpsControlFnRef controlFn) {
  // Sink relayout operations to the end of the funcOp.
  RewritePatternSet propagationPatterns(ctx);
  tensor::populateFoldTensorEmptyPatterns(propagationPatterns);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(propagationPatterns, ctx);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(propagationPatterns,
                                                       ctx);
  if (doReshapeByExpansion) {
    // Only sink reshape ops, so bail if the consumer operation is a reshape.
    auto controlSinkReshapesFn = [](OpOperand *operand) -> bool {
      Operation *consumer = operand->getOwner();
      return !isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(consumer);
    };
    linalg::populateFoldReshapeOpsByExpansionPatterns(propagationPatterns,
                                                      controlSinkReshapesFn);
  }
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
  linalg::populateDataLayoutPropagationPatterns(
      propagationPatterns, controlPropagationFn, /*PoisonPaddingOk=*/true);
  // TODO(Max191): The propagation patterns could be applied at the same time as
  // relayout ops are folded into the map_store, which may enable even more
  // folding. This requires the relayout op folding to be done as pattern
  // rewrites, and also direct foldings for pack and unpack ops.
  if (failed(applyPatternsGreedily(funcOp, std::move(propagationPatterns)))) {
    return failure();
  }

  // Apply some preprocessing to convert complex layout transformation
  // ops like pack and unpack into simpler supported ops.
  IRRewriter rewriter(ctx);
  simplifyComplexRelayoutOps(rewriter, funcOp);

  // Resolve dims aggressively and canonicalize to simplify the IR.
  // This improves pattern matching and makes it easier for subsequent
  // rewrites/proofs to recognize identity/no-op cases.
  {
    RewritePatternSet patterns(ctx);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    IREE::Util::AssumeIntOp::getCanonicalizationPatterns(patterns, ctx);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return failure();
    }
  }

  // Combine relayout operations into new the map_store ops.
  RewritePatternSet relayoutCombinationPatterns(ctx);
  relayoutCombinationPatterns.add<InsertMapStoreOpPattern>(ctx, controlFn);
  populateCombineRelayoutOpPatterns(relayoutCombinationPatterns,
                                    padDistributionConfigFn);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(
      relayoutCombinationPatterns);
  if (failed(applyPatternsGreedily(funcOp,
                                   std::move(relayoutCombinationPatterns)))) {
    return failure();
  }

  // Clean up any identity map_store ops after combining.
  funcOp->walk([&](MapStoreOp mapStoreOp) {
    if (mapStoreOp.isIdentity()) {
      rewriter.replaceOp(mapStoreOp, mapStoreOp.getInput());
    }
  });
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

CombineRelayoutOpsControlFn
getCombineRelayoutOpsControlFn(IREE::Codegen::RelayoutCombinationScope scope) {
  CombineRelayoutOpsControlFn controlFn;
  switch (scope) {
  // Control function for Dispatch scope. Filters to only relayout ops with
  // a single iree_codegen.store_to_buffer user.
  case IREE::Codegen::RelayoutCombinationScope::Dispatch:
    controlFn = [](OpResult leaf) {
      if (leaf.getNumUses() != 1) {
        return false;
      }
      return isa<IREE::Codegen::StoreToBufferOp>(*leaf.getUsers().begin());
    };
    break;
  // Control function for Workgroup scope. Filters to only relayout ops with
  // a single tensor.parallel_insert_slice user inside of a workgroup
  // scf.forall op. Relayout chains of only reshapes are also filtered out,
  // because these chains can usually be handled by bufferization.
  case IREE::Codegen::RelayoutCombinationScope::Workgroup:
    controlFn = [](OpResult leaf) {
      if (leaf.getNumUses() != 1) {
        return false;
      }
      auto parallelInsertOp =
          dyn_cast<tensor::ParallelInsertSliceOp>(*leaf.getUsers().begin());
      if (!parallelInsertOp) {
        return false;
      }
      auto forallOp = parallelInsertOp->getParentOfType<scf::ForallOp>();
      if (!forallOp || !forallOp.getMapping() ||
          !llvm::all_of(forallOp.getMapping().value(),
                        llvm::IsaPred<IREE::Codegen::WorkgroupMappingAttr>)) {
        return false;
      }
      auto bbArg = dyn_cast<BlockArgument>(parallelInsertOp.getDest());
      if (!bbArg || forallOp.getCombiningOps(bbArg).size() != 1) {
        return false;
      }
      // If there are only reshape ops, then bufferization can usually handle
      // it, so don't introduce map_store.
      llvm::SetVector<Operation *> slice;
      BackwardSliceOptions options;
      options.filter = isSupportedSingleInputRelayoutOp;
      options.inclusive = true;
      LogicalResult result =
          getBackwardSlice(parallelInsertOp.getSource(), &slice, options);
      if (failed(result)) {
        return false;
      }
      return !llvm::all_of(
          slice, llvm::IsaPred<tensor::CollapseShapeOp, tensor::ExpandShapeOp>);
    };
    break;
  }
  return controlFn;
}

namespace {

struct CombineResultLayoutTransformationPass final
    : impl::CombineResultLayoutTransformationPassBase<
          CombineResultLayoutTransformationPass> {
  using Base::Base;

  void runOnOperation() override {
    CombineRelayoutOpsControlFn controlFn =
        getCombineRelayoutOpsControlFn(this->scope);
    bool doReshapesByExpansion = shouldDoReshapesByExpansion(this->scope);
    if (failed(
            combineLayoutTransformation(&getContext(), getOperation(),
                                        defaultPadWorkgroupDistributionConfigFn,
                                        doReshapesByExpansion, controlFn))) {
      return signalPassFailure();
    }

    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();
    {
      RewritePatternSet patterns(context);
      populateFuseTilableForallConsumersPattern(patterns);
      scf::ForallOp::getCanonicalizationPatterns(patterns, context);
      tensor::populateFoldTensorEmptyPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Combining Layout Transformation Ops into MapLoadOp
//===----------------------------------------------------------------------===//

/// Folds `consumerOp` into the producer `mapLoadOp` using `rewriter`.
/// The `indexTransformBuilder` lambda defines how output indices of the new
/// map_load map to input indices of the original map_load. Optionally,
/// `newFillValue` can be provided to override the fill value (e.g., for pad).
///
/// This handles the shared boilerplate:
/// 1. Create new dest tensor from consumer result shape
/// 2. Clone map_load with new dest and region
/// 3. Apply index transformation via insertTransformationAtStart
/// 4. Optionally update fill value
/// 5. Replace consumer with new map_load result
static FailureOr<MapLoadOp> foldConsumerIntoMapLoadImpl(
    RewriterBase &rewriter, Operation *consumerOp, MapLoadOp mapLoadOp,
    function_ref<SmallVector<Value>(ArrayRef<BlockArgument>)>
        indexTransformBuilder,
    std::optional<Value> newFillValue = std::nullopt) {
  Location loc = consumerOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(consumerOp);

  // Create new dest tensor matching consumer output shape.
  Value consumerResult = consumerOp->getResult(0);
  Type elementType = getElementTypeOrSelf(consumerResult.getType());
  SmallVector<OpFoldResult> newSizes =
      tensor::getMixedSizes(rewriter, loc, consumerResult);
  Value newDest = tensor::EmptyOp::create(rewriter, loc, newSizes, elementType);

  // Clone the map_load with new dest.
  auto newMapLoad = MapLoadOp::create(rewriter, loc, newDest.getType(),
                                      mapLoadOp.getSource(), newDest);
  rewriter.cloneRegionBefore(mapLoadOp.getTransformationRegion(),
                             newMapLoad.getTransformationRegion(),
                             newMapLoad.getTransformationRegion().begin());

  // Prepend the index transformation.
  int64_t newRank = cast<ShapedType>(consumerResult.getType()).getRank();
  newMapLoad.insertTransformationAtStart(rewriter, indexTransformBuilder,
                                         newRank);

  // Optionally update fill value (for pad ops).
  if (newFillValue.has_value()) {
    Block &transformBody = newMapLoad.getTransformationRegion().front();
    auto yieldOp =
        cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp.setOperand(yieldOp.getNumOperands() - 1, *newFillValue);
    });
  }

  rewriter.replaceOp(consumerOp, newMapLoad.getResult(0));
  LDBG() << "Folded consumer " << consumerOp->getName().getStringRef()
         << " into map_load:\n"
         << newMapLoad;
  return newMapLoad;
}

/// Fold an `op` that does not affect index computation into a `mapLoadOp`.
/// This is used for ops like `linalg::CopyOp`.
static FailureOr<MapLoadOp>
foldIdentityLikeOpIntoMapLoad(RewriterBase &rewriter, Operation *op,
                              MapLoadOp mapLoadOp) {
  assert(op->getOperand(0) == mapLoadOp.getResult(0) &&
         "expected mapLoadOp to be the producer of op input");
  rewriter.replaceOp(op, mapLoadOp.getResult(0));
  LDBG() << "Folded consumer " << op->getName().getStringRef()
         << " into map_load (identity fold)";
  return mapLoadOp;
}

/// Fold a consumer `transposeOp` into a producer `mapLoadOp`.
/// For consumer folding, we apply the INVERSE permutation.
/// Example: perm=[1,2,0] means output[i,j,k] = input[k,i,j] (inverse=[2,0,1]).
static FailureOr<MapLoadOp>
foldTransposeIntoMapLoad(RewriterBase &rewriter,
                         linalg::TransposeOp transposeOp, MapLoadOp mapLoadOp) {
  assert(transposeOp.getInput() == mapLoadOp.getResult(0) &&
         "expected mapLoadOp to be the producer of transposeOp input");

  SmallVector<int64_t> inversePerm =
      invertPermutationVector(transposeOp.getPermutation());

  return foldConsumerIntoMapLoadImpl(
      rewriter, transposeOp, mapLoadOp,
      [inversePerm](ArrayRef<BlockArgument> indices) -> SmallVector<Value> {
        SmallVector<Value> indexValues(indices.begin(), indices.end());
        return applyPermutation(indexValues, inversePerm);
      });
}

/// Fold a consumer reshape op (expand_shape or collapse_shape) into a producer
/// `mapLoadOp`. Index transformation: linearize in result space, delinearize
/// in source space.
template <typename ReshapeOpTy>
static FailureOr<MapLoadOp> foldReshapeIntoMapLoad(RewriterBase &rewriter,
                                                   ReshapeOpTy reshapeOp,
                                                   MapLoadOp mapLoadOp) {
  assert(reshapeOp.getSrc() == mapLoadOp.getResult(0) &&
         "expected mapLoadOp to be the producer of reshapeOp input");

  Location loc = reshapeOp->getLoc();
  SmallVector<OpFoldResult> srcDims =
      tensor::getMixedSizes(rewriter, loc, reshapeOp.getSrc());
  SmallVector<OpFoldResult> resultDims =
      tensor::getMixedSizes(rewriter, loc, reshapeOp.getResult());

  return foldConsumerIntoMapLoadImpl(
      rewriter, reshapeOp, mapLoadOp,
      [&rewriter, loc, resultDims,
       srcDims](ArrayRef<BlockArgument> indices) -> SmallVector<Value> {
        SmallVector<Value> indexValues(indices.begin(), indices.end());
        auto linearizeIndexOp = affine::AffineLinearizeIndexOp::create(
            rewriter, loc, indexValues, resultDims, /*disjoint=*/true);
        auto delinearizeIndexOp = affine::AffineDelinearizeIndexOp::create(
            rewriter, loc, linearizeIndexOp.getResult(), srcDims,
            /*hasOuterBound=*/true);
        return delinearizeIndexOp->getResults();
      });
}

/// Fold a consumer tensor::ExpandShapeOp into a producer `mapLoadOp`.
static FailureOr<MapLoadOp>
foldExpandShapeIntoMapLoad(RewriterBase &rewriter,
                           tensor::ExpandShapeOp expandShapeOp,
                           MapLoadOp mapLoadOp) {
  return foldReshapeIntoMapLoad(rewriter, expandShapeOp, mapLoadOp);
}

/// Fold a consumer tensor::CollapseShapeOp into a producer `mapLoadOp`.
static FailureOr<MapLoadOp>
foldCollapseShapeIntoMapLoad(RewriterBase &rewriter,
                             tensor::CollapseShapeOp collapseShapeOp,
                             MapLoadOp mapLoadOp) {
  return foldReshapeIntoMapLoad(rewriter, collapseShapeOp, mapLoadOp);
}

/// Fold a consumer `extractSliceOp` into a producer `mapLoadOp`.
/// Index transformation: original_idx = offset + new_idx * stride
static FailureOr<MapLoadOp>
foldExtractSliceIntoMapLoad(RewriterBase &rewriter,
                            tensor::ExtractSliceOp extractSliceOp,
                            MapLoadOp mapLoadOp) {
  assert(extractSliceOp.getSource() == mapLoadOp.getResult(0) &&
         "expected mapLoadOp to be the producer of extractSliceOp input");

  if (extractSliceOp.getSourceType().getRank() !=
      extractSliceOp.getResultType().getRank()) {
    return rewriter.notifyMatchFailure(
        extractSliceOp, "rank reducing extract_slice op is not supported");
  }

  Location loc = extractSliceOp->getLoc();
  SmallVector<OpFoldResult> offsets = extractSliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> strides = extractSliceOp.getMixedStrides();

  auto indexTransformBuilder =
      [&](ArrayRef<BlockArgument> newOutputIndices) -> SmallVector<Value> {
    SmallVector<Value> originalIndices;
    for (auto [idx, offset, stride] :
         llvm::zip_equal(newOutputIndices, offsets, strides)) {
      Value offsetValue =
          getValueOrCreateConstantIndexOp(rewriter, loc, offset);
      Value strideValue =
          getValueOrCreateConstantIndexOp(rewriter, loc, stride);
      // original_idx = offset + idx * stride
      Value scaledIdx = arith::MulIOp::create(rewriter, loc, idx, strideValue,
                                              arith::IntegerOverflowFlags::nsw);
      Value originalIdx =
          arith::AddIOp::create(rewriter, loc, offsetValue, scaledIdx,
                                arith::IntegerOverflowFlags::nsw);
      originalIndices.push_back(originalIdx);
    }
    return originalIndices;
  };

  return foldConsumerIntoMapLoadImpl(rewriter, extractSliceOp, mapLoadOp,
                                     indexTransformBuilder);
}

/// Fold a consumer `padOp` into a producer `mapLoadOp`.
/// Index transformation: source_idx = new_idx - low_pad
/// Fill value is set to the pad value.
static FailureOr<MapLoadOp> foldPadIntoMapLoad(RewriterBase &rewriter,
                                               tensor::PadOp padOp,
                                               MapLoadOp mapLoadOp) {
  assert(padOp.getSource() == mapLoadOp.getResult(0) &&
         "expected mapLoadOp to be the producer of padOp input");

  // Only support constant pad values for now.
  Value padValue = padOp.getConstantPaddingValue();
  if (!padValue) {
    return rewriter.notifyMatchFailure(
        padOp, "non-constant padding value is not supported");
  }

  // Check if the map_load already has a real (non-poison) padding value.
  // If so, we cannot safely replace it as that would lose the existing padding.
  Value currentPadValue = mapLoadOp.getPaddingValue();
  if (!currentPadValue.getDefiningOp<ub::PoisonOp>()) {
    return rewriter.notifyMatchFailure(
        padOp, "map_load already has a non-poison padding value; folding "
               "another pad would overwrite it");
  }

  Location loc = padOp->getLoc();
  SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();

  auto indexTransformBuilder =
      [&](ArrayRef<BlockArgument> newOutputIndices) -> SmallVector<Value> {
    SmallVector<Value> sourceIndices;
    for (auto [idx, low] : llvm::zip_equal(newOutputIndices, lowPad)) {
      Value lowValue = getValueOrCreateConstantIndexOp(rewriter, loc, low);
      // source_idx = idx - low_pad
      Value sourceIdx = arith::SubIOp::create(rewriter, loc, idx, lowValue,
                                              arith::IntegerOverflowFlags::nsw);
      sourceIndices.push_back(sourceIdx);
    }
    return sourceIndices;
  };

  return foldConsumerIntoMapLoadImpl(rewriter, padOp, mapLoadOp,
                                     indexTransformBuilder, padValue);
}

/// Fold a consumer relayout op into a producer map_load.
FailureOr<MapLoadOp> foldIntoMapLoad(RewriterBase &rewriter, Operation *op,
                                     MapLoadOp mapLoadOp) {
  return llvm::TypeSwitch<Operation *, FailureOr<MapLoadOp>>(op)
      .Case<linalg::CopyOp>([&](linalg::CopyOp copyOp) {
        return foldIdentityLikeOpIntoMapLoad(rewriter, copyOp, mapLoadOp);
      })
      .Case<linalg::TransposeOp>([&](linalg::TransposeOp transposeOp) {
        return foldTransposeIntoMapLoad(rewriter, transposeOp, mapLoadOp);
      })
      .Case<tensor::ExpandShapeOp>([&](tensor::ExpandShapeOp expandOp) {
        return foldExpandShapeIntoMapLoad(rewriter, expandOp, mapLoadOp);
      })
      .Case<tensor::CollapseShapeOp>([&](tensor::CollapseShapeOp collapseOp) {
        return foldCollapseShapeIntoMapLoad(rewriter, collapseOp, mapLoadOp);
      })
      .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp extractSliceOp) {
        return foldExtractSliceIntoMapLoad(rewriter, extractSliceOp, mapLoadOp);
      })
      .Case<tensor::PadOp>([&](tensor::PadOp padOp) {
        return foldPadIntoMapLoad(rewriter, padOp, mapLoadOp);
      })
      .Default([](Operation *) { return failure(); });
}

/// Pattern to fold consumer relayout ops into a producer map_load.
struct FoldConsumerRelayoutIntoMapLoadPattern
    : public OpRewritePattern<IREE::LinalgExt::MapLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::MapLoadOp mapLoadOp,
                                PatternRewriter &rewriter) const override {
    // Find a consumer relayout op.
    Operation *consumerOp = nullptr;
    for (Operation *user : mapLoadOp->getUsers()) {
      if (isSupportedSingleInputRelayoutOp(user)) {
        consumerOp = user;
        break;
      }
    }
    if (!consumerOp) {
      return failure();
    }
    if (failed(foldIntoMapLoad(rewriter, consumerOp, mapLoadOp))) {
      return failure();
    }
    return success();
  }
};

// Insert identity map_load op after the root and replace uses.
static MapLoadOp insertIdentityMapLoad(RewriterBase &rewriter, OpResult root) {
  Location loc = root.getLoc();
  SetVector<OpOperand *> originalUses;
  for (OpOperand &use : root.getUses()) {
    originalUses.insert(&use);
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfterValue(root);
  Type elementType = getElementTypeOrSelf(root.getType());
  SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(rewriter, loc, root);
  Value mapLoadDest =
      tensor::EmptyOp::create(rewriter, loc, sizes, elementType);
  auto mapLoadOp =
      MapLoadOp::createIdentityMapLoad(rewriter, loc, root, mapLoadDest);
  rewriter.replaceUsesWithIf(root, mapLoadOp.getResult(0), [&](OpOperand &use) {
    return originalUses.contains(&use);
  });
  LDBG() << "Created identity map_load:\n" << mapLoadOp;
  return mapLoadOp;
}

/// Insert identity map_load op after a LoadFromBufferOp if it has relayout
/// op consumers. The identity map_load can then be used to fold consumer
/// relayout ops into it iteratively.
struct InsertMapLoadOpPattern
    : public OpRewritePattern<IREE::Codegen::LoadFromBufferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Codegen::LoadFromBufferOp loadOp,
                                PatternRewriter &rewriter) const override {
    // Check if the load has at least one relayout op user.
    bool hasRelayoutUser =
        llvm::any_of(loadOp->getUsers(), [](Operation *user) {
          return isSupportedSingleInputRelayoutOp(user);
        });
    if (!hasRelayoutUser) {
      return failure();
    }
    // Check that the load doesn't already have a map_load user (avoid
    // infinite loop).
    bool hasMapLoadUser = llvm::any_of(loadOp->getUsers(), [](Operation *user) {
      return isa<IREE::LinalgExt::MapLoadOp>(user);
    });
    if (hasMapLoadUser) {
      return failure();
    }
    (void)insertIdentityMapLoad(rewriter, loadOp->getResult(0));
    return success();
  }
};

namespace {

struct CombineSourceLayoutTransformationPass final
    : impl::CombineSourceLayoutTransformationPassBase<
          CombineSourceLayoutTransformationPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    // Apply some preprocessing to convert complex layout transformation
    // ops like unpack into simpler supported ops.
    IRRewriter rewriter(context);
    simplifyComplexRelayoutOps(rewriter, funcOp);

    // Insert identity map_load ops after load_from_buffer ops and fold
    // consumer relayout ops into them.
    RewritePatternSet patterns(context);
    patterns.add<InsertMapLoadOpPattern>(context);
    patterns.add<FoldConsumerRelayoutIntoMapLoadPattern>(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
