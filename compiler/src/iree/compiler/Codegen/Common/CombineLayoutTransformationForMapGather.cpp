// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CombineLayoutTransformationForMapGather.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-combine-layout-transformation-for-map-gather"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_COMBINELAYOUTTRANSFORMATIONFORMAPGATHERPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using IREE::LinalgExt::MapGatherOp;

//===----------------------------------------------------------------------===//
// Preprocessing Utilities
//===----------------------------------------------------------------------===//

/// Convert complex ops into simpler ops by decomposing or raising to a named
/// op.
///  - `UnPackOp`s are decomposed.
///  - Transpose `linalg::GenericOp`s are raised to `linalg::TransposeOp`s.
static void simplifyComplexRelayoutOpsForGather(RewriterBase &rewriter,
                                                FunctionOpInterface funcOp) {
  OpBuilder::InsertionGuard g(rewriter);
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
// Combining Layout Transformation Ops into MapGatherOp
//===----------------------------------------------------------------------===//

/// Folds an `op` that does not affect index computation into a `mapGatherOp`.
/// This is used for ops like `linalg::CopyOp`.
static MapGatherOp foldIdentityLikeOpIntoMapGather(RewriterBase &rewriter,
                                                   Operation *op,
                                                   MapGatherOp mapGatherOp) {
  assert(mapGatherOp.getSource() == op->getResult(0) &&
         "expected op to be the producer of mapGatherOp source");
  rewriter.modifyOpInPlace(mapGatherOp, [&]() {
    mapGatherOp.getSourceMutable().assign(op->getOperand(0));
  });
  return mapGatherOp;
}

/// Fold a `transposeOp` into a consumer `mapGatherOp`, by applying the
/// permutation to the yielded source indices.
static MapGatherOp foldTransposeIntoMapGather(RewriterBase &rewriter,
                                              linalg::TransposeOp transposeOp,
                                              MapGatherOp mapGatherOp) {
  assert(mapGatherOp.getSource() == transposeOp->getResult(0) &&
         "expected transposeOp to be the producer of mapGatherOp source");

  // For map_gather, we iterate over OUTPUT indices and yield SOURCE indices.
  // transpose: output[i,j,k] = input[perm(i,j,k)]
  // So: input_idx = perm^-1(output_idx)
  // Since oldSourceIndices represents the current yielded indices (which map
  // to the transpose result), we need to apply the permutation to get the
  // input indices.
  ArrayRef<int64_t> perm = transposeOp.getPermutation();
  auto indexTransformBuilder =
      [&](ValueRange oldSourceIndices) -> SmallVector<Value> {
    SmallVector<Value> indexValues(oldSourceIndices);
    return applyPermutation(indexValues, perm);
  };
  rewriter.modifyOpInPlace(mapGatherOp, [&]() {
    mapGatherOp.insertTransformationAtEnd(rewriter, indexTransformBuilder);
    mapGatherOp.getSourceMutable().assign(transposeOp.getInput());
  });
  return mapGatherOp;
}

/// Fold a tensor::ExpandShapeOp or tensor::CollapseShapeOp into a consumer
/// `mapGatherOp`, by linearizing and then delinearizing the yielded source
/// indices.
template <typename ReshapeOpTy>
static MapGatherOp foldReshapeIntoMapGather(RewriterBase &rewriter,
                                            ReshapeOpTy reshapeOp,
                                            MapGatherOp mapGatherOp) {
  assert(mapGatherOp.getSource() == reshapeOp->getResult(0) &&
         "expected reshapeOp to be the producer of mapGatherOp source");
  Location loc = reshapeOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(reshapeOp);
  SmallVector<OpFoldResult> srcDims =
      tensor::getMixedSizes(rewriter, loc, reshapeOp.getSrc());
  SmallVector<OpFoldResult> resultDims =
      tensor::getMixedSizes(rewriter, loc, reshapeOp.getResult());

  auto indexTransformBuilder =
      [&](ValueRange oldSourceIndices) -> SmallVector<Value> {
    // Linearize indices in result space, then delinearize to source space.
    auto linearizeIndexOp = affine::AffineLinearizeIndexOp::create(
        rewriter, mapGatherOp->getLoc(), oldSourceIndices, resultDims,
        /*disjoint=*/true);
    auto delinearizeIndexOp = affine::AffineDelinearizeIndexOp::create(
        rewriter, mapGatherOp->getLoc(), linearizeIndexOp.getResult(), srcDims,
        /*hasOuterBound=*/true);
    return delinearizeIndexOp->getResults();
  };
  rewriter.modifyOpInPlace(mapGatherOp, [&]() {
    mapGatherOp.insertTransformationAtEnd(rewriter, indexTransformBuilder);
    mapGatherOp.getSourceMutable().assign(reshapeOp->getOperand(0));
  });
  return mapGatherOp;
}

/// Fold a tensor::ExpandShapeOp into a consumer `mapGatherOp`.
static MapGatherOp
foldExpandShapeIntoMapGather(RewriterBase &rewriter,
                             tensor::ExpandShapeOp expandShapeOp,
                             MapGatherOp mapGatherOp) {
  return foldReshapeIntoMapGather(rewriter, expandShapeOp, mapGatherOp);
}

/// Fold a tensor::CollapseShapeOp into a consumer `mapGatherOp`.
static MapGatherOp
foldCollapseShapeIntoMapGather(RewriterBase &rewriter,
                               tensor::CollapseShapeOp collapseShapeOp,
                               MapGatherOp mapGatherOp) {
  return foldReshapeIntoMapGather(rewriter, collapseShapeOp, mapGatherOp);
}

/// Fold an `extractSliceOp` into a consumer `mapGatherOp` by adding offsets
/// to the yielded source indices.
static FailureOr<MapGatherOp>
foldExtractSliceIntoMapGather(RewriterBase &rewriter,
                              tensor::ExtractSliceOp extractSliceOp,
                              MapGatherOp mapGatherOp) {
  assert(mapGatherOp.getSource() == extractSliceOp->getResult(0) &&
         "expected extractSliceOp to be the producer of mapGatherOp source");
  if (extractSliceOp.getSourceType().getRank() !=
      extractSliceOp.getResultType().getRank()) {
    return rewriter.notifyMatchFailure(
        extractSliceOp, "rank reducing extract_slice op is not supported");
  }
  if (!areAllConstantIntValue(extractSliceOp.getMixedStrides(), 1)) {
    return rewriter.notifyMatchFailure(extractSliceOp,
                                       "non-unit strides are not supported");
  }

  // Check if this is an identity extract_slice (all offsets are zero and all
  // sizes match the source). If so, just replace the source without adding
  // any offset computation.
  SmallVector<OpFoldResult> sliceOffsets = extractSliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sliceSizes = extractSliceOp.getMixedSizes();
  Value source = extractSliceOp.getSource();
  auto isIdentitySlice = [&]() {
    if (!areAllConstantIntValue(sliceOffsets, 0)) {
      return false;
    }
    for (auto [dim, sliceSize] : llvm::enumerate(sliceSizes)) {
      ValueBoundsConstraintSet::Variable sourceDimVar(source, dim);
      FailureOr<bool> areEqual =
          ValueBoundsConstraintSet::areEqual(sliceSize, sourceDimVar);
      if (!succeeded(areEqual) || !*areEqual) {
        return false;
      }
    }
    return true;
  };

  if (isIdentitySlice()) {
    // Identity extract_slice: just replace the source without any changes.
    rewriter.modifyOpInPlace(
        mapGatherOp, [&]() { mapGatherOp.getSourceMutable().assign(source); });
    return mapGatherOp;
  }

  Location loc = mapGatherOp->getLoc();
  auto indexTransformBuilder =
      [&](ValueRange oldSourceIndices) -> SmallVector<Value> {
    SmallVector<Value> newIndices;
    for (auto [idx, offset] : llvm::zip_equal(oldSourceIndices, sliceOffsets)) {
      Value offsetValue =
          getValueOrCreateConstantIndexOp(rewriter, loc, offset);
      Value newIdx = arith::AddIOp::create(rewriter, loc, idx, offsetValue);
      newIndices.push_back(newIdx);
    }
    return newIndices;
  };
  rewriter.modifyOpInPlace(mapGatherOp, [&]() {
    mapGatherOp.insertTransformationAtEnd(rewriter, indexTransformBuilder);
    mapGatherOp.getSourceMutable().assign(extractSliceOp.getSource());
  });
  return mapGatherOp;
}

FailureOr<MapGatherOp> foldIntoMapGather(RewriterBase &rewriter, Operation *op,
                                         MapGatherOp mapGatherOp) {
  return llvm::TypeSwitch<Operation *, FailureOr<MapGatherOp>>(op)
      .Case<linalg::CopyOp>([&](linalg::CopyOp copyOp) {
        return foldIdentityLikeOpIntoMapGather(rewriter, copyOp, mapGatherOp);
      })
      .Case<linalg::TransposeOp>([&](linalg::TransposeOp transposeOp) {
        return foldTransposeIntoMapGather(rewriter, transposeOp, mapGatherOp);
      })
      .Case<tensor::ExpandShapeOp>([&](tensor::ExpandShapeOp expandOp) {
        return foldExpandShapeIntoMapGather(rewriter, expandOp, mapGatherOp);
      })
      .Case<tensor::CollapseShapeOp>([&](tensor::CollapseShapeOp collapseOp) {
        return foldCollapseShapeIntoMapGather(rewriter, collapseOp,
                                              mapGatherOp);
      })
      .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp extractSliceOp) {
        return foldExtractSliceIntoMapGather(rewriter, extractSliceOp,
                                             mapGatherOp);
      })
      .Default([](Operation *) { return failure(); });
}

// Insert identity map_gather op after the root and replace uses.
static MapGatherOp insertIdentityMapGather(RewriterBase &rewriter,
                                           OpResult root) {
  Location loc = root.getLoc();
  SetVector<OpOperand *> originalUses;
  for (OpOperand &use : root.getUses()) {
    originalUses.insert(&use);
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfterValue(root);
  Type elementType = getElementTypeOrSelf(root.getType());
  SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(rewriter, loc, root);
  Value mapGatherDest =
      tensor::EmptyOp::create(rewriter, loc, sizes, elementType);
  auto mapGatherOp =
      MapGatherOp::createIdentityMapGather(rewriter, loc, root, mapGatherDest);
  rewriter.replaceUsesWithIf(
      root, mapGatherOp.getResult(0),
      [&](OpOperand &use) { return originalUses.contains(&use); });
  LDBG() << "Created identity map_gather:\n" << mapGatherOp;
  return mapGatherOp;
}

bool isSupportedGatherRelayoutOp(Operation *op) {
  // Note: tensor::PadOp is NOT supported for map_gather. Unlike map_scatter
  // which can write padding values directly to the output buffer (obtained from
  // store_to_buffer), map_gather reads from a source buffer and produces a
  // tensor result. There is no output memref available to write padding values
  // to at this stage.
  return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
             tensor::ExtractSliceOp, linalg::CopyOp, linalg::TransposeOp>(op);
}

// This is only desirable in the dispatch scope but not in the workgroup scope.
static bool shouldDoReshapesByExpansion(
    IREE::Codegen::GatherRelayoutCombinationScope scope) {
  if (scope == IREE::Codegen::GatherRelayoutCombinationScope::Dispatch) {
    return true;
  }
  return false;
}

/// Insert identity map_gather ops after the given operation if it is a valid
/// leaf op of a relayout op chain. A relayout op chain is a sequence of
/// relayout ops (defined by `isSupportedGatherRelayoutOp`) for which the only
/// users of the ops in the chain are relayout ops, except for the leaves of the
/// chain. The leaves are simply relayout ops that have non relayout op users.
/// The `controlFn` is a callback on the leaf OpResult that provides control
/// over whether or not to insert a map_gather op.
struct InsertMapGatherOpPattern : public RewritePattern {
  InsertMapGatherOpPattern(
      MLIRContext *context,
      CombineRelayoutOpsForGatherControlFnRef controlFn = nullptr,
      PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        controlFn(controlFn) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isSupportedGatherRelayoutOp(op)) {
      return failure();
    }
    // Relayout ops with only relayout op users are not leaves.
    auto isDimOrSupportedRelayoutOp = [](Operation *user) {
      if (isa<tensor::DimOp>(user)) {
        return true;
      }
      return isSupportedGatherRelayoutOp(user);
    };
    if (llvm::all_of(op->getUsers(), isDimOrSupportedRelayoutOp)) {
      return failure();
    }
    // All relayout ops have a single result.
    OpResult leaf = op->getResult(0);
    if (controlFn && !controlFn(leaf)) {
      return failure();
    }
    (void)insertIdentityMapGather(rewriter, leaf);
    return success();
  }

private:
  CombineRelayoutOpsForGatherControlFnRef controlFn;
};

LogicalResult combineLayoutTransformationForMapGather(
    MLIRContext *ctx, FunctionOpInterface funcOp, bool doReshapeByExpansion,
    CombineRelayoutOpsForGatherControlFnRef controlFn) {
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
  auto controlPropagationFn = [](OpOperand *operand) -> bool {
    Operation *producer = operand->get().getDefiningOp();
    Operation *consumer = operand->getOwner();
    if (!isa_and_nonnull<linalg::UnPackOp>(producer)) {
      return false;
    }
    // Reshapes will not produce extra pack ops.
    if (isa<tensor::ExpandShapeOp>(consumer)) {
      return true;
    }
    // Otherwise, the consumer must be a GenericOp with all of its `outs`
    // operands coming from tensor.empty ops, and the `operand` must be the
    // sole `ins` operand of the generic op.
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
  if (failed(applyPatternsGreedily(funcOp, std::move(propagationPatterns)))) {
    return failure();
  }

  // Apply some preprocessing to convert complex layout transformation
  // ops like unpack into simpler supported ops.
  IRRewriter rewriter(ctx);
  simplifyComplexRelayoutOpsForGather(rewriter, funcOp);

  // Combine relayout operations into new map_gather ops.
  RewritePatternSet relayoutCombinationPatterns(ctx);
  relayoutCombinationPatterns.add<InsertMapGatherOpPattern>(ctx, controlFn);
  // Use populateCombineRelayoutOpPatterns without pad distribution (no pad
  // support for gather). This adds both scatter and gather folding patterns,
  // but since we only insert map_gather ops, only those will be matched.
  populateCombineRelayoutOpPatterns(relayoutCombinationPatterns,
                                    /*padDistributionConfigFn=*/nullptr);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(
      relayoutCombinationPatterns);
  if (failed(applyPatternsGreedily(funcOp,
                                   std::move(relayoutCombinationPatterns)))) {
    return failure();
  }

  // Clean up any identity map_gather ops after combining.
  funcOp->walk([&](MapGatherOp mapGatherOp) {
    if (mapGatherOp.isIdentity()) {
      rewriter.replaceOp(mapGatherOp, mapGatherOp.getSource());
    }
  });
  return success();
}

CombineRelayoutOpsForGatherControlFn getCombineRelayoutOpsForGatherControlFn(
    IREE::Codegen::GatherRelayoutCombinationScope scope) {
  CombineRelayoutOpsForGatherControlFn controlFn;
  switch (scope) {
  // Control function for Dispatch scope. Checks that producer chain starts
  // from LoadFromBufferOp.
  case IREE::Codegen::GatherRelayoutCombinationScope::Dispatch:
    controlFn = [](OpResult leaf) {
      // Walk back to verify root is LoadFromBufferOp.
      Operation *current = leaf.getDefiningOp();
      while (current) {
        Value input = current->getOperand(0);
        Operation *producer = input.getDefiningOp();
        if (!producer) {
          return false;
        }
        if (isa<IREE::Codegen::LoadFromBufferOp>(producer)) {
          return true;
        }
        if (!isSupportedGatherRelayoutOp(producer)) {
          return false;
        }
        current = producer;
      }
      return false;
    };
    break;
  }
  return controlFn;
}

namespace {

struct CombineLayoutTransformationForMapGatherPass final
    : impl::CombineLayoutTransformationForMapGatherPassBase<
          CombineLayoutTransformationForMapGatherPass> {
  using Base::Base;

  void runOnOperation() override {
    CombineRelayoutOpsForGatherControlFn controlFn =
        getCombineRelayoutOpsForGatherControlFn(this->scope);
    bool doReshapesByExpansion = shouldDoReshapesByExpansion(this->scope);
    if (failed(combineLayoutTransformationForMapGather(
            &getContext(), getOperation(), doReshapesByExpansion, controlFn))) {
      return signalPassFailure();
    }

    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();
    {
      RewritePatternSet patterns(context);
      scf::ForallOp::getCanonicalizationPatterns(patterns, context);
      tensor::populateFoldTensorEmptyPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
