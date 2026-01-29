// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CombineLayoutTransformation.h"
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

/// Fold a `padOp` into a consumer `mapGatherOp` by adjusting the index
/// transformation to subtract low padding offsets and updating the padding
/// value.
///
/// For a tensor.pad:
///   padded[i, j, k] = source[i - lowPad[0], j - lowPad[1], k - lowPad[2]]
///                     if indices are in bounds, else paddingValue
///
/// The map_gather's built-in bounds checking will automatically use the
/// padding value when the computed source indices are out of bounds.
static FailureOr<MapGatherOp> foldPadIntoMapGather(RewriterBase &rewriter,
                                                   tensor::PadOp padOp,
                                                   MapGatherOp mapGatherOp) {
  assert(mapGatherOp.getSource() == padOp->getResult(0) &&
         "expected padOp to be the producer of mapGatherOp source");

  // We only support constant padding values for map_gather folding.
  Value paddingValue = padOp.getConstantPaddingValue();
  if (!paddingValue) {
    return rewriter.notifyMatchFailure(
        padOp, "non-constant padding value is not supported");
  }

  // Get the low padding offsets.
  SmallVector<OpFoldResult> lowPadding = padOp.getMixedLowPad();
  Location loc = mapGatherOp->getLoc();

  // Build the index transformation: subtract low padding from each index.
  auto indexTransformBuilder =
      [&](ValueRange oldSourceIndices) -> SmallVector<Value> {
    SmallVector<Value> newIndices;
    for (auto [idx, lowPad] : llvm::zip_equal(oldSourceIndices, lowPadding)) {
      Value lowPadValue =
          getValueOrCreateConstantIndexOp(rewriter, loc, lowPad);
      Value newIdx = arith::SubIOp::create(rewriter, loc, idx, lowPadValue);
      newIndices.push_back(newIdx);
    }
    return newIndices;
  };

  // Update the map_gather: apply the index transformation and update padding.
  rewriter.modifyOpInPlace(mapGatherOp, [&]() {
    mapGatherOp.insertTransformationAtEnd(rewriter, indexTransformBuilder);
    mapGatherOp.getSourceMutable().assign(padOp.getSource());

    // Update the padding value in the yield op.
    Block &transformBody = mapGatherOp.getTransformationRegion().front();
    auto yieldOp =
        cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
    // The last operand is the padding value.
    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp->setOperand(yieldOp->getNumOperands() - 1, paddingValue);
    });
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
      .Case<tensor::PadOp>([&](tensor::PadOp padOp) {
        return foldPadIntoMapGather(rewriter, padOp, mapGatherOp);
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

/// Check if a relayout op chain starts from a LoadFromBufferOp. This is used
/// to determine if the chain should be combined into a map_gather op.
static bool startsFromLoadFromBuffer(OpResult leaf) {
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
    if (!isSupportedRelayoutOp(producer)) {
      return false;
    }
    current = producer;
  }
  return false;
}

/// Insert identity map_gather ops after the given operation if it is a valid
/// leaf op of a relayout op chain. A relayout op chain is a sequence of
/// relayout ops (defined by `isSupportedRelayoutOp`) for which the only
/// users of the ops in the chain are relayout ops, except for the leaves of the
/// chain. The leaves are simply relayout ops that have non relayout op users.
struct InsertMapGatherOpPattern : public RewritePattern {
  InsertMapGatherOpPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isSupportedRelayoutOp(op)) {
      return failure();
    }
    // Relayout ops with only relayout op users are not leaves.
    auto isDimOrSupportedRelayoutOp = [](Operation *user) {
      return (isa<tensor::DimOp>(user) || isSupportedRelayoutOp(user));
    };
    if (llvm::all_of(op->getUsers(), isDimOrSupportedRelayoutOp)) {
      return failure();
    }
    // All relayout ops have a single result.
    OpResult leaf = op->getResult(0);
    // Only combine chains that start from LoadFromBufferOp.
    if (!startsFromLoadFromBuffer(leaf)) {
      return failure();
    }
    (void)insertIdentityMapGather(rewriter, leaf);
    return success();
  }
};

namespace {

struct CombineLayoutTransformationForMapGatherPass final
    : impl::CombineLayoutTransformationForMapGatherPassBase<
          CombineLayoutTransformationForMapGatherPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    // Apply some preprocessing to convert complex layout transformation
    // ops like unpack into simpler supported ops.
    IRRewriter rewriter(context);
    simplifyComplexRelayoutOps(rewriter, funcOp);

    // Combine relayout operations into new map_gather ops.
    RewritePatternSet patterns(context);
    patterns.add<InsertMapGatherOpPattern>(context);
    populateCombineRelayoutOpPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
