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

/// Folds `consumerOp` into the producer `mapGatherOp` using `rewriter`.
/// The `indexTransformBuilder` lambda defines how output indices of the new
/// map_gather map to input indices of the original map_gather. Optionally,
/// `newFillValue` can be provided to override the fill value (e.g., for pad).
///
/// This handles the shared boilerplate:
/// 1. Create new dest tensor from consumer result shape
/// 2. Clone map_gather with new dest and region
/// 3. Apply index transformation via insertTransformationAtStart
/// 4. Optionally update fill value
/// 5. Replace consumer with new map_gather result
template <typename ConsumerOpTy>
static FailureOr<MapGatherOp> foldConsumerIntoMapGatherImpl(
    RewriterBase &rewriter, ConsumerOpTy consumerOp, MapGatherOp mapGatherOp,
    function_ref<SmallVector<Value>(ArrayRef<BlockArgument>)>
        indexTransformBuilder,
    Value newFillValue = nullptr) {
  Location loc = consumerOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(consumerOp);

  // Create new dest tensor matching consumer output shape.
  Value consumerResult = consumerOp->getResult(0);
  Type elementType = getElementTypeOrSelf(consumerResult.getType());
  SmallVector<OpFoldResult> newSizes =
      tensor::getMixedSizes(rewriter, loc, consumerResult);
  Value newDest = tensor::EmptyOp::create(rewriter, loc, newSizes, elementType);

  // Clone the map_gather with new dest.
  auto newMapGather = MapGatherOp::create(rewriter, loc, newDest.getType(),
                                          mapGatherOp.getSource(), newDest);
  rewriter.cloneRegionBefore(mapGatherOp.getTransformationRegion(),
                             newMapGather.getTransformationRegion(),
                             newMapGather.getTransformationRegion().begin());

  // Prepend the index transformation.
  int64_t newRank = cast<ShapedType>(consumerResult.getType()).getRank();
  newMapGather.insertTransformationAtStart(rewriter, indexTransformBuilder,
                                           newRank);

  // Optionally update fill value (for pad ops).
  if (newFillValue) {
    Block &transformBody = newMapGather.getTransformationRegion().front();
    auto yieldOp =
        cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp.setOperand(yieldOp.getNumOperands() - 1, newFillValue);
    });
  }

  rewriter.replaceOp(consumerOp, newMapGather.getResult(0));
  LDBG() << "Folded consumer " << consumerOp->getName().getStringRef()
         << " into map_gather:\n"
         << newMapGather;
  return newMapGather;
}

/// Fold a consumer `copyOp` into a producer `mapGatherOp`.
/// Copy doesn't change indices, so we just replace uses.
static FailureOr<MapGatherOp> foldCopyIntoMapGather(RewriterBase &rewriter,
                                                    linalg::CopyOp copyOp,
                                                    MapGatherOp mapGatherOp) {
  assert(copyOp.getInputs()[0] == mapGatherOp.getResult(0) &&
         "expected mapGatherOp to be the producer of copyOp input");
  rewriter.replaceOp(copyOp, mapGatherOp.getResult(0));
  LDBG() << "Folded consumer copy into map_gather (identity fold)";
  return mapGatherOp;
}

/// Fold a consumer `transposeOp` into a producer `mapGatherOp`.
/// For consumer folding, we apply the INVERSE permutation.
/// Example: perm=[1,2,0] means output[i,j,k] = input[k,i,j] (inverse=[2,0,1]).
static FailureOr<MapGatherOp>
foldTransposeIntoMapGather(RewriterBase &rewriter,
                           linalg::TransposeOp transposeOp,
                           MapGatherOp mapGatherOp) {
  assert(transposeOp.getInput() == mapGatherOp.getResult(0) &&
         "expected mapGatherOp to be the producer of transposeOp input");

  ArrayRef<int64_t> perm = transposeOp.getPermutation();
  SmallVector<int64_t> inversePerm = invertPermutationVector(perm);

  auto indexTransformBuilder =
      [&](ArrayRef<BlockArgument> newOutputIndices) -> SmallVector<Value> {
    SmallVector<Value> indexValues(newOutputIndices.begin(),
                                   newOutputIndices.end());
    return applyPermutation(indexValues, inversePerm);
  };

  return foldConsumerIntoMapGatherImpl(rewriter, transposeOp, mapGatherOp,
                                       indexTransformBuilder);
}

/// Fold a consumer reshape op (expand_shape or collapse_shape) into a producer
/// `mapGatherOp`. Index transformation: linearize in result space, delinearize
/// in source space.
template <typename ReshapeOpTy>
static FailureOr<MapGatherOp>
foldReshapeIntoMapGather(RewriterBase &rewriter, ReshapeOpTy reshapeOp,
                         MapGatherOp mapGatherOp) {
  assert(reshapeOp.getSrc() == mapGatherOp.getResult(0) &&
         "expected mapGatherOp to be the producer of reshapeOp input");

  Location loc = reshapeOp->getLoc();
  SmallVector<OpFoldResult> srcDims =
      tensor::getMixedSizes(rewriter, loc, reshapeOp.getSrc());
  SmallVector<OpFoldResult> resultDims =
      tensor::getMixedSizes(rewriter, loc, reshapeOp.getResult());

  auto indexTransformBuilder =
      [&](ArrayRef<BlockArgument> newOutputIndices) -> SmallVector<Value> {
    SmallVector<Value> indices(newOutputIndices.begin(),
                               newOutputIndices.end());
    auto linearizeIndexOp = affine::AffineLinearizeIndexOp::create(
        rewriter, loc, indices, resultDims, /*disjoint=*/true);
    auto delinearizeIndexOp = affine::AffineDelinearizeIndexOp::create(
        rewriter, loc, linearizeIndexOp.getResult(), srcDims,
        /*hasOuterBound=*/true);
    return delinearizeIndexOp->getResults();
  };

  return foldConsumerIntoMapGatherImpl(rewriter, reshapeOp, mapGatherOp,
                                       indexTransformBuilder);
}

/// Fold a consumer tensor::ExpandShapeOp into a producer `mapGatherOp`.
static FailureOr<MapGatherOp>
foldExpandShapeIntoMapGather(RewriterBase &rewriter,
                             tensor::ExpandShapeOp expandShapeOp,
                             MapGatherOp mapGatherOp) {
  return foldReshapeIntoMapGather(rewriter, expandShapeOp, mapGatherOp);
}

/// Fold a consumer tensor::CollapseShapeOp into a producer `mapGatherOp`.
static FailureOr<MapGatherOp>
foldCollapseShapeIntoMapGather(RewriterBase &rewriter,
                               tensor::CollapseShapeOp collapseShapeOp,
                               MapGatherOp mapGatherOp) {
  return foldReshapeIntoMapGather(rewriter, collapseShapeOp, mapGatherOp);
}

/// Fold a consumer `extractSliceOp` into a producer `mapGatherOp`.
/// Index transformation: original_idx = offset + new_idx * stride
static FailureOr<MapGatherOp>
foldExtractSliceIntoMapGather(RewriterBase &rewriter,
                              tensor::ExtractSliceOp extractSliceOp,
                              MapGatherOp mapGatherOp) {
  assert(extractSliceOp.getSource() == mapGatherOp.getResult(0) &&
         "expected mapGatherOp to be the producer of extractSliceOp input");

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
      Value scaledIdx = arith::MulIOp::create(rewriter, loc, idx, strideValue);
      Value originalIdx =
          arith::AddIOp::create(rewriter, loc, offsetValue, scaledIdx);
      originalIndices.push_back(originalIdx);
    }
    return originalIndices;
  };

  return foldConsumerIntoMapGatherImpl(rewriter, extractSliceOp, mapGatherOp,
                                       indexTransformBuilder);
}

/// Fold a consumer `padOp` into a producer `mapGatherOp`.
/// Index transformation: source_idx = new_idx - low_pad
/// Fill value is set to the pad value.
static FailureOr<MapGatherOp> foldPadIntoMapGather(RewriterBase &rewriter,
                                                   tensor::PadOp padOp,
                                                   MapGatherOp mapGatherOp) {
  assert(padOp.getSource() == mapGatherOp.getResult(0) &&
         "expected mapGatherOp to be the producer of padOp input");

  // Only support constant pad values for now.
  Value padValue = padOp.getConstantPaddingValue();
  if (!padValue) {
    return rewriter.notifyMatchFailure(
        padOp, "non-constant padding value is not supported");
  }

  Location loc = padOp->getLoc();
  SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();

  auto indexTransformBuilder =
      [&](ArrayRef<BlockArgument> newOutputIndices) -> SmallVector<Value> {
    SmallVector<Value> sourceIndices;
    for (auto [idx, low] : llvm::zip_equal(newOutputIndices, lowPad)) {
      Value lowValue = getValueOrCreateConstantIndexOp(rewriter, loc, low);
      // source_idx = idx - low_pad
      Value sourceIdx = arith::SubIOp::create(rewriter, loc, idx, lowValue);
      sourceIndices.push_back(sourceIdx);
    }
    return sourceIndices;
  };

  return foldConsumerIntoMapGatherImpl(rewriter, padOp, mapGatherOp,
                                       indexTransformBuilder, padValue);
}

/// Fold a consumer relayout op into a producer map_gather.
FailureOr<MapGatherOp> foldIntoMapGather(RewriterBase &rewriter, Operation *op,
                                         MapGatherOp mapGatherOp) {
  return llvm::TypeSwitch<Operation *, FailureOr<MapGatherOp>>(op)
      .Case<linalg::CopyOp>([&](linalg::CopyOp copyOp) {
        return foldCopyIntoMapGather(rewriter, copyOp, mapGatherOp);
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

/// Pattern to fold consumer relayout ops into a producer map_gather.
struct FoldConsumerRelayoutIntoMapGatherPattern
    : public OpRewritePattern<IREE::LinalgExt::MapGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::MapGatherOp mapGatherOp,
                                PatternRewriter &rewriter) const override {
    // Find a consumer relayout op.
    Operation *consumerOp = nullptr;
    for (Operation *user : mapGatherOp->getUsers()) {
      if (isSupportedSingleInputRelayoutOp(user)) {
        consumerOp = user;
        break;
      }
    }
    if (!consumerOp) {
      return failure();
    }
    if (failed(foldIntoMapGather(rewriter, consumerOp, mapGatherOp))) {
      return failure();
    }
    return success();
  }
};

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

/// Insert identity map_gather op after a LoadFromBufferOp if it has relayout
/// op consumers. The identity map_gather can then be used to fold consumer
/// relayout ops into it iteratively.
struct InsertMapGatherOpPattern
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
    // Check that the load doesn't already have a map_gather user (avoid
    // infinite loop).
    bool hasMapGatherUser =
        llvm::any_of(loadOp->getUsers(), [](Operation *user) {
          return isa<IREE::LinalgExt::MapGatherOp>(user);
        });
    if (hasMapGatherUser) {
      return failure();
    }
    (void)insertIdentityMapGather(rewriter, loadOp->getResult(0));
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

    // Insert identity map_gather ops after load_from_buffer ops and fold
    // consumer relayout ops into them.
    RewritePatternSet patterns(context);
    patterns.add<InsertMapGatherOpPattern>(context);
    patterns.add<FoldConsumerRelayoutIntoMapGatherPattern>(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
