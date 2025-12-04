// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/DispatchCreation/Passes.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-form-split-reduction-dispatches"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FORMSPLITREDUCTIONDISPATCHESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// WAR (Workaround): Pattern to fold extract_slice(broadcast) into the
/// broadcast input when the extract_slice is rank-reducing and only extracts
/// along the broadcasted dimensions (with size 1), leaving all non-broadcasted
/// dimensions intact. This is valid because the broadcast only duplicates data
/// along the broadcasted dimension, and extracting a single slice from that
/// dimension gives back the original input.
///
/// TODO(Bangtian): Investigate and fix the root cause of the race condition in
/// the lower levels of the stack to eliminate the need for this fold pattern
/// workaround.
///
/// This fold masks an underlying race condition in the lower levels of the
/// stack. When the broadcast's outs operand comes from extracting
/// a slice of a shared_outs tensor in scf.forall, the broadcast operation
/// writes to that extracted slice, which aliases back to the shared tensor.
/// Since the extracted slice spans parallel dimensions (e.g., [4, 1, 1] where
/// dim 0 is the batch dimension), multiple parallel workgroups would all write
/// to the same aliased memory location, creating a race condition. By folding
/// to the broadcast input (which doesn't alias the shared tensor), this pattern
/// works around the race, allowing subsequent passes to properly tile and
/// distribute the initialization per workgroup. However, the root cause of the
/// race condition exists deeper in the stack and is not addressed by this fold.
///
/// Example:
///   %broadcast = linalg.broadcast ins(%in : tensor<4x1xf16>)
///                                 outs(%out : tensor<4x1x1xf16>) dimensions =
///                                 [2]
///   %extract = tensor.extract_slice %broadcast[0, 0, 0] [4, 1, 1] [1, 1, 1]
///              : tensor<4x1x1xf16> to tensor<4x1xf16>
///   // Extracts all of dims 0,1 (sizes 4,1) and reduces dim 2 (size 1).
/// ->
///   %extract is replaced by %in (tensor<4x1xf16>)
struct FoldExtractSliceOfBroadcast final
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto broadcastOp =
        extractOp.getSource().getDefiningOp<linalg::BroadcastOp>();
    if (!broadcastOp) {
      return rewriter.notifyMatchFailure(
          extractOp, "source is not a linalg.broadcast operation");
    }

    if (!extractOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(extractOp,
                                         "extract_slice has non-unit stride");
    }

    auto inputType =
        dyn_cast<RankedTensorType>(broadcastOp.getInput().getType());
    auto broadcastOutputType =
        dyn_cast<RankedTensorType>(broadcastOp.getInit().getType());
    auto extractResultType =
        dyn_cast<RankedTensorType>(extractOp.getResult().getType());
    if (!inputType || !broadcastOutputType || !extractResultType) {
      return rewriter.notifyMatchFailure(
          extractOp, "operand or result types are not RankedTensorType");
    }

    // Extract result type must match broadcast input type.
    if (inputType != extractResultType) {
      return rewriter.notifyMatchFailure(
          extractOp, "extract result type does not match broadcast input type");
    }

    // Verify that we're extracting from offset [0, 0, ..., 0] with the same
    // shape as the input (essentially undoing the broadcast).
    SmallVector<OpFoldResult> offsets = extractOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = extractOp.getMixedSizes();
    if (llvm::any_of(offsets, [](OpFoldResult offset) {
          return !isConstantIntValue(offset, 0);
        })) {
      return rewriter.notifyMatchFailure(
          extractOp, "extract_slice offsets are not all zeros");
    }

    // Sizes should match input dimensions (accounting for broadcast dims).
    ArrayRef<int64_t> broadcastDims = broadcastOp.getDimensions();
    int64_t broadcastRank = broadcastOutputType.getRank();

    // Verify that for broadcast dimensions, the size is 1.
    if (llvm::any_of(broadcastDims, [&](int64_t broadcastDim) {
          return !isOneInteger(sizes[broadcastDim]);
        })) {
      return rewriter.notifyMatchFailure(
          extractOp, "broadcast dimensions do not all have size 1");
    }

    // Collect the indices of dimensions in the broadcast output that were not
    // broadcasted (i.e., dimensions that existed in the original input).
    auto nonBroadcastDims = llvm::to_vector(llvm::make_filter_range(
        llvm::seq<int64_t>(0, broadcastRank),
        [&](int64_t i) { return !llvm::is_contained(broadcastDims, i); }));

    // Verify that for non-broadcast dimensions, sizes match input shape.
    if (llvm::any_of(llvm::enumerate(nonBroadcastDims), [&](auto pair) {
          auto [idx, inputDim] = pair;
          int64_t inputDimSize = inputType.getDimSize(idx);
          return !isConstantIntValue(sizes[inputDim], inputDimSize);
        })) {
      return rewriter.notifyMatchFailure(
          extractOp, "non-broadcast dimension sizes do not match input shape");
    }

    // Only fold if broadcast has a single use to avoid leaving dead broadcast
    // ops or breaking other uses that expect the broadcast result.
    if (!broadcastOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(
          extractOp, "broadcast operation has multiple uses");
    }

    rewriter.replaceOp(extractOp, broadcastOp.getInput());
    return success();
  }
};

struct FormSplitReductionDispatchesPass final
    : public impl::FormSplitReductionDispatchesPassBase<
          FormSplitReductionDispatchesPass> {
  using Base::Base;
  void runOnOperation() override;

private:
  std::optional<SmallVector<OpFoldResult>>
  getUserSpecifiedTileSize(PartialReductionOpInterface op) const;
};
} // namespace

static SmallVector<unsigned> getReductionDims(TilingInterface op) {
  SmallVector<unsigned> dims;
  for (auto [i, loopType] : llvm::enumerate(op.getLoopIteratorTypes())) {
    if (loopType == utils::IteratorType::reduction) {
      dims.push_back(i);
    }
  }
  return dims;
}

static FailureOr<IREE::Flow::DispatchRegionOp>
tileOpAndWrapInDispatch(RewriterBase &rewriter, TilingInterface op,
                        ArrayRef<OpFoldResult> splitSize, bool fusePad) {
  IRRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  scf::SCFTilingOptions options;
  options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  options.setReductionTilingStrategy(
      ReductionTilingStrategy::PartialReductionOuterParallel);

  // Set tile sizes.
  SmallVector<OpFoldResult> tileSizes;
  auto zeroAttr = rewriter.getIndexAttr(0);
  int splitSizeIndex = 0;
  for (utils::IteratorType iteratorType : op.getLoopIteratorTypes()) {
    if (iteratorType == utils::IteratorType::parallel) {
      tileSizes.push_back(zeroAttr);
    } else {
      tileSizes.push_back(splitSize[splitSizeIndex++]);
    }
  }
  options.setTileSizes(tileSizes);
  SmallVector<unsigned> reductionDims = getReductionDims(op);
  auto mapping = llvm::map_to_vector(
      llvm::seq<int64_t>(0, reductionDims.size()),
      [&](int64_t index) -> Attribute {
        return IREE::LinalgExt::SplitReductionMappingAttr::get(
            rewriter.getContext(), reductionDims.size() - 1 - index);
      });
  options.setReductionDims(reductionDims);
  options.setMapping(mapping);

  // Tile the operation and fuse with producers.
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  // Only fuse along the dest operand.
  scf::SCFTileAndFuseOptions::ControlFnTy fusionControlFn =
      [](tensor::ExtractSliceOp, OpResult result, bool isDestArg)
      -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    if (isDestArg) {
      return scf::SCFTileAndFuseOptions::ControlFnResult{false};
    }
    return std::nullopt;
  };
  tileAndFuseOptions.setFusionControlFn(fusionControlFn);
  tileAndFuseOptions.setTilingOptions(std::move(options));

  if (fusePad) {
    MLIRContext *context = rewriter.getContext();
    RewritePatternSet cleanupPatterns(context);
    // When fusing pads we do not want to generate zeroSliceGuards.
    cleanupPatterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
        context,
        [](tensor::ExtractSliceOp) { return /*zeroSliceGuard=*/false; });
    tileAndFuseOptions.cleanupPatterns =
        FrozenRewritePatternSet(std::move(cleanupPatterns));
  }

  FailureOr<scf::SCFTileAndFuseResult> result =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, op,
                                                tileAndFuseOptions);
  if (failed(result)) {
    return op.emitOpError("failed to tile using scf.forall");
  }
  for (auto [origValue, replacement] : result->replacements) {
    rewriter.replaceAllUsesWith(origValue, replacement);
  }

  // Didn't tile.
  if (result->loops.size() == 0) {
    return success();
  }
  assert(result->loops.size() == 1 &&
         "expected to get a single loop after tiling");

  // Wrap loop in `flow.dispatch.region`.
  LoopLikeOpInterface loop = result->loops[0];
  FailureOr<IREE::Flow::DispatchRegionOp> maybeRegionOp =
      IREE::Flow::wrapOpInDispatchRegion(rewriter, loop);
  if (failed(maybeRegionOp)) {
    return loop.emitOpError("failed to wrap in dispatch region");
  }
  return maybeRegionOp.value();
}

std::optional<SmallVector<OpFoldResult>>
FormSplitReductionDispatchesPass::getUserSpecifiedTileSize(
    PartialReductionOpInterface op) const {
  {
    // First preference given to attribute set on the op.
    std::optional<SmallVector<int64_t>> attributeTileSize =
        IREE::LinalgExt::getSplitReductionSizes(op);
    if (attributeTileSize) {
      MLIRContext *context = op->getContext();
      return getAsIndexOpFoldResult(context, attributeTileSize.value());
    }
  }

  // Use the pass option as the next lever. This is mostly used for testing.
  if (!splitSize.empty()) {
    unsigned numReduction = llvm::count_if(
        op.getLoopIteratorTypes(), [](utils::IteratorType iteratorType) {
          return iteratorType == utils::IteratorType::reduction;
        });
    if (numReduction == 0) {
      return std::nullopt;
    }
    SmallVector<int64_t> tileSizes(numReduction, 0);
    for (auto [index, tileSize] : llvm::enumerate(llvm::reverse(splitSize))) {
      tileSizes[numReduction - 1 - index] = tileSize;
    }
    MLIRContext *context = op->getContext();
    return getAsIndexOpFoldResult(context, tileSizes);
  }

  // Default.
  return std::nullopt;
}

void FormSplitReductionDispatchesPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  SmallVector<std::pair<PartialReductionOpInterface, SmallVector<OpFoldResult>>>
      reductionOps;
  funcOp.walk([&](PartialReductionOpInterface tilingOp) {
    std::optional<SmallVector<OpFoldResult>> tileSizes =
        getUserSpecifiedTileSize(tilingOp);
    if (!tileSizes) {
      return;
    }
    reductionOps.emplace_back(tilingOp, std::move(tileSizes.value()));
  });

  if (reductionOps.empty()) {
    // Nothing to do.
    return;
  }

  for (auto [op, tileSizes] : reductionOps) {
    FailureOr<IREE::Flow::DispatchRegionOp> formedDispatch =
        tileOpAndWrapInDispatch(rewriter, op, tileSizes, enableFusePad);
    if (failed(formedDispatch)) {
      op->emitOpError("failed to form split reduction dispatch");
      return signalPassFailure();
    }
  }

  // Run some canonicalization patterns within dispatches.
  RewritePatternSet patterns(context);
  linalg::populateSwapExtractSliceWithFillPatterns(patterns);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  patterns.add<FoldExtractSliceOfBroadcast>(context);
  GreedyRewriteConfig config;
  config.setMaxIterations(GreedyRewriteConfig::kNoLimit).enableFolding(true);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
    funcOp.emitOpError("failed to apply tiling canonicalization patterns");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
