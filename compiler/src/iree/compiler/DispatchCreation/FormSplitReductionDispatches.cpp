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

struct FormSplitReductionDispatchesPass final
    : public impl::FormSplitReductionDispatchesPassBase<
          FormSplitReductionDispatchesPass> {
  using Base::Base;
  void runOnOperation() override;
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
                        OpFoldResult splitSize) {
  IRRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  scf::SCFTilingOptions options;
  options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  options.setReductionTilingStrategy(
      ReductionTilingStrategy::PartialReductionOuterParallel);

  // Set tile sizes.
  SmallVector<OpFoldResult> tileSizes;
  auto zeroAttr = rewriter.getIndexAttr(0);
  for (utils::IteratorType iteratorType : op.getLoopIteratorTypes()) {
    if (iteratorType == utils::IteratorType::parallel) {
      tileSizes.push_back(zeroAttr);
    } else {
      tileSizes.push_back(splitSize);
    }
  }
  options.setTileSizes(tileSizes);
  SmallVector<unsigned> reductionDims = getReductionDims(op);
  if (reductionDims.size() != 1) {
    return op.emitError("op must only have one reduction dim");
  }
  options.setReductionDims(reductionDims);
  options.setMapping(
      {IREE::LinalgExt::SplitReductionMappingAttr::get(rewriter.getContext())});

  // Tile the operation and fuse with producers.
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  // Only fuse along the dest operand.
  scf::SCFTileAndFuseOptions::ControlFnTy fusionControlFn =
      [](tensor::ExtractSliceOp, OpResult, bool isDestArg)
      -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    if (isDestArg) {
      return scf::SCFTileAndFuseOptions::ControlFnResult{false};
    }
    return std::nullopt;
  };
  tileAndFuseOptions.setFusionControlFn(fusionControlFn);
  tileAndFuseOptions.setTilingOptions(std::move(options));

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

void FormSplitReductionDispatchesPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  SmallVector<TilingInterface> reductionOps;
  funcOp.walk([&](TilingInterface tilingOp) {
    // TODO: implement better selection.
    if (llvm::is_contained(tilingOp.getLoopIteratorTypes(),
                           utils::IteratorType::reduction)) {
      reductionOps.push_back(tilingOp);
    }
  });

  SmallVector<IREE::Flow::DispatchRegionOp> splitReductionDispatches;
  for (TilingInterface op : reductionOps) {
    FailureOr<IREE::Flow::DispatchRegionOp> formedDispatch =
        tileOpAndWrapInDispatch(rewriter, op,
                                rewriter.getIndexAttr(splitSize.getValue()));
    if (failed(formedDispatch)) {
      op->emitOpError("failed to form split reduction dispatch");
      return signalPassFailure();
    }
    splitReductionDispatches.push_back(formedDispatch.value());
  }

  // Run some canonicalization patterns within dispatches.
  RewritePatternSet patterns(context);
  linalg::populateSwapExtractSliceWithFillPatterns(patterns);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  GreedyRewriteConfig config;
  config.setMaxIterations(GreedyRewriteConfig::kNoLimit).enableFolding(true);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
    funcOp.emitOpError("failed to apply tiling canonicalization patterns");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
