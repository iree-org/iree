// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CPUTILEANDDISTRIBUTETOWORKGROUPSPASS
#include "iree/compiler/Codegen/Common/CPU/Passes.h.inc"

namespace {
struct CPUTileAndDistributeToWorkgroupsPass final
    : public impl::CPUTileAndDistributeToWorkgroupsPassBase<
          CPUTileAndDistributeToWorkgroupsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void CPUTileAndDistributeToWorkgroupsPass::runOnOperation() {
  auto funcOp = getOperation();
  auto *context = &getContext();
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  IRRewriter rewriter(context);
  FailureOr<TilingInfo> tilingInfo =
      getTiledAndDistributionInfo(rewriter, getCPURootOperation(computeOps));
  if (failed(tilingInfo)) {
    return signalPassFailure();
  }
  auto tilableOp = dyn_cast_or_null<TilingInterface>(tilingInfo->tilableOp);
  if (!tilableOp) {
    // Did not find a tileable op. So do nothing.
    return;
  }
  mlir::DominanceInfo dominanceInfo(tilableOp);
  llvm::SmallDenseSet<Operation *> tiledAndFusedOps;
  collectTiledAndFusedOps(tilableOp, tiledAndFusedOps);
  bool useWARForConsumerFusionSSAViolation =
      warForConsumerFusionSSAViolation(tilableOp, tiledAndFusedOps);

  llvm::DenseSet<Operation *> yieldReplacementsFor;
  for (auto op : tiledAndFusedOps) {
    // If tiledAndFused ops doesn't contain the user; add an replacement
    // for that.
    if (llvm::any_of(op->getUsers(), [&](Operation *user) {
          return dominanceInfo.properlyDominates(tilableOp, user) &&
                 !tiledAndFusedOps.contains(user);
        })) {
      yieldReplacementsFor.insert(op);
    }
  }
  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tilingInfo->tileSizes);
  tilingOptions.setInterchange(tilingInfo->interchange);
  tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  SmallVector<Attribute> deviceMappingAttribute =
      getDistributionMapping(context, tilingInfo->tileSizes);
  if (failed(IREE::Codegen::WorkgroupMappingAttr::verifyAttrList(
          context, funcOp.getLoc(), deviceMappingAttribute))) {
    return signalPassFailure();
  }
  tilingOptions.setMapping(deviceMappingAttribute);

  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);
  RewritePatternSet cleanupPatterns(context);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(cleanupPatterns, context);
  tensor::DimOp::getCanonicalizationPatterns(cleanupPatterns, context);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(cleanupPatterns);
  // When fusing pads we do not want to generate zeroSliceGuards when doing
  // workgroup tiling.
  cleanupPatterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
      context, [](tensor::ExtractSliceOp) { return /*zeroSliceGuard=*/false; });
  tileAndFuseOptions.cleanupPatterns =
      FrozenRewritePatternSet(std::move(cleanupPatterns));

  // The control function that determines whether a tiled producer should yield
  // its replacement.
  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
          bool isDestinationOperand)
      -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    Operation *owner = originalProducer.getOwner();
    if (isa<tensor::PadOp>(owner)) {
      return std::nullopt;
    }
    bool yieldProducerReplacement = yieldReplacementsFor.contains(owner);
    return scf::SCFTileAndFuseOptions::ControlFnResult{
        yieldProducerReplacement};
    return std::nullopt;
  };
  tileAndFuseOptions.setFusionControlFn(controlFn);
  rewriter.setInsertionPoint(tilableOp);

  // If the `tilableOp` is a `memref` op, then just tile the operation.
  SmallVector<LoopLikeOpInterface> tilingLoops;
  if (tilableOp->getNumResults() == 0) {
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilableOp, tilingOptions);
    if (failed(tilingResult)) {
      funcOp.emitOpError("tiling failed");
      return signalPassFailure();
    }
    rewriter.eraseOp(tilableOp);
    std::swap(tilingResult->loops, tilingLoops);
  } else {
    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilableOp,
                                                  tileAndFuseOptions);
    if (failed(tileAndFuseResult)) {
      funcOp.emitOpError("tile and fuse greedily failed");
      return signalPassFailure();
    }
    for (auto [origValue, replacement] : tileAndFuseResult->replacements) {
      rewriter.replaceAllUsesWith(origValue, replacement);
    }
    std::swap(tileAndFuseResult->loops, tilingLoops);
    Operation *rootTiledOp = tileAndFuseResult->tiledAndFusedOps.front();
    FailureOr<std::queue<Operation *>> newFusionOpportunities =
        fuseConsumersIntoLoops(rewriter, rootTiledOp, tilingLoops,
                               useWARForConsumerFusionSSAViolation);
    if (failed(newFusionOpportunities)) {
      rootTiledOp->emitOpError("failed to fuse consumers");
      return signalPassFailure();
    }

    // Because we restrict to at most a single tilable consumer for yielding
    // a replacement, no new fusion opportunities will yield a replacement,
    // meaning there is no need to run consumer fusion again afterwards.
    // TODO: run producer and consumer fusion in one worklist.
    fuseProducersOfSlices(rewriter, *newFusionOpportunities, tileAndFuseOptions,
                          tilingLoops);
  }
  if (!tilingLoops.empty()) {
    if (tilingLoops.size() != 1 || !isa<scf::ForallOp>(tilingLoops[0])) {
      funcOp.emitOpError(
          "expected tiling to produce a single `scf.forall` loop");
      return signalPassFailure();
    }

    auto forallOp =
        dropUnitDistributedDims(rewriter, cast<scf::ForallOp>(tilingLoops[0]));
    if (failed(forallOp)) {
      tilingLoops[0]->emitOpError("failed to drop unit dimensions");
      return signalPassFailure();
    }
  }

  // Cleanup patterns for tile and distribute
  {
    RewritePatternSet patterns(context);
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    context->getOrLoadDialect<tensor::TensorDialect>()
        ->getCanonicalizationPatterns(patterns);
    context->getOrLoadDialect<IREE::LinalgExt::IREELinalgExtDialect>()
        ->getCanonicalizationPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    scf::ForallOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError("tiling canonicalization failed");
      return signalPassFailure();
    }
  }

  return;
}

} // namespace mlir::iree_compiler
