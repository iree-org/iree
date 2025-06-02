// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "tile-and-distribute-to-workgroups-using-forall-op"

namespace mlir::iree_compiler {

#define CEILDIV(a, b) ((a + b - 1) / b)

#define GEN_PASS_DEF_TILEANDDISTRIBUTETOWORKGROUPSUSINGFORALLOPPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct TileAndDistributeToWorkgroupsUsingForallOpPass final
    : public impl::TileAndDistributeToWorkgroupsUsingForallOpPassBase<
          TileAndDistributeToWorkgroupsUsingForallOpPass> {
  explicit TileAndDistributeToWorkgroupsUsingForallOpPass(
      bool transposeWorkgroup) {
    this->transposeWorkgroup = transposeWorkgroup;
  }
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

/// Find the lowering config to use for getting the tile sizes.
// TODO: For now this is taking the "last op" in the dispatch, but
// ideally this should take the "root op" that gets tiled and everything
// gets fused with it. For now to keep consistent with the legacy
// tile-and-distribute it is still looking for the "last compute operation".
struct TilingInfo {
  Operation *tilableOp;
  SmallVector<OpFoldResult> tileSizes;
  SmallVector<int64_t> interchange;
};

static FailureOr<TilingInfo>
getTiledAndDistributionInfo(RewriterBase &rewriter,
                            ArrayRef<Operation *> computeOps) {
  // TODO: It is expected that at most one compute op has a workgroup tiling
  // level. Currently, it selects the last compute op that has workgroup tiling
  // level.
  Operation *tilableOp = nullptr;
  for (Operation *op : llvm::reverse(computeOps)) {
    if (getLoweringConfig(op)) {
      if (!getLoweringConfig(op).hasWorkgroupTilingLevel()) {
        continue;
      }
      tilableOp = op;
      break;
    }
  }
  if (!tilableOp) {
    // There is no lowering config. Return `null`.
    return TilingInfo{nullptr, {}, {}};
  }

  IREE::Codegen::LoweringConfigAttrInterface tilableOpConfig =
      getLoweringConfig(tilableOp);
  if (!tilableOpConfig) {
    return tilableOp->emitOpError("unable to find configuration of root op to "
                                  "define workgroup count region");
  }
  auto tileSizes = llvm::map_to_vector(
      tilableOpConfig.getWorkgroupTileSizes(),
      [&](int64_t t) -> OpFoldResult { return rewriter.getIndexAttr(t); });
  SmallVector<int64_t> interchange = tilableOpConfig.getWorkgroupInterchange();

  // Avoid distributing unit-trip count loops.

  // Set tile sizes for non-partitioned loops to zero.
  if (auto partitionableLoopsInterface =
          dyn_cast<PartitionableLoopsInterface>(tilableOp)) {
    SmallVector<unsigned> partitionableLoops =
        partitionableLoopsInterface.getPartitionableLoops(std::nullopt);
    llvm::SmallDenseSet<unsigned> partitionableLoopsSet(
        partitionableLoops.begin(), partitionableLoops.end());
    OpFoldResult zero = rewriter.getIndexAttr(0);
    for (auto loopId : llvm::seq<unsigned>(0, tileSizes.size())) {
      if (partitionableLoopsSet.count(loopId)) {
        continue;
      }
      tileSizes[loopId] = zero;
    }
  }

  // Set tile sizes for full tiles to zero. This prevents single trip loops from
  // being created, which can sometimes block certain cleanup patterns from
  // applying during producer fusion.
  if (auto tilingInterfaceOp = dyn_cast<TilingInterface>(tilableOp)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(tilingInterfaceOp);
    SmallVector<Range> bounds = tilingInterfaceOp.getIterationDomain(rewriter);
    SmallVector<int64_t> staticLoopSizes;
    SmallVector<Value> d;
    for (Range bound : bounds) {
      dispatchIndexOpFoldResult(bound.size, d, staticLoopSizes);
    }
    OpFoldResult zero = rewriter.getIndexAttr(0);
    SmallVector<int64_t> tileSizesInt = tilableOpConfig.getWorkgroupTileSizes();
    for (auto loopId : llvm::seq<unsigned>(0, tileSizesInt.size())) {
      if (loopId < staticLoopSizes.size() &&
          staticLoopSizes[loopId] == tileSizesInt[loopId]) {
        tileSizes[loopId] = zero;
      }
    }
  }

  return TilingInfo{tilableOp, tileSizes, interchange};
}

/// Helper function to return the mapping attribute to use given the tile sizes.
static SmallVector<Attribute> getMapping(MLIRContext *context,
                                         ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<Attribute> mapping;
  mapping.reserve(tileSizes.size());
  for (auto tileSize : llvm::reverse(tileSizes)) {
    if (isZeroInteger(tileSize)) {
      continue;
    }
    uint64_t currSize = mapping.size();
    switch (currSize) {
    case 0:
    case 1:
    case 2:
      mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
          context, IREE::Codegen::symbolizeWorkgroupId(currSize).value()));
      break;
    default:
      mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
          context, IREE::Codegen::WorkgroupId::IdZ, currSize - 2));
    }
  }
  return llvm::to_vector(llvm::reverse(mapping));
}

//===---------------------------------------------------------------------===//
// Post tiling cleanup patterns
//===---------------------------------------------------------------------===//

/// Prune the values corresponding to the dropped loops.
static SmallVector<OpFoldResult>
pruneDroppedLoops(ArrayRef<OpFoldResult> inputs,
                  const llvm::SmallDenseSet<int> &droppedLoops) {
  SmallVector<OpFoldResult> prunedInputs;
  for (auto [index, input] : llvm::enumerate(inputs)) {
    if (droppedLoops.contains(index)) {
      continue;
    }
    prunedInputs.push_back(input);
  }
  return prunedInputs;
}

/// Prune the mapping attributes corresponding to the dropped loops.
/// Note that we cant just drop them. We need to rebalance the
/// attributes so that the workgroup attributes are perfectly ordered.
/// For example, if the attribute list is
///
/// ```
/// [workgroup_mapping<x>, workgroup_mapping<z:1>,
///  workgroup_mapping<z>, workgroup_mapping<y>,
///  workgroup_mapping<z:3>, workgroup_mapping<z:2>]
/// ```
///
/// and the droppedloops are `{1, 3}`, then the new mapping should be
///
/// ```
/// [workgroup_mapping<x>, workgroup_mapping<y>,
///  workgroup_mapping<z:1>, workgroup_mapping<z>]
/// ```
SmallVector<Attribute>
pruneDroppedLoops(ArrayRef<Attribute> inputs,
                  const llvm::SmallDenseSet<int> &droppedLoops) {
  SmallVector<IREE::Codegen::WorkgroupMappingAttr> droppedMappings;
  SmallVector<Attribute> prunedAttrs;
  for (auto [index, input] : llvm::enumerate(inputs)) {
    if (droppedLoops.contains(index)) {
      droppedMappings.push_back(
          cast<IREE::Codegen::WorkgroupMappingAttr>(input));
    } else {
      prunedAttrs.push_back(input);
    }
  }
  for (auto droppedMapping : droppedMappings) {
    for (auto [index, prunedAttr] : llvm::enumerate(prunedAttrs)) {
      auto prunedMappingAttr =
          cast<IREE::Codegen::WorkgroupMappingAttr>(prunedAttr);
      if (droppedMapping < prunedMappingAttr) {
        prunedAttrs[index] =
            IREE::Codegen::WorkgroupMappingAttr::getAttributeFromMappingId(
                prunedAttr.getContext(), prunedMappingAttr.getMappingId() - 1);
      }
    }
  }
  return prunedAttrs;
}

/// Checks whether we have static dimension for all the loop bounds and steps.
/// This is a requirement if the reordering strategy is set to `transpose`.
static bool areAllStaticLoopBounds(scf::ForallOp forallOp) {

  for (auto [lb, ub, step] : llvm::zip_equal(forallOp.getMixedLowerBound(),
                                             forallOp.getMixedUpperBound(),
                                             forallOp.getMixedStep())) {

    std::optional<int64_t> lbVal = getConstantIntValue(lb);
    std::optional<int64_t> ubVal = getConstantIntValue(ub);
    std::optional<int64_t> stepVal = getConstantIntValue(step);

    if (!(lbVal && ubVal && stepVal)) {
      return false;
    }
  }
  return true;
}

/// Find dimensions of the loop that are unit-trip count and drop them from the
/// distributed dimensions.
static FailureOr<scf::ForallOp>
dropUnitDistributedDims(RewriterBase &rewriter, scf::ForallOp forallOp) {
  SmallVector<OpFoldResult> mixedLbs = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> mixedUbs = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> mixedSteps = forallOp.getMixedStep();

  // Find the index of loops to be dropped.
  llvm::SmallDenseSet<int> droppedLoops;
  for (auto [index, lb, ub, step] :
       llvm::enumerate(mixedLbs, mixedUbs, mixedSteps)) {

    std::optional<int64_t> lbVal = getConstantIntValue(lb);
    std::optional<int64_t> ubVal = getConstantIntValue(ub);
    std::optional<int64_t> stepVal = getConstantIntValue(step);

    if (!(lbVal && ubVal && stepVal)) {
      continue;
    }

    if (CEILDIV(ubVal.value() - lbVal.value(), stepVal.value()) == 1) {
      droppedLoops.insert(index);
    }
  }
  if (droppedLoops.empty()) {
    return forallOp;
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);
  SmallVector<OpFoldResult> newLbs =
      pruneDroppedLoops(ArrayRef<OpFoldResult>(mixedLbs), droppedLoops);
  SmallVector<OpFoldResult> newUbs =
      pruneDroppedLoops(ArrayRef<OpFoldResult>(mixedUbs), droppedLoops);
  SmallVector<OpFoldResult> newSteps =
      pruneDroppedLoops(ArrayRef<OpFoldResult>(mixedSteps), droppedLoops);
  std::optional<ArrayAttr> newMapping;
  if (auto currMapping = forallOp.getMapping()) {
    SmallVector<Attribute> newMappingAttrs =
        pruneDroppedLoops(currMapping.value().getValue(), droppedLoops);
    newMapping = rewriter.getArrayAttr(newMappingAttrs);
  }

  Value zero = rewriter.create<arith::ConstantIndexOp>(forallOp.getLoc(), 0);
  auto newForallOp = rewriter.create<scf::ForallOp>(
      forallOp.getLoc(), newLbs, newUbs, newSteps, forallOp.getInits(),
      newMapping, [](OpBuilder &, Location, ValueRange) {});

  SmallVector<Value> argReplacements;
  int newLoopBlockArgNum = 0;
  auto newLoopBodyArgs = newForallOp.getInductionVars();
  for (auto [index, oldBlockArg] :
       llvm::enumerate(forallOp.getInductionVars())) {
    if (droppedLoops.contains(index)) {
      argReplacements.push_back(zero);
    } else {
      argReplacements.push_back(newLoopBodyArgs[newLoopBlockArgNum++]);
    }
  }
  argReplacements.append(newForallOp.getRegionIterArgs().begin(),
                         newForallOp.getRegionIterArgs().end());

  Block *oldLoopBody = forallOp.getBody();
  Block *newLoopBody = newForallOp.getBody();
  rewriter.mergeBlocks(oldLoopBody, newLoopBody, argReplacements);

  rewriter.replaceOp(forallOp, newForallOp.getResults());
  return newForallOp;
}

//===---------------------------------------------------------------------===//
// Pass implementation.
//===---------------------------------------------------------------------===//

void TileAndDistributeToWorkgroupsUsingForallOpPass::runOnOperation() {
  auto funcOp = getOperation();
  auto *context = &getContext();
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);

  IRRewriter rewriter(context);
  FailureOr<TilingInfo> tilingInfo =
      getTiledAndDistributionInfo(rewriter, computeOps);
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
      getMapping(context, tilingInfo->tileSizes);
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
  // TODO(Max191): Replace populateSwapExtractWithExpandPattern with upstream
  // MLIR version once it is available (llvm-project/pull/126898).
  populateSwapExtractWithExpandPattern(cleanupPatterns);
  // When fusing pads we do not want to generate zeroSliceGuards when doing
  // workgroup tiling. In `GPUApplyTilingLevelPass` we do have an option called
  // `allowZeroSlices` that can control this but we do not want these
  // generated if workgroup tiling is happening first.
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
        fuseConsumersIntoForall(rewriter, rootTiledOp, tilingLoops,
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

    // Reorder the workgroups if the strategy is set to `transpose`.
    // This just transposes the first two dimensions of the workgroup i.e., the
    // #iree.codegen.workgroup_id_x and #iree.codegen.workgroup_id_y.
    // Only reorders if the loop bounds are static.
    if (transposeWorkgroup) {
      SmallVector<Attribute> mappingAttrs(
          forallOp->getMappingAttr().getValue());
      int64_t mappingSize = mappingAttrs.size();
      if (areAllStaticLoopBounds(*forallOp) && mappingSize >= 2) {
        std::swap(mappingAttrs[mappingSize - 1], mappingAttrs[mappingSize - 2]);
        forallOp->setMappingAttr(ArrayAttr::get(context, mappingAttrs));
      }
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
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTileAndDistributeToWorkgroupsWithReordering(bool transposeWorkgroup) {
  return std::make_unique<TileAndDistributeToWorkgroupsUsingForallOpPass>(
      transposeWorkgroup);
}
} // namespace mlir::iree_compiler
