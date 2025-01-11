// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
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
  TileAndDistributeToWorkgroupsUsingForallOpPass(bool strategy)
      : transposeWorkgroups(strategy) {}

  using Base::Base;
  void runOnOperation() override;

  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override {
    if (failed(Pass::initializeOptions(options, errorHandler))) {
      return failure();
    }
    auto selectedStrategy = llvm::StringSwitch<FailureOr<bool>>(strategy)
                                .Case("", false)
                                .Case("transpose", true)
                                .Default(failure());
    if (failed(selectedStrategy))
      return failure();

    transposeWorkgroups = *selectedStrategy;
    return success();
  }

private:
  bool transposeWorkgroups = false;
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

  return TilingInfo{tilableOp, tileSizes, interchange};
}

/// Helper function to return the mapping attribute to use given the tile sizes.
static SmallVector<Attribute> getMapping(MLIRContext *context,
                                         ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<Attribute> mapping;
  mapping.reserve(tileSizes.size());
  for (auto tileSize : llvm::reverse(tileSizes)) {
    if (isConstantIntValue(tileSize, 0)) {
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

// Checks whether we have static dimension for all the loop bounds and steps.
// This is a requirement if the reordering strategy is set to `transpose`.
static bool checkStaticLoopBounds(scf::ForallOp forallOp) {

  SmallVector<OpFoldResult> mixedLbs = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> mixedUbs = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> mixedSteps = forallOp.getMixedStep();

  for (auto [index, lb, ub, step] :
       llvm::enumerate(mixedLbs, mixedUbs, mixedSteps)) {

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
static LogicalResult dropUnitDistributedDims(RewriterBase &rewriter,
                                             scf::ForallOp forallOp) {
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
    return success();
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
  return success();
}

//===---------------------------------------------------------------------===//
// Pass implementation.
//===---------------------------------------------------------------------===//

// Fuse all consumers of the given `tiledOp` into the surrounding scf.forall.
// Returns a list of new `tensor.extract_slice` ops with new fusion
// opportunities, as well as the new surrounding `scf.forall` (because consumer
// fusion replaces the loop).
static std::pair<std::queue<Operation *>, scf::ForallOp>
fuseConsumers(RewriterBase &rewriter, Operation *tiledOp) {
  auto addCandidateSlices =
      [](Operation *fusedOp,
         std::queue<tensor::ParallelInsertSliceOp> &candidates) {
        for (auto *userOp : fusedOp->getResults().getUsers()) {
          if (auto sliceOp =
                  llvm::dyn_cast<tensor::ParallelInsertSliceOp>(userOp)) {
            candidates.push(sliceOp);
          }
        }
      };

  // Collect the candidate slices which can be potential consumers that can be
  // fused.
  std::queue<tensor::ParallelInsertSliceOp> candidates;
  addCandidateSlices(tiledOp, candidates);

  std::queue<Operation *> newFusionOpportunities;
  scf::ForallOp newLoop = tiledOp->getParentOfType<scf::ForallOp>();
  while (!candidates.empty()) {

    // Traverse the slices in BFS fashion.
    tensor::ParallelInsertSliceOp candidateSliceOp = candidates.front();
    candidates.pop();

    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedResult =
        mlir::scf::tileAndFuseConsumerOfSlice(rewriter, candidateSliceOp);
    if (failed(fusedResult)) {
      LLVM_DEBUG(llvm::dbgs() << "failed to fuse consumer of slice: "
                              << candidateSliceOp << "\n");
      continue;
    }

    // Replace the original consumer operation with the tiled implementation.
    rewriter.replaceOp(fusedResult->origConsumerOperand->getOwner(),
                       fusedResult->tiledOps.front());

    // The result of the fused consumers might themselved be slices of
    // values produced by operations that implement the `TilingInterface`.
    // Add these operations to the worklist.
    addCandidateSlices(fusedResult->tiledAndFusedConsumerOperand->getOwner(),
                       candidates);

    // Add the list of new producer fusion opportunities.
    for (auto tiledOp : fusedResult.value().tiledOps) {
      for (auto operand : tiledOp->getOperands()) {
        if (auto sliceProducer =
                operand.getDefiningOp<tensor::ExtractSliceOp>()) {
          if (llvm::isa_and_present<TilingInterface>(
                  sliceProducer.getSource().getDefiningOp())) {
            newFusionOpportunities.push(sliceProducer);
          }
        }
      }
      // Store the new loop for follow up producer fusion.
      newLoop = tiledOp->getParentOfType<scf::ForallOp>();
    }
  }
  return std::make_pair(newFusionOpportunities, newLoop);
}

static void fuseProducersOfSlices(RewriterBase &rewriter,
                                  std::queue<Operation *> &worklist,
                                  scf::SCFTileAndFuseOptions &options,
                                  scf::ForallOp forallOp) {
  SmallVector<LoopLikeOpInterface> loops = {
      cast<LoopLikeOpInterface>(&*forallOp)};
  while (!worklist.empty()) {
    auto candidateSlice = cast<tensor::ExtractSliceOp>(worklist.front());
    worklist.pop();

    auto fusableProducer =
        candidateSlice.getSource().getDefiningOp<TilingInterface>();
    if (!fusableProducer)
      continue;

    std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> controlFnResult =
        options.fusionControlFn(candidateSlice,
                                cast<OpResult>(candidateSlice.getSource()),
                                /*destinationInitArg=*/false);
    if (!controlFnResult)
      continue;

    // The operands of the fused producer might themselved be slices of
    // values produced by operations that implement the `TilingInterface`.
    // Add these operations to the worklist.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
        scf::tileAndFuseProducerOfSlice(rewriter, candidateSlice, loops);
    if (!fusedResult)
      continue;

    for (auto newSlice : fusedResult->generatedSlices) {
      worklist.push(newSlice);
    }
  }
}

/// Starting from `op` walk all operands backwards to find all
/// potentially fusable operations, i.e. operations that implement
/// the `TilingInterface`.
static void collectTiledAndFusedOps(Operation *rootOp,
                                    llvm::SmallDenseSet<Operation *> &result) {
  SmallVector<Operation *> worklist;
  worklist.push_back(rootOp);
  result.insert(rootOp);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    // Collect all tilable producers.
    for (OpOperand &operand : current->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer || !isa<TilingInterface>(producer) ||
          result.count(producer))
        continue;
      worklist.push_back(producer);
      result.insert(producer);
    }
    // Collect all tilable consumers.
    for (auto user : current->getUsers()) {
      if (result.count(user)) {
        continue;
      }
      if (isa<TilingInterface>(user)) {
        worklist.push_back(user);
        result.insert(user);
      }
    }
  }
}

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

  llvm::DenseSet<Operation *> yieldReplacementsFor;
  for (auto op : tiledAndFusedOps) {
    // Yield a replacement if:
    //  a) All users of fused op are dominated by the tiling root.
    //  b) There is at most a single tiled user. If there is more than one
    //     then yielding a replacement may result in multiple incompatible
    //     consumer fusions.
    if (llvm::any_of(op->getUsers(),
                     [&](Operation *user) {
                       return dominanceInfo.properlyDominates(tilableOp, user);
                     }) &&
        (llvm::count_if(op->getUsers(), [&](Operation *user) {
           return tiledAndFusedOps.contains(user);
         }) < 2)) {
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

  // The control function that determines whether a tiled producer should yield
  // its replacement.
  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
          bool isDestinationOperand)
      -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    Operation *owner = originalProducer.getOwner();
    bool yieldProducerReplacement = yieldReplacementsFor.contains(owner);
    return scf::SCFTileAndFuseOptions::ControlFnResult{
        yieldProducerReplacement};
    return std::nullopt;
  };
  tileAndFuseOptions.setFusionControlFn(controlFn);
  rewriter.setInsertionPoint(tilableOp);

  // If the `tilableOp` is a `memref` op, then just tile the operation.
  SmallVector<LoopLikeOpInterface> tilingLoops;
  Operation *rootTiledOp = nullptr;
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
    rootTiledOp = tileAndFuseResult->tiledAndFusedOps.front();
  }
  if (!tilingLoops.empty()) {
    if (tilingLoops.size() != 1 || !isa<scf::ForallOp>(tilingLoops[0])) {
      funcOp.emitOpError(
          "expected tiling to produce a single `scf.forall` loop");
      return signalPassFailure();
    }

    auto forallOp = cast<scf::ForallOp>(tilingLoops[0]);
    if (failed(dropUnitDistributedDims(rewriter, forallOp))) {
      forallOp.emitOpError("failed to drop unit dimensions");
      return signalPassFailure();
    }

    if (rootTiledOp) {
      auto [newFusionOpportunities, newLoop] =
          fuseConsumers(rewriter, rootTiledOp);

      // Because we restrict to at most a single tilable consumer for yielding
      // a replacement, no new fusion opportunities will yield a replacement,
      // meaning there is no need to run consumer fusion again afterwards.
      // TODO: run producer and consumer fusion in one worklist.
      fuseProducersOfSlices(rewriter, newFusionOpportunities,
                            tileAndFuseOptions, newLoop);
      forallOp = newLoop;
    }

    // Reorder the workgroups if the strategy is set to `transpose`.
    // This just transposes the first two dimensions of the workgroup i.e., the
    // #iree.codegen.workgroup_id_x and #iree.codegen.workgroup_id_y.
    // Only reorders if the loop bounds are static.
    if (transposeWorkgroups) {
      SmallVector<Attribute> mappingAttrs(forallOp.getMappingAttr().getValue());
      int64_t mappingSize = mappingAttrs.size();
      if (checkStaticLoopBounds(forallOp) && mappingAttrs.size() >= 2) {
        std::swap(mappingAttrs[mappingSize - 1], mappingAttrs[mappingSize - 2]);
        forallOp.setMappingAttr(ArrayAttr::get(context, mappingAttrs));
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
createTileAndDistributeToWorkgroupsWithReordering(
    bool reorderWorkgroupsWithTranspose) {
  return std::make_unique<TileAndDistributeToWorkgroupsUsingForallOpPass>(
      reorderWorkgroupsWithTranspose);
}
} // namespace mlir::iree_compiler
