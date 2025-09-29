// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "tile-and-distribute-to-workgroups-using-forall-op"

namespace mlir::iree_compiler {

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
    int numNonZero = tileSizesInt.size() - llvm::count(tileSizesInt, 0);
    for (auto loopId :
         llvm::reverse(llvm::seq<unsigned>(0, tileSizesInt.size()))) {
      // Do not set all sizes to 0, or else the distribution loop will not be
      // created.
      if (numNonZero <= 1) {
        break;
      }
      if (loopId < staticLoopSizes.size() &&
          staticLoopSizes[loopId] == tileSizesInt[loopId]) {
        tileSizes[loopId] = zero;
        --numNonZero;
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

/// Returns true if it is allowed to leave the `op` outside distribution loops,
/// i.e., scf.forall loops. The consumer fusion could fail even the `op`
/// implements TilingInterface. E.g., linalg.pack op can only be fused as a
/// consumer in perfect tiling scenario.
static bool isAllowedToFailOnCunsumerFusion(Operation *op) {
  return isa<linalg::PackOp>(op);
}

/// Returns true if all the compute ops are within scf.forall distribution
/// loops, except the ops that are allowed to stay outside.
static bool verifyComputeOpsAfterDistribution(FunctionOpInterface funcOp) {
  WalkResult res = funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<scf::ForallOp>(op) || !isComputeOp(op)) {
      return WalkResult::skip();
    }
    if (!isAllowedToFailOnCunsumerFusion(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !res.wasInterrupted();
}

//===---------------------------------------------------------------------===//
// Pass implementation.
//===---------------------------------------------------------------------===//

/// Returns true if any value produced by `producer` is used as an init value
/// for the DPS `user`. Returns false if the user is not in DPS.
static bool isUsedAsInit(Operation *producer, Operation *user) {
  auto dpsIface = dyn_cast<DestinationStyleOpInterface>(user);
  if (!dpsIface)
    return false;
  ValueRange results = producer->getResults();
  return llvm::any_of(dpsIface.getDpsInits(), [&](Value operand) {
    return llvm::is_contained(results, operand);
  });
}

static std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
                  SmallVector<OpFoldResult>>
getLoopBounds(RewriterBase &rewriter, Location loc, ArrayRef<Range> loopRanges,
              ArrayRef<OpFoldResult> givenTileSizes) {
  SmallVector<OpFoldResult> lbs, ubs, steps;
  for (auto [loopRange, givenTileSize] :
       llvm::zip_equal(loopRanges, givenTileSizes)) {
    // No loop if the tile size is 0.
    if (isZeroInteger(givenTileSize))
      continue;
    lbs.push_back(loopRange.offset);
    ubs.push_back(loopRange.size);
    steps.push_back(givenTileSize);
  }
  return {lbs, ubs, steps};
}

static Value computeNbTiles(OpBuilder &b, Location loc, OpFoldResult size,
                            OpFoldResult tileSize) {
  Value sizeVal = getValueOrCreateConstantIntOp(b, loc, size);
  Value tileSizeVal = getValueOrCreateConstantIntOp(b, loc, tileSize);
  auto nbTiles =
      b.create<mlir::arith::DivUIOp>(loc, sizeVal, tileSizeVal).getResult();
  return getValueOrCreateConstantIntOp(b, loc, nbTiles);
}

/// Computes the total number of Tile loads per iteration for the pingpong
/// matmul kernel.
static Value computeNumTileLoads(OpBuilder &b, Location loc, OpFoldResult size,
                                 OpFoldResult tileSize, Value nbXcds,
                                 Value nbCus) {

  Value nbTiles = computeNbTiles(b, loc, size, tileSize);
  auto totalCus = b.create<mlir::arith::MulIOp>(loc, nbXcds, nbCus);
  auto nbX = b.create<mlir::arith::MinUIOp>(loc, nbTiles, totalCus).getResult();

  auto nbY = b.create<mlir::arith::CeilDivUIOp>(loc, totalCus, nbX);

  return getValueOrCreateConstantIndexOp(
      b, loc, b.create<mlir::arith::AddIOp>(loc, nbX, nbY).getResult());
}

static Value getCondition(OpBuilder &b, Location loc,
                          ArrayRef<OpFoldResult> ubs,
                          ArrayRef<OpFoldResult> steps, Value nbXcds,
                          Value nbCus) {
  assert(ubs.size() == 2 && steps.size() == 2 && "rank must be 2");
  Value transposedOrder =
      computeNumTileLoads(b, loc, ubs[0], steps[0], nbXcds, nbCus);
  Value defaultOrder =
      computeNumTileLoads(b, loc, ubs[1], steps[1], nbXcds, nbCus);

  AffineExpr s0, s1, s2;
  bindSymbols(b.getContext(), s0, s1, s2);
  AffineExpr transposedOrderBumpExpr = (s0 * s1).floorDiv(s2);
  Value c4 = b.create<arith::ConstantIntOp>(loc, 4, 32);
  Value c5 = b.create<arith::ConstantIntOp>(loc, 5, 32);

  Value defaultOrderTolerance = getValueOrCreateConstantIntOp(
      b, loc,
      affine::makeComposedFoldedAffineApply(b, loc, transposedOrderBumpExpr,
                                            {defaultOrder, c4, c5}));

  Value cond =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult,
                                    transposedOrder, defaultOrderTolerance);
  return cond;
}

static void swapIf(OpBuilder &b, Location loc, OpFoldResult pred,
                   SmallVector<OpFoldResult> &values,
                   ArrayRef<size_t> ids = {0, 1}) {

  assert(ids.size() == 2 && "Can only swap between 2 indices");

  Value v0 = getValueOrCreateConstantIndexOp(b, loc, values[ids[0]]);
  Value v1 = getValueOrCreateConstantIndexOp(b, loc, values[ids[1]]);
  Value predVal = getValueOrCreateConstantIndexOp(b, loc, pred);
  values[ids[0]] =
      b.create<mlir::arith::SelectOp>(loc, predVal, v1, v0).getResult();
  values[ids[1]] =
      b.create<mlir::arith::SelectOp>(loc, predVal, v0, v1).getResult();
}

static std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
                  SmallVector<size_t>>
computeOffsetAndSize(RewriterBase &rewriter, Location loc,
                     ArrayRef<Range> loopRanges,
                     ArrayRef<OpFoldResult> givenTileSizes,
                     ArrayRef<Value> ivs) {
  SmallVector<size_t> ids;
  SmallVector<OpFoldResult> offsets, sizes;

  offsets.reserve(loopRanges.size());
  sizes.reserve(loopRanges.size());
  const Value *ivIt = ivs.begin();

  for (auto [loopRange, givenTileSize] :
       llvm::zip_equal(loopRanges, givenTileSizes)) {
    if (isZeroInteger(givenTileSize)) {
      offsets.push_back(loopRange.offset);
      sizes.push_back(loopRange.size);
    } else {
      offsets.push_back(*ivIt++);
      sizes.push_back(givenTileSize);
      ids.push_back(sizes.size() - 1);
    }
  }
  return {offsets, sizes, ids};
}

void TileAndDistributeToWorkgroupsUsingForallOpPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
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
    // Require replacement for values that are used after the main tilable op or
    // by ops that will definitely not be fused. Note that if a value is used as
    // an init of a DPS op, the user currently cannot be fused. Having a
    // replacement for it would attempt fusion and fail, so avoid such cases.
    if (llvm::any_of(op->getUsers(), [&](Operation *user) {
          if (isUsedAsInit(op, user))
            return false;
          return dominanceInfo.properlyDominates(tilableOp, user) ||
                 !tiledAndFusedOps.contains(user);
        })) {
      yieldReplacementsFor.insert(op);
    }
  }

  SmallVector<Attribute> deviceMappingAttribute =
      getMapping(context, tilingInfo->tileSizes);
  if (failed(IREE::Codegen::WorkgroupMappingAttr::verifyAttrList(
          context, funcOp.getLoc(), deviceMappingAttribute))) {
    return signalPassFailure();
  }

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tilingInfo->tileSizes);
  tilingOptions.setInterchange(tilingInfo->interchange);
  tilingOptions.setMapping(deviceMappingAttribute);
  const char *env_var = std::getenv("TEST");

  if (!env_var) {
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  } else {
    llvm::outs() << "Test mode\n";
    // START

    scf::SCFTilingOptions::GenerateLoopHeaderFn loopHeaderFn =
        [&](RewriterBase &rewriter, Location loc, ArrayRef<Range> loopRanges,
            ArrayRef<OpFoldResult> givenTileSizes,
            ValueRange outerDestinationTensors)
        -> FailureOr<scf::SCFTilingOptions::CustomLoopHeaderInfo> {
      scf::ForallOp forallOp;
      // Get loop bounds
      SmallVector<OpFoldResult> lbs, ubs, steps;
      std::tie(lbs, ubs, steps) =
          getLoopBounds(rewriter, loc, loopRanges, givenTileSizes);

      assert(lbs.size() == 2 && "only rank 2 is supported");

      // Rewrite ourselve ?
      std::optional<ArrayAttr> mappingAttr;
      if (!tilingOptions.mappingVector.empty())
        mappingAttr = rewriter.getArrayAttr(tilingOptions.mappingVector);

      const int NBCUS = 38;
      const int NBXCDS = 8;

      // Apply condition on loop bounds
      Value nbCusVal = rewriter.create<arith::ConstantIndexOp>(loc, NBCUS);
      Value nbXCDs = rewriter.create<arith::ConstantIndexOp>(loc, NBXCDS);
      Value cond = getCondition(rewriter, loc, ubs, steps, nbXCDs, nbCusVal);

      swapIf(rewriter, loc, cond, lbs);
      swapIf(rewriter, loc, cond, ubs);
      swapIf(rewriter, loc, cond, steps);

      forallOp = scf::ForallOp::create(rewriter, loc, lbs, ubs, steps,
                                       outerDestinationTensors, mappingAttr);

      SmallVector<Value> ivs = forallOp.getInductionVars();

      SmallVector<OpFoldResult> offsets, sizes;
      SmallVector<size_t> ids;
      std::tie(offsets, sizes, ids) =
          computeOffsetAndSize(rewriter, loc, loopRanges, givenTileSizes, ivs);

      swapIf(rewriter, loc, cond, offsets, ids);
      swapIf(rewriter, loc, cond, sizes, ids);

      ValueRange innerDestinationTensors = forallOp.getRegionOutArgs();
      rewriter.setInsertionPoint(forallOp.getTerminator());

      return scf::SCFTilingOptions::CustomLoopHeaderInfo{
          {cast<LoopLikeOpInterface>(forallOp.getOperation())},
          offsets,
          sizes,
          innerDestinationTensors};
    };

    scf::SCFTilingOptions::GenerateLoopTerminatorFn terminatorFn =
        [&](RewriterBase &rewriter, Location loc, ValueRange tiledResults,
            ArrayRef<SmallVector<OpFoldResult>> resultOffsets,
            ArrayRef<SmallVector<OpFoldResult>> resultSizes,
            ValueRange destinationTensors) -> LogicalResult {
      for (auto [tiledValue, destinationTensor, resultOffset, resultSize] :
           llvm::zip_equal(tiledResults, destinationTensors, resultOffsets,
                           resultSizes)) {
        SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                               rewriter.getIndexAttr(1));

        tensor::ParallelInsertSliceOp::create(rewriter, loc, tiledValue,
                                              destinationTensor, resultOffset,
                                              resultSize, resultStride);
      }
      return success();
    };
    // END

    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::CustomOp);
    tilingOptions.setCustomLoopGenerationFns(loopHeaderFn, terminatorFn);
  }

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
      Value replacementCopy = replacement;
      rewriter.replaceUsesWithIf(origValue, replacement, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return !isa<tensor::DimOp>(user) &&
               dominanceInfo.dominates(replacementCopy, user);
      });
    }
    std::swap(tileAndFuseResult->loops, tilingLoops);

    FailureOr<std::queue<Operation *>> newFusionOpportunities =
        fuseConsumersIntoForall(
            rewriter, tileAndFuseResult->tiledAndFusedOps.getArrayRef(),
            tilingLoops, [&tiledAndFusedOps](Operation *op) {
              return tiledAndFusedOps.contains(op);
            });
    if (failed(newFusionOpportunities)) {
      // Continue the work if the failure is allowed.
      if (!verifyComputeOpsAfterDistribution(funcOp)) {
        tileAndFuseResult->tiledAndFusedOps.front()->emitOpError(
            "failed to fuse consumers");
        return signalPassFailure();
      }
    } else {
      // Because we restrict to at most a single tilable consumer for yielding
      // a replacement, no new fusion opportunities will yield a replacement,
      // meaning there is no need to run consumer fusion again afterwards.
      // TODO: run producer and consumer fusion in one worklist.
      fuseProducersOfSlices(rewriter, *newFusionOpportunities,
                            tileAndFuseOptions, tilingLoops);
    }
  }
  if (!tilingLoops.empty()) {
    if (tilingLoops.size() != 1 || !isa<scf::ForallOp>(tilingLoops[0])) {
      funcOp.emitOpError(
          "expected tiling to produce a single `scf.forall` loop");
      return signalPassFailure();
    }

    // Reorder the workgroups if the strategy is set to `transpose`.
    // This just transposes the first two dimensions of the workgroup i.e., the
    // #iree.codegen.workgroup_id_x and #iree.codegen.workgroup_id_y.
    // Only reorders if the loop bounds are static.
    auto forallOp = cast<scf::ForallOp>(tilingLoops[0]);
    if (transposeWorkgroup) {
      SmallVector<Attribute> mappingAttrs(forallOp.getMappingAttr().getValue());
      int64_t mappingSize = mappingAttrs.size();
      if (areAllStaticLoopBounds(forallOp) && mappingSize >= 2) {
        std::swap(mappingAttrs[mappingSize - 1], mappingAttrs[mappingSize - 2]);
        forallOp.setMappingAttr(ArrayAttr::get(context, mappingAttrs));
      }
    }
  }

  // Cleanup patterns for tile and distribute
  {
    RewritePatternSet patterns(context);
    populateSwapExtractWithCollapsePattern(patterns);
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
