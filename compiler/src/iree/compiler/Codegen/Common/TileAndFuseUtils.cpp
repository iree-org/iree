// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/TilingInterface.h"

#include <cassert>

#define DEBUG_TYPE "iree-codegen-common-tile-and-fuse-utils"

namespace mlir::iree_compiler {

scf::InnerTileAlignmentFnTy
makeInnerTileAlignmentFn(IREE::CPU::TilingLevel tilingLevel) {
  return [tilingLevel](
             TilingInterface op, ArrayRef<OpFoldResult>,
             ArrayRef<Operation *>) -> SmallVector<mlir::InnerTileAlignment> {
    // Tile-size selection already recorded this op's per-dimension inner-tile
    // alignments per tiling level. We simply forward those along.
    auto hint =
        IREE::CPU::InnerTileAlignmentsAttr::getFromOp(op.getOperation());
    if (!hint) {
      return {};
    }
    auto alignments = hint.getAlignments().getAs<DenseI64ArrayAttr>(
        IREE::CPU::getTilingLevelName(tilingLevel));
    return alignments
               ? mlir::convertInnerTileAlignments(alignments.asArrayRef())
               : SmallVector<mlir::InnerTileAlignment>{};
  };
}

void fuseProducersOfSlices(RewriterBase &rewriter,
                           std::queue<Operation *> &worklist,
                           scf::SCFTileAndFuseOptions &options,
                           MutableArrayRef<LoopLikeOpInterface> loops) {
  while (!worklist.empty()) {
    auto candidateSlice = cast<tensor::ExtractSliceOp>(worklist.front());
    worklist.pop();

    auto fusableProducer =
        candidateSlice.getSource().getDefiningOp<TilingInterface>();
    if (!fusableProducer) {
      continue;
    }

    std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> controlFnResult =
        options.fusionControlFn(candidateSlice,
                                cast<OpResult>(candidateSlice.getSource()),
                                /*destinationInitArg=*/false);
    if (!controlFnResult) {
      continue;
    }

    // The operands of the fused producer might themselves be slices of
    // values produced by operations that implement the `TilingInterface`.
    // Add these operations to the worklist.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
        scf::tileAndFuseProducerOfSlice(
            rewriter, candidateSlice, loops,
            options.tilingOptions.innerTileAlignmentFn);
    if (!fusedResult) {
      continue;
    }

    for (auto newSlice : fusedResult->generatedSlices) {
      worklist.push(newSlice);
    }
  }
}

void collectTiledAndFusedOps(Operation *rootOp,
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
          result.contains(producer)) {
        continue;
      }
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

/// Adds the fusable users of the loop results that `fusedOps` write to (through
/// `tensor.parallel_insert_slice`) to `candidates`, keeping it free of
/// duplicates and reverse-sorted by dominance. The most-dominating candidate is
/// to be fused first since not fusing it may prevent dominated candidates to be
/// fused:
///
///     A
///     |
///     B
///   / |
///  |  D
///  | /
///  C
///
/// here, B must be fused before both C and D, and D must be fused before C.
///
/// Every fusion rebuilds the loop and erases the old one, so handles to the
/// loop and its results are only valid within one call and none is kept across
/// calls. Consumer ops survive fusions with their loop-result operands
/// redirected to the rebuilt loop, so only consumers are kept in `candidates`.
static void
collectConsumerFusionCandidates(ArrayRef<Operation *> fusedOps,
                                llvm::function_ref<bool(Operation *)> filterFn,
                                SmallVectorImpl<Operation *> &candidates) {
  DominanceInfo dominanceInfo;
  for (Operation *fusedOp : fusedOps) {
    for (Operation *userOp : fusedOp->getUsers()) {
      auto sliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(userOp);
      if (!sliceOp) {
        continue;
      }
      auto loop = cast<scf::ForallOp>(sliceOp->getParentOp()->getParentOp());
      OpResult loopResult = loop.getTiedOpResult(
          loop.getTiedOpOperand(cast<BlockArgument>(sliceOp.getDest())));
      SmallVector<Operation *> users = llvm::to_vector(
          llvm::make_filter_range(loopResult.getUsers(), filterFn));
      if (users.empty()) {
        continue;
      }
      mlir::computeTopologicalSorting(users);
      for (Operation *fusableUser : users) {
        // Several slices may write the same loop result and a user may read
        // several loop results, so deduplicate by identity (dominance is only
        // a partial order, so equality cannot be probed through the ordering).
        // `tileAndFuseConsumer` checks that each loop result the user reads is
        // written by exactly one slice.
        if (llvm::is_contained(candidates, fusableUser)) {
          continue;
        }
        // Comparator that puts the dominating user last.
        auto comp = [&](Operation *lhs, Operation *rhs) {
          return dominanceInfo.properlyDominates(rhs, lhs);
        };
        candidates.insert(llvm::lower_bound(candidates, fusableUser, comp),
                          fusableUser);
      }
    }
  }
}

FailureOr<std::queue<Operation *>> fuseConsumersIntoForall(
    RewriterBase &rewriter, ArrayRef<Operation *> tiledOps,
    MutableArrayRef<LoopLikeOpInterface> loops,
    std::function<bool(Operation *)> filterFn,
    const scf::InnerTileAlignmentFnTy &innerTileAlignmentFn) {
  SmallVector<Operation *> candidates;
  // Add candidates from all tiled ops, not only the "main" one.
  collectConsumerFusionCandidates(tiledOps, filterFn, candidates);

  std::queue<Operation *> newFusionOpportunities;
  while (!candidates.empty()) {
    // Get the next candidate.
    Operation *consumer = candidates.pop_back_val();

    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedResult =
        mlir::scf::tileAndFuseConsumer(rewriter, consumer, loops,
                                       innerTileAlignmentFn);
    if (failed(fusedResult)) {
      return failure();
    }
    assert(!fusedResult->origConsumerOperands.empty() &&
           "candidate must use a result of loops.front()");

    // Replace the original consumer operation with the tiled implementation.
    rewriter.replaceOp(fusedResult->origConsumerOperands.front()->getOwner(),
                       fusedResult->tiledOps.front());

    // The fused consumer is written into a new result of the rebuilt loop, so
    // its users are the next fusion candidates.
    collectConsumerFusionCandidates(
        fusedResult->tiledAndFusedConsumerOperands.front()->getOwner(),
        filterFn, candidates);

    // Add the list of new producer fusion opportunities.
    for (auto tiledOp : fusedResult.value().tiledOps) {
      for (auto operand : tiledOp->getOperands()) {
        if (auto sliceProducer =
                operand.getDefiningOp<tensor::ExtractSliceOp>()) {
          if (isa_and_present<TilingInterface>(
                  sliceProducer.getSource().getDefiningOp())) {
            newFusionOpportunities.push(sliceProducer);
          }
        }
      }
    }
  }
  return newFusionOpportunities;
}

/// This collects the set of operations to tile + fuse starting from the given
/// root |op| and walking up to its producers. Stops at operations given by
/// |exclude| which are expected to receive their own independent tiling for the
/// given level.
static llvm::SmallDenseSet<Operation *>
collectTiledAndFusedOps(Operation *op,
                        llvm::SmallDenseSet<TilingInterface> exclude) {
  SmallVector<Operation *> worklist;
  llvm::SmallDenseSet<Operation *> ops;
  worklist.push_back(op);
  ops.insert(op);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (OpOperand &operand : current->getOpOperands()) {
      auto producer = operand.get().getDefiningOp<TilingInterface>();
      if (!producer || ops.contains(producer) || exclude.contains(producer)) {
        continue;
      }
      worklist.push_back(producer);
      ops.insert(producer);
    }
    for (auto user : current->getUsers()) {
      auto consumer = dyn_cast<TilingInterface>(user);
      if (!consumer || ops.contains(consumer) || exclude.contains(consumer)) {
        continue;
      }
      worklist.push_back(consumer);
      ops.insert(consumer);
    }
  }
  return ops;
}

LogicalResult applyTileAndFuseToEachRoot(
    RewriterBase &rewriter, llvm::SmallDenseSet<TilingInterface> &payloadOps,
    IREE::GPU::TilingLevel tilingLevel, bool allowZeroSlices,
    std::optional<
        llvm::SmallDenseMap<TilingInterface, SmallVector<OpFoldResult>>>
        targetTileMap,
    bool fuseConsumers) {
  MLIRContext *context = rewriter.getContext();
  for (TilingInterface tilingInterfaceOp : payloadOps) {
    mlir::DominanceInfo dominanceInfo(tilingInterfaceOp);

    llvm::SmallDenseSet<Operation *> tiledAndFusedOps =
        collectTiledAndFusedOps(tilingInterfaceOp, payloadOps);
    llvm::DenseSet<Operation *> yieldReplacementsFor;
    for (auto op : tiledAndFusedOps) {
      if (llvm::any_of(op->getUsers(), [&](Operation *user) {
            return dominanceInfo.properlyDominates(tilingInterfaceOp, user);
          })) {
        yieldReplacementsFor.insert(op);
      }
    }

    rewriter.setInsertionPoint(tilingInterfaceOp);
    SmallVector<OpFoldResult> tileSizes;
    if (targetTileMap && targetTileMap->contains(tilingInterfaceOp)) {
      tileSizes = (*targetTileMap)[tilingInterfaceOp];
    } else {
      tileSizes =
          getLoweringConfig(tilingInterfaceOp)
              .getTilingLevelSizes(rewriter, llvm::to_underlying(tilingLevel),
                                   tilingInterfaceOp);
    }

    // Pad the tile sizes with zero.
    auto zero = rewriter.getIndexAttr(0);
    int64_t numLoops = tilingInterfaceOp.getLoopIteratorTypes().size();
    if (tileSizes.size() > numLoops) {
      return failure();
    }
    while (tileSizes.size() < numLoops) {
      tileSizes.push_back(zero);
    }

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);
    if (tilingLevel == IREE::GPU::TilingLevel::Thread ||
        tilingLevel == IREE::GPU::TilingLevel::Subgroup) {
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

      // TODO: Add some helpers to construct this based on the enum type rather
      // than doing it here.
      SmallVector<Attribute> mapping;
      int idx = 0;
      for (auto size : tileSizes) {
        if (!isZeroInteger(size)) {
          unsigned mappingId =
              static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx++;
          if (tilingLevel == IREE::GPU::TilingLevel::Thread) {
            mapping.push_back(gpu::GPUThreadMappingAttr::get(
                context, static_cast<gpu::MappingId>(mappingId)));
          } else {
            // Else it must be subgroup tiling.
            mapping.push_back(gpu::GPUWarpMappingAttr::get(
                context, static_cast<gpu::MappingId>(mappingId)));
          }
        }
      }
      tilingOptions.setMapping(llvm::to_vector(llvm::reverse(mapping)));
    }

    if (tilingLevel == IREE::GPU::TilingLevel::Serial) {
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    }

    if (tilingLevel == IREE::GPU::TilingLevel::PartialReduction) {
      tilingOptions.setReductionTilingStrategy(
          ReductionTilingStrategy::PartialReductionOuterReduction);
      SmallVector<unsigned> reductionDims;
      for (auto [index, iteratorType] :
           llvm::enumerate(tilingInterfaceOp.getLoopIteratorTypes())) {
        if (iteratorType == utils::IteratorType::reduction) {
          reductionDims.push_back(index);
        }
      }
      tilingOptions.setReductionDims(reductionDims);
    }

    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);

    scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool isDestinationOperand)
        -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
      Operation *owner = originalProducer.getOwner();
      if (tilingLevel == IREE::GPU::TilingLevel::Reduction ||
          tilingLevel == IREE::GPU::TilingLevel::PartialReduction ||
          tilingLevel == IREE::GPU::TilingLevel::Subgroup) {
        // Do not fuse pad in reduction and subgroup tiling. We instead fuse
        // pad without zero slice guard as a cleanup pattern.
        if (isa<tensor::PadOp>(owner)) {
          return std::nullopt;
        }
      }
      bool yieldProducerReplacement = false;
      // We dont want this for reduction tiling as it can lead to large tensors
      // being yielded.
      if (tilingLevel != IREE::GPU::TilingLevel::Reduction &&
          tilingLevel != IREE::GPU::TilingLevel::PartialReduction) {
        yieldProducerReplacement = yieldReplacementsFor.contains(owner);
      }
      bool shouldFuse = false;
      if (auto tilingOwner = dyn_cast<TilingInterface>(owner)) {
        shouldFuse = !payloadOps.contains(tilingOwner);
      }
      // Do not fuse destination operands for reduction tiling.
      if (isDestinationOperand &&
          (tilingLevel == IREE::GPU::TilingLevel::Reduction ||
           tilingLevel == IREE::GPU::TilingLevel::PartialReduction)) {
        shouldFuse = false;
      }
      if (shouldFuse) {
        return scf::SCFTileAndFuseOptions::ControlFnResult{
            yieldProducerReplacement};
      }
      return std::nullopt;
    };
    tileAndFuseOptions.setFusionControlFn(controlFn);

    RewritePatternSet cleanupPatterns(context);

    if (allowZeroSlices) {
      // Add pattern to fuse pad operations without zero slice guard, if we
      // know we have no zero slices.
      auto zeroSliceGuard = [](tensor::ExtractSliceOp) -> std::optional<bool> {
        // Do not use zero slice guard.
        return false;
      };
      cleanupPatterns.add<linalg::ExtractSliceOfPadTensorSwapPattern>(
          context, zeroSliceGuard);
    }

    // Avoid cleanup for subgroup level tiling because cleanup/fusion must
    // happen later during lane tiling because failure to fuse at the lane
    // tiling level is irrecoverable if fusion happens now.
    if (tilingLevel != IREE::GPU::TilingLevel::Subgroup) {
      tensor::ExtractSliceOp::getCanonicalizationPatterns(cleanupPatterns,
                                                          context);
      tensor::DimOp::getCanonicalizationPatterns(cleanupPatterns, context);
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(
          cleanupPatterns);
      populateSwapExtractWithExpandPattern(cleanupPatterns);
    }

    tileAndFuseOptions.cleanupPatterns =
        FrozenRewritePatternSet(std::move(cleanupPatterns));

    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp,
                                                  tileAndFuseOptions);
    if (failed(tiledResults)) {
      return failure();
    }

    if (IREE::Codegen::LoweringConfigAttrInterface originalConfig =
            getLoweringConfig(tilingInterfaceOp)) {
      if (!tiledResults->tiledAndFusedOps.empty()) {
        setLoweringConfig(tiledResults->tiledAndFusedOps[0], originalConfig);
      }
    }

    // Perform the replacement of tiled and fused values.
    SmallVector<Operation *> opsToReplace{tilingInterfaceOp};
    llvm::append_range(opsToReplace, tiledResults->fusedProducers);
    for (Operation *toReplace : opsToReplace) {
      for (OpResult res : toReplace->getResults()) {
        if (auto replacement = tiledResults->replacements.lookup(res)) {
          Operation *replacementOp = replacement.getDefiningOp();
          rewriter.replaceUsesWithIf(res, replacement, [&](OpOperand &use) {
            Operation *user = use.getOwner();
            return dominanceInfo.properlyDominates(replacementOp, user);
          });
        }
      }

      if (toReplace->use_empty()) {
        rewriter.eraseOp(toReplace);
      }
    }

    // Consumer fusion for scf.forall ops.
    if (fuseConsumers &&
        tilingOptions.loopType == scf::SCFTilingOptions::LoopType::ForallOp) {
      FailureOr<std::queue<Operation *>> newFusionOpportunities =
          fuseConsumersIntoForall(
              rewriter, tiledResults->tiledAndFusedOps.getArrayRef(),
              tiledResults->loops, [&tiledAndFusedOps](Operation *op) {
                return tiledAndFusedOps.contains(op);
              });
      // Allow failure here to continue the worklist.
      if (!failed(newFusionOpportunities)) {
        // Because we restrict to at most a single tilable consumer for yielding
        // a replacement, no new fusion opportunities will yield a replacement,
        // meaning there is no need to run consumer fusion again afterwards.
        // TODO: run producer and consumer fusion in one worklist.
        fuseProducersOfSlices(rewriter, *newFusionOpportunities,
                              tileAndFuseOptions, tiledResults->loops);
      }
    }

    // Coalesce scf.for loops created during reduction tiling.
    // This is done at the very end after all other transformations
    // to avoid invalidating dominance info or affecting fusion logic.
    if (tilingLevel == IREE::GPU::TilingLevel::Reduction &&
        !tiledResults->loops.empty()) {
      SmallVector<scf::ForOp> forLoops;

      // Check if tiling happened inside an existing scf.for loop
      // If so, include that parent loop in the coalescing.
      Operation *parentOp =
          tiledResults->loops.front().getOperation()->getParentOp();
      scf::ForOp parentForOp = dyn_cast<scf::ForOp>(parentOp);
      if (parentForOp) {
        forLoops.push_back(parentForOp);
      }

      // Collect all the tiled loops first.
      for (LoopLikeOpInterface loop : tiledResults->loops) {
        if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
          forLoops.push_back(forOp);
        }
      }

      // If loops have dynamic trip counts and we coalesce them, it can
      // cause range analysis to not find static bounds. This was mainly
      // noticed as a problem in applyPaddingLevel, to prevent a regression
      // we dont coalesce such loops.
      bool hasDynamicTripCount = false;
      for (scf::ForOp forOp : forLoops) {
        if (!getConstantIntValue(forOp.getLowerBound()) ||
            !getConstantIntValue(forOp.getUpperBound()) ||
            !getConstantIntValue(forOp.getStep())) {
          hasDynamicTripCount = true;
          LLVM_DEBUG(llvm::dbgs()
                     << "Skipping coalescing: loop has dynamic trip count\n");
          break;
        }
      }

      if (forLoops.size() > 1 && !hasDynamicTripCount) {
        if (failed(coalesceLoops(rewriter, forLoops))) {
          // Coalescing failure is not critical, just log and continue.
          LLVM_DEBUG(llvm::dbgs() << "Failed to coalesce reduction loops\n");
        }
      }
    }
  }
  return success();
}
} // namespace mlir::iree_compiler
