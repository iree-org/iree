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
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"

#include <cassert>

#define DEBUG_TYPE "iree-codegen-common-tile-and-fuse-utils"

namespace mlir::iree_compiler {

void fuseProducersOfSlices(RewriterBase &rewriter,
                           std::queue<Operation *> &worklist,
                           scf::SCFTileAndFuseOptions &options,
                           MutableArrayRef<LoopLikeOpInterface> loops) {
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

    // The operands of the fused producer might themselves be slices of
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

FailureOr<std::queue<Operation *>>
fuseConsumersIntoForall(RewriterBase &rewriter, Operation *tiledOp,
                        MutableArrayRef<LoopLikeOpInterface> loops,
                        std::function<bool(Operation *)> filterFn) {
  // Collect the candidate slices which can be potential consumers that can be
  // fused.
  std::queue<SmallVector<Operation *>> candidates;
  llvm::SmallDenseSet<tensor::ParallelInsertSliceOp> allCandidates;
  auto addCandidateSlices = [&candidates, &allCandidates,
                             &filterFn](Operation *fusedOp) {
    for (auto *userOp : fusedOp->getResults().getUsers()) {
      auto sliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(userOp);
      if (!sliceOp || allCandidates.contains(sliceOp)) {
        continue;
      }

      auto currLoop =
          cast<scf::ForallOp>(sliceOp->getParentOp()->getParentOp());
      OpResult loopResult = currLoop.getTiedOpResult(
          currLoop.getTiedOpOperand(cast<BlockArgument>(sliceOp.getDest())));
      SmallVector<Operation *> users = llvm::to_vector(
          llvm::make_filter_range(loopResult.getUsers(), filterFn));
      if (users.empty()) {
        continue;
      }
      mlir::computeTopologicalSorting(users);

      Operation *fusableUser = users.front();
      // Check all operands from the `scf.forall`
      SmallVector<OpResult> loopResults;
      for (OpOperand &opOperand : fusableUser->getOpOperands()) {
        if (opOperand.get().getDefiningOp() == currLoop.getOperation()) {
          loopResults.push_back(cast<OpResult>(opOperand.get()));
        }
      }

      SmallVector<Operation *> fusedSlices;
      for (OpResult result : loopResults) {
        BlockArgument tiedBlockArg =
            currLoop.getTiedBlockArgument(currLoop.getTiedOpOperand(result));
        SmallVector<tensor::ParallelInsertSliceOp> slices = llvm::map_to_vector(
            currLoop.getCombiningOps(tiedBlockArg), [](Operation *op) {
              return cast<tensor::ParallelInsertSliceOp>(op);
            });
        llvm::append_range(fusedSlices, slices);
        allCandidates.insert_range(slices);
      }
      if (!fusedSlices.empty()) {
        candidates.emplace(std::move(fusedSlices));
      }
    }
  };

  addCandidateSlices(tiledOp);

  std::queue<Operation *> newFusionOpportunities;
  while (!candidates.empty()) {
    // Traverse the slices in BFS fashion.
    SmallVector<Operation *> candidateSlices = candidates.front();
    candidates.pop();

    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedResult =
        mlir::scf::tileAndFuseConsumerOfSlices(rewriter, candidateSlices,
                                               loops);
    if (failed(fusedResult)) {
      return failure();
    }

    // Replace the original consumer operation with the tiled implementation.
    rewriter.replaceOp(fusedResult->origConsumerOperands.front()->getOwner(),
                       fusedResult->tiledOps.front());

    // The result of the fused consumers might themselves be slices of
    // values produced by operations that implement the `TilingInterface`.
    // Add these operations to the worklist.
    addCandidateSlices(
        fusedResult->tiledAndFusedConsumerOperands.front()->getOwner());

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
  llvm::SmallDenseSet<Operation *> producers;
  worklist.push_back(op);
  producers.insert(op);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (OpOperand &operand : current->getOpOperands()) {
      auto producer = operand.get().getDefiningOp<TilingInterface>();
      if (!producer || producers.contains(producer) ||
          exclude.contains(producer))
        continue;
      worklist.push_back(producer);
      producers.insert(producer);
    }
  }
  return producers;
}

LogicalResult applyTileAndFuseToEachRoot(
    RewriterBase &rewriter, llvm::SmallDenseSet<TilingInterface> &payloadOps,
    IREE::GPU::TilingLevel tilingLevel, bool allowZeroSlices,
    std::optional<
        llvm::SmallDenseMap<TilingInterface, SmallVector<OpFoldResult>>>
        targetTileMap) {
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
          tilingLevel != IREE::GPU::TilingLevel::PartialReduction)
        yieldProducerReplacement = yieldReplacementsFor.contains(owner);
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
      // Add pattern to fuse pad operations without zero slice gaurd, if we
      // know we have no zero slices.
      auto zeroSliceGuard = [](tensor::ExtractSliceOp) -> std::optional<bool> {
        // Do not use zero slice gaurd.
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
      for (OpResult res : toReplace->getResults())
        if (auto replacement = tiledResults->replacements.lookup(res)) {
          Operation *replacementOp = replacement.getDefiningOp();
          rewriter.replaceUsesWithIf(res, replacement, [&](OpOperand &use) {
            Operation *user = use.getOwner();
            return dominanceInfo.properlyDominates(replacementOp, user);
          });
        }

      if (toReplace->use_empty()) {
        rewriter.eraseOp(toReplace);
      }
    }
  }
  return success();
}
} // namespace mlir::iree_compiler
