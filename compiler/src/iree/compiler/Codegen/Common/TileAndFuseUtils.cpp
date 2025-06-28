// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <cassert>

#define DEBUG_TYPE "iree-codegen-common-tile-and-fuse-utils"

namespace mlir::iree_compiler {

#define CEILDIV(a, b) ((a + b - 1) / b)

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

bool warForConsumerFusionSSAViolation(
    Operation *rootOp,
    const llvm::SmallDenseSet<Operation *> &tiledAndFusedOps) {
  auto linalgRootOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgRootOp) {
    return false;
  }
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgRootOp.getIteratorTypesArray();
  for (AffineMap map :
       llvm::map_range(linalgRootOp.getIndexingMaps(), [](Attribute attr) {
         return cast<AffineMapAttr>(attr).getValue();
       })) {
    if (!compressUnusedDims(map).isIdentity()) {
      return false;
    }
  }

  for (OpOperand &use : linalgRootOp->getUses()) {
    auto linalgUser = dyn_cast<linalg::LinalgOp>(use.getOwner());
    if (!linalgUser) {
      return false;
    }
    if (!linalgUser.getMatchingIndexingMap(&use).isIdentity()) {
      return false;
    }
  }
  return true;
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
fuseConsumersIntoLoops(RewriterBase &rewriter, Operation *tiledOp,
                       MutableArrayRef<LoopLikeOpInterface> loops,
                       bool useWARForConsumerFusionSSAViolation) {
  auto addCandidateSlices = [](Operation *fusedOp,
                               std::queue<Operation *> &candidates) {
    for (auto *userOp : fusedOp->getResults().getUsers()) {
      if (llvm::isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(
              userOp)) {
        // Users of tiledOp should either be all of type `tensor.insert_slice`
        // or all of`tensor.parallel_insert_slice`.
        //
        // Pattern 1 - tileing with scf.for:
        //   %out = scf.for ... {
        //     %0 = scf.for ... {
        //       %t0 = op
        //       %t1 = op  %t0                 // <- `tiledOp`
        //       %1 = tensor.insert_slice %t1
        //       yield %1
        //     }
        //     yield %0
        //   }
        //
        // Pattern 2 - tiling with scf.forall:
        //   % out = scf.forall ... {
        //       %t0 = op
        //       %t1 = op  %t0                 // <- `tiledOp`
        //       scf.forall.in_parallel {
        //         tensor.parallel_insert_slice %tile
        //       }
        //   }
        assert((candidates.empty() ||
                candidates.front()->getName() == userOp->getName()) &&
               "expected all slice users to be of type tensor.insert_slice "
               "or of tensor.parallel_insert_slice.");
        candidates.push(userOp);
      }
    }
  };

  // Collect the candidate slices which can be potential consumers that can be
  // fused.
  std::queue<Operation *> candidates;
  addCandidateSlices(tiledOp, candidates);

  std::queue<Operation *> newFusionOpportunities;
  while (!candidates.empty()) {
    // Traverse the slices in BFS fashion.
    Operation *candidateSliceOp = candidates.front();
    candidates.pop();

    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedResult =
        mlir::scf::tileAndFuseConsumerOfSlices(rewriter, candidateSliceOp,
                                               loops);
    if (failed(fusedResult)) {
      LLVM_DEBUG(llvm::dbgs() << "failed to fuse consumer of slice: "
                              << candidateSliceOp << "\n");
      continue;
    }

    // Implement the WAR for consumer fusion SSA violation (as described in the
    // comments for `warForConsumerFusionSSAViolation`)
    if (useWARForConsumerFusionSSAViolation) {
      for (auto [tiledOpResult, loopResult] :
           llvm::zip(tiledOp->getResults(), loops.back()->getResults())) {
        for (OpOperand &use : loopResult.getUses()) {
          Operation *user = use.getOwner();
          if (user->getParentOp() != loops.back()) {
            continue;
          }
          auto slice = dyn_cast<tensor::ExtractSliceOp>(user);
          if (!slice) {
            return failure();
          }
          rewriter.replaceAllOpUsesWith(slice, tiledOpResult);
        }
      }
    }

    // Replace the original consumer operation with the tiled implementation.
    rewriter.replaceOp(fusedResult->origConsumerOperands.front()->getOwner(),
                       fusedResult->tiledOps.front());

    // The result of the fused consumers might themselves be slices of
    // values produced by operations that implement the `TilingInterface`.
    // Add these operations to the worklist.
    addCandidateSlices(
        fusedResult->tiledAndFusedConsumerOperands.front()->getOwner(),
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
    }
  }
  return newFusionOpportunities;
}

FailureOr<TilingInfo> getTiledAndDistributionInfo(RewriterBase &rewriter,
                                                  Operation *tilableOp) {
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

SmallVector<Attribute>
getDistributionMapping(MLIRContext *context, ArrayRef<OpFoldResult> tileSizes) {
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

bool areAllStaticLoopBounds(scf::ForallOp forallOp) {
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

static SmallVector<Attribute>
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

FailureOr<scf::ForallOp> dropUnitDistributedDims(RewriterBase &rewriter,
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

} // namespace mlir::iree_compiler
