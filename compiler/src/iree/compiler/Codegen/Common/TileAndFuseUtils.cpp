// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

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
} // namespace mlir::iree_compiler
