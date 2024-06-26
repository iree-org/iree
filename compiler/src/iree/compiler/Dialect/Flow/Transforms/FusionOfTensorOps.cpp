// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- FusionOfTensorsOps.cpp - Pass to fuse operations on tensors-------===//
//
// Pass to fuse operations on tensors after conversion to Linalg. Uses the
// patterns from MLIR for fusion linalg operations on tensors, and a few
// patterns to fuse these with IREE specific operations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-fusion-of-tensor-ops"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_FUSIONOFTENSOROPSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

// TODO: Remove this and the backing code once consteval is beyond being
// rolled back.
static llvm::cl::opt<int64_t> clLinalgMaxConstantFoldElements(
    "iree-codegen-linalg-max-constant-fold-elements",
    llvm::cl::desc("Maximum number of elements to try to constant fold."),
    llvm::cl::init(0));

/// Check if any of the use dominates all other uses of the operation.
static std::optional<OpOperand *> getFusableUse(Operation *op,
                                                DominanceInfo &dominanceInfo) {
  auto uses = op->getUses();
  for (OpOperand &source : uses) {
    Operation *sourceOp = source.getOwner();
    bool dominatesAllUsers = true;
    for (OpOperand &target : uses) {
      Operation *targetOp = target.getOwner();
      if (!dominanceInfo.dominates(sourceOp, targetOp)) {
        dominatesAllUsers = false;
        break;
      }
    }
    if (dominatesAllUsers) {
      return &source;
    }
  }
  return std::nullopt;
}

static OpOperand *getFirstUseInConsumer(Operation *producer,
                                        Operation *consumer) {
  for (OpOperand &opOperand : consumer->getOpOperands()) {
    if (opOperand.get().getDefiningOp() == producer) {
      return &opOperand;
    }
  }
  return nullptr;
}

static SmallVector<OpOperand *> getAllUsesInConsumer(Operation *producer,
                                                     Operation *consumer) {
  SmallVector<OpOperand *> allUses;
  for (OpOperand &opOperand : consumer->getOpOperands()) {
    if (opOperand.get().getDefiningOp() == producer) {
      allUses.push_back(&opOperand);
    }
  }
  return allUses;
}

/// Perform the fusion of `rootOp` with all the operations in `fusableOps`
/// using elementwise fusion.
static LogicalResult doMultiUseFusion(Operation *rootOp,
                                      llvm::SetVector<Operation *> &fusableOps,
                                      RewriterBase &rewriter) {
  assert(rootOp && "root op cant be null");

  LLVM_DEBUG({
    llvm::dbgs() << "Fusion root : \n";
    rootOp->print(llvm::dbgs());
    llvm::dbgs() << "\nFused with :";

    for (auto producer : fusableOps) {
      llvm::dbgs() << "\t";
      producer->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  SmallVector<Operation *> fusedOpsVec = llvm::to_vector(fusableOps);
  mlir::computeTopologicalSorting(fusedOpsVec);

  Operation *consumerOp = rootOp;
  OpBuilder::InsertionGuard g(rewriter);
  for (Operation *producerOp : llvm::reverse(fusedOpsVec)) {
    // Fuse all uses from producer -> consumer. It has been checked
    // before that all uses are fusable.
    while (OpOperand *fusedOperand =
               getFirstUseInConsumer(producerOp, consumerOp)) {
      rewriter.setInsertionPoint(consumerOp);
      FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
          linalg::fuseElementwiseOps(rewriter, fusedOperand);
      if (failed(fusionResult)) {
        return rewriter.notifyMatchFailure(consumerOp,
                                           "failed to fuse with producer");
      }
      for (auto replacement : fusionResult->replacements) {
        rewriter.replaceUsesWithIf(
            replacement.first, replacement.second, [&](OpOperand &use) {
              return use.getOwner() != fusionResult->fusedOp &&
                     fusableOps.count(use.getOwner()) == 0;
            });
      }
      consumerOp = fusionResult->fusedOp;
      if (failed(cast<linalg::GenericOp>(consumerOp).verify())) {
        return consumerOp->emitOpError("failed to verify op");
      }
    }
  }
  return success();
}

static FailureOr<unsigned> fuseMultiUseProducers(Operation *funcOp,
                                                 MLIRContext *context,
                                                 DominanceInfo &dominanceInfo) {
  OpBuilder builder(context);
  llvm::MapVector<Operation *, llvm::SetVector<Operation *>> fusedOps;
  DenseMap<Operation *, Operation *> opToRootMap;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](linalg::GenericOp genericOp) {
        if (!isNonNullAndOutsideDispatch(genericOp)) {
          return;
        }

        // 1. Only look at all parallel consumers.
        if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
          return;
        }

        // Dequantization-like operations should be fused with consumers to keep
        // the smaller bit width on the dispatch boundary.
        if (isBitExtendOp(genericOp)) {
          return;
        }

        Operation *fusableProducer = nullptr;
        for (OpOperand &operand : genericOp->getOpOperands()) {
          // 2. Only fuse with `linalg.generic` producers that arent
          //    already part of another fusion group.
          auto producer = dyn_cast_or_null<linalg::GenericOp>(
              operand.get().getDefiningOp());
          if (!producer || opToRootMap.count(producer)) {
            continue;
          }

          // 3. For now do not fuse with ops in another block.
          if (producer->getBlock() != genericOp->getBlock()) {
            continue;
          }

          // 4. Basic fusability checks.
          if (!linalg::areElementwiseOpsFusable(&operand)) {
            continue;
          }

          // 5. Only consider all parallel `producer` with same iteration space
          //    as the consumer.
          if (producer.getNumLoops() != producer.getNumParallelLoops() ||
              genericOp.getNumLoops() != producer.getNumLoops()) {
            continue;
          }

          // 6. Check that the `genericOp` dominates all uses of `producer`.
          std::optional<OpOperand *> fusableUse =
              getFusableUse(producer, dominanceInfo);
          if (!fusableUse || fusableUse.value()->getOwner() != genericOp) {
            continue;
          }

          // 7. Skip dequantization-like `producer` ops as we would rather fuse
          //    by cloning the producer instead of multi-use fusion.
          if (isBitExtendOp(producer)) {
            return;
          }

          // 8. All uses from `producer` -> `consumer` need to be fusable.
          //    Without this the `producer` is still live, and there is no
          //    advantage to do the fusion.
          if (llvm::any_of(getAllUsesInConsumer(producer, genericOp),
                           [&](OpOperand *use) {
                             return !linalg::areElementwiseOpsFusable(use);
                           })) {
            continue;
          }

          fusableProducer = producer;
          break;
        }
        if (!fusableProducer)
          return;

        // If the `genericOp` is already part of a fusion group, just add the
        // the `fusableProducer` to the same group.
        llvm::SetVector<Operation *> &fusedOpSet = fusedOps[genericOp];
        fusedOpSet.insert(fusableProducer);
        opToRootMap[fusableProducer] = genericOp;
        return;
      });

  if (fusedOps.empty()) {
    return 0;
  }

  IRRewriter rewriter(context);
  for (auto it = fusedOps.rbegin(), ie = fusedOps.rend(); it != ie; ++it) {
    if (failed(doMultiUseFusion(it->first, it->second, rewriter))) {
      return funcOp->emitOpError("failed multi use fusion");
    }
  }

  RewritePatternSet fusionPatterns(context);
  linalg::populateEraseUnusedOperandsAndResultsPatterns(fusionPatterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(fusionPatterns)))) {
    return funcOp->emitOpError("multi use producer -> consumer fusion failed");
  }
  return fusedOps.size();
}

namespace {

/// Pass to fuse linalg on tensor operations as well as fusion of hal.interface*
/// operations with linalg.tensor_reshape operation.
struct FusionOfTensorOpsPass
    : public IREE::Flow::impl::FusionOfTensorOpsPassBase<
          FusionOfTensorOpsPass> {
  using IREE::Flow::impl::FusionOfTensorOpsPassBase<
      FusionOfTensorOpsPass>::FusionOfTensorOpsPassBase;
  void runOnOperation() override;
};

} // namespace

void FusionOfTensorOpsPass::runOnOperation() {
  Operation *funcOp = getOperation();
  MLIRContext *context = funcOp->getContext();

  // Run fusion of producer with consumer when producer has multiple uses.
  // For now run this sequence a fixed times (2 by default). Ideally we
  // would run it till no candidates exist.
  for (auto i : llvm::seq<unsigned>(0, multiUseFusionIteration)) {
    (void)i;
    auto &dominanceInfo = getAnalysis<DominanceInfo>();
    FailureOr<unsigned> numOfFusableCandidates =
        fuseMultiUseProducers(funcOp, context, dominanceInfo);
    if (failed(numOfFusableCandidates)) {
      funcOp->emitError("failed to fuse multi-use producers");
      return signalPassFailure();
    }
    if (numOfFusableCandidates.value() == 0)
      break;
  }
}

} // namespace mlir::iree_compiler::IREE::Flow
