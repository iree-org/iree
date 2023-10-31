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

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

#define DEBUG_TYPE "iree-flow-fusion-of-tensor-ops"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

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

/// Check if the producer generic op is fusable with the consumer generic op.
static bool areFusableOps(MLIRContext *context, OpOperand *fusedOperand) {
  Operation *producerOp = fusedOperand->get().getDefiningOp();
  Operation *consumerOp = fusedOperand->getOwner();
  if (!producerOp)
    return false;

  // Check for i1 return types, if so aggressively fuse to avoid `i1` buffers.
  if (llvm::all_of(producerOp->getResultTypes(), [](Type t) {
        if (t.isInteger(1))
          return true;
        if (auto shapedType = llvm::dyn_cast<ShapedType>(t)) {
          if (shapedType.getElementType().isInteger(1))
            return true;
        }
        return false;
      })) {
    return true;
  }

  // Don't fuse if all of the consumer maps aren't projected permutations.
  if (auto linalgConsumerOp = dyn_cast<linalg::LinalgOp>(consumerOp)) {
    if (!llvm::all_of(
            linalgConsumerOp.getIndexingMapsArray(),
            [](AffineMap map) { return map.isProjectedPermutation(); })) {
      return false;
    }
  }

  // If the generic op is "just" copy, then fuse always.
  Block &body = producerOp->getRegion(0).front();
  if (std::begin(body)->hasTrait<OpTrait::IsTerminator>())
    return true;
  if (llvm::all_of(body.getArguments(),
                   [](BlockArgument arg) { return arg.use_empty(); })) {
    // THe operands arent used, its just an `linalg.index` op.
    return true;
  }

  // If producer does not have a single user, dont fuse.
  if (!producerOp->hasOneUse())
    return false;

  // If the producer has a single use (this op), only fuse if
  // - 1) The consumer op is all parallel loops. The parallelism of the consumer
  //      can be used as a way to amortize cost of redundant computation
  // - 2) If consumer op is a reduction, only fuse if the indexing map in the
  //      consumer for the producer result is a permutation. If it is a
  //      broadcast this ends up redundantly computing operations without more
  //      parallelism.
  if (auto linalgConsumerOp = dyn_cast<linalg::LinalgOp>(consumerOp)) {
    if (linalgConsumerOp.getNumParallelLoops() ==
        linalgConsumerOp.getNumLoops()) {
      return true;
    }
    if (linalgConsumerOp.getNumReductionLoops() != 1 ||
        !linalgConsumerOp.getMatchingIndexingMap(fusedOperand)
             .isPermutation()) {
      return false;
    }
    return true;
  }

  // All other cases dont fuse.
  return false;
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

          // 7. All uses from `producer` -> `consumer` need to be fusable.
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
    : public FusionOfTensorOpsBase<FusionOfTensorOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    linalg::LinalgDialect, math::MathDialect>();
  }
  FusionOfTensorOpsPass(bool fuseMultiUse, unsigned multiUseFusionIteration) {
    this->fuseMultiUse = fuseMultiUse;
    this->multiUseFusionIteration = multiUseFusionIteration;
  }
  FusionOfTensorOpsPass(const FusionOfTensorOpsPass &pass)
      : FusionOfTensorOpsPass(pass.fuseMultiUse, pass.multiUseFusionIteration) {
  }

  void runOnOperation() override {
    Operation *funcOp = getOperation();
    MLIRContext *context = funcOp->getContext();

    {
      RewritePatternSet fusionPatterns(&getContext());
      // Only fuse operations where all uses of the producer are generic
      // operations. If an operation is used in a named op, it will be computed
      // anyway, so the consumers can just use that value.
      linalg::ControlFusionFn fuseElementwiseOpsControlFn =
          [&](OpOperand *fusedOperand) {
            Operation *producer = fusedOperand->get().getDefiningOp();
            Operation *consumer = fusedOperand->getOwner();

            if (!isNonNullAndOutsideDispatch({producer, consumer})) {
              return false;
            }

            // Limit the number of operands. We have hard limit (32) of bindings
            // passing down to HAL. Set the number to be as same as the limit --
            // IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT.
            constexpr int64_t kIreeMaxOperandCount = 32;
            DenseSet<Value> operands;
            operands.insert(producer->operand_begin(), producer->operand_end());
            operands.insert(consumer->operand_begin(),
                            std::next(consumer->operand_begin(),
                                      fusedOperand->getOperandNumber()));
            operands.insert(std::next(consumer->operand_begin(),
                                      fusedOperand->getOperandNumber() + 1),
                            consumer->operand_end());
            if (operands.size() >= kIreeMaxOperandCount)
              return false;

            return areFusableOps(context, fusedOperand);
          };
      linalg::populateElementwiseOpsFusionPatterns(fusionPatterns,
                                                   fuseElementwiseOpsControlFn);

      // Always fold reshape by expansion.
      linalg::ControlFusionFn fuseByExpansionControlFn =
          [](OpOperand *fusedOperand) {
            Operation *producer = fusedOperand->get().getDefiningOp();
            Operation *consumer = fusedOperand->getOwner();
            if (!isNonNullAndOutsideDispatch({producer, consumer})) {
              return false;
            }

            // Do not fuse producer generic op if it has more than one user.
            if (auto producerGenericOp =
                    dyn_cast<linalg::GenericOp>(producer)) {
              return producerGenericOp->hasOneUse();
            }
            // Fuse in all other cases.
            return true;
          };
      linalg::populateFoldReshapeOpsByExpansionPatterns(
          fusionPatterns, fuseByExpansionControlFn);

      // Constant fold Linalg operations.
      auto constantFoldControlFn = [](OpOperand *fusedOperand) {
        Operation *producer = fusedOperand->get().getDefiningOp();
        Operation *consumer = fusedOperand->getOwner();
        if (!isNonNullAndOutsideDispatch({producer, consumer})) {
          return false;
        }
        if (auto shapedType =
                dyn_cast<ShapedType>(fusedOperand->get().getType())) {
          if (shapedType.hasStaticShape() &&
              shapedType.getNumElements() > clLinalgMaxConstantFoldElements) {
            return false;
          }
        }
        return producer->hasOneUse();
      };
      linalg::populateConstantFoldLinalgOperations(fusionPatterns,
                                                   constantFoldControlFn);

      affine::AffineApplyOp::getCanonicalizationPatterns(fusionPatterns,
                                                         context);
      linalg::GenericOp::getCanonicalizationPatterns(fusionPatterns, context);
      tensor::ExpandShapeOp::getCanonicalizationPatterns(fusionPatterns,
                                                         context);
      tensor::populateFoldTensorEmptyPatterns(fusionPatterns);
      tensor::CollapseShapeOp::getCanonicalizationPatterns(fusionPatterns,
                                                           context);
      context->getLoadedDialect<linalg::LinalgDialect>()
          ->getCanonicalizationPatterns(fusionPatterns);
      memref::populateResolveRankedShapedTypeResultDimsPatterns(fusionPatterns);

      GreedyRewriteConfig rewriteConfig;
      rewriteConfig.maxIterations = GreedyRewriteConfig::kNoLimit;
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(fusionPatterns),
                                              rewriteConfig))) {
        funcOp->emitError("failed to apply fusion patterns");
        return signalPassFailure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "\n--- After first fixed point ---\n";
        funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    {
      // For fusion by collapsing, do so if the reshape is blocking tile and
      // fuse.
      linalg::ControlFusionFn fuseByCollapsingControlFn =
          [](OpOperand *fusedOperand) {
            Operation *producer = fusedOperand->get().getDefiningOp();
            Operation *consumer = fusedOperand->getOwner();
            if (!isNonNullAndOutsideDispatch({producer, consumer})) {
              return false;
            }

            auto reshapeOp = dyn_cast<tensor::ExpandShapeOp>(producer);
            if (!reshapeOp)
              return true;

            return reshapeOp.getSrc().getDefiningOp<linalg::LinalgOp>() !=
                   nullptr;
          };

      RewritePatternSet collapsingReshapePatterns(&getContext());
      linalg::populateFoldReshapeOpsByCollapsingPatterns(
          collapsingReshapePatterns, fuseByCollapsingControlFn);
      tensor::CollapseShapeOp::getCanonicalizationPatterns(
          collapsingReshapePatterns, context);
      tensor::ExpandShapeOp::getCanonicalizationPatterns(
          collapsingReshapePatterns, context);
      tensor::populateFoldTensorEmptyPatterns(collapsingReshapePatterns);
      memref::populateResolveRankedShapedTypeResultDimsPatterns(
          collapsingReshapePatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(collapsingReshapePatterns)))) {
        funcOp->emitError("failed to apply collapsing reshape patterns");
        return signalPassFailure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "\n--- After second fixed point ---\n";
        funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    // Run some patterns that fold away a few operations.
    {
      RewritePatternSet opFoldingPatterns(&getContext());
      tensor::populateFoldTensorEmptyPatterns(opFoldingPatterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                              std::move(opFoldingPatterns)))) {
        funcOp->emitError("failed to apply op folding patterns");
        return signalPassFailure();
      }
    }

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
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFusionOfTensorOpsPass(bool fuseMultiUse,
                            unsigned multiUseFusionIteration) {
  return std::make_unique<FusionOfTensorOpsPass>(fuseMultiUse,
                                                 multiUseFusionIteration);
}

} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
