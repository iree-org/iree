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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-fusion-of-tensor-ops"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

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
      // For now check that the `sourceOp` is only used once in the consumer.
      // This can be generalized if needed
      unsigned numUsesOfOp = 0;
      for (OpOperand &operand : sourceOp->getOpOperands()) {
        if (operand.get().getDefiningOp() == op) numUsesOfOp++;
      }
      if (numUsesOfOp != 1) return std::nullopt;
      return &source;
    }
  }
  return std::nullopt;
}

/// Check if the producer generic op is fusable with the consumer generic op.
static bool areFusableOps(MLIRContext *context, OpOperand *fusedOperand) {
  Operation *producerOp = fusedOperand->get().getDefiningOp();
  Operation *consumerOp = fusedOperand->getOwner();
  if (!producerOp) return false;

  // Check for i1 return types, if so aggressively fuse to avoid `i1` buffers.
  if (llvm::all_of(producerOp->getResultTypes(), [](Type t) {
        if (t.isInteger(1)) return true;
        if (auto shapedType = t.dyn_cast<ShapedType>()) {
          if (shapedType.getElementType().isInteger(1)) return true;
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
  if (std::begin(body)->hasTrait<OpTrait::IsTerminator>()) return true;

  // If producer does not have a single user, dont fuse.
  if (!producerOp->hasOneUse()) return false;

  // If the producer has a single use (this op), only fuse if
  // - 1) The consumer op is all parallel loops. The parallelism of the consumer
  //      can be used as a way to amortize cost of redundant computation
  // - 2) If consumer op is a reduction, only fuse if the indexing map in the
  //      consumer for the producer result is a permutation. If it is a
  //      broadcast this ends up redundantly computing operations without more
  //      parallelism.
  if (auto linalgConsumerOp = dyn_cast<linalg::LinalgOp>(consumerOp)) {
    return linalgConsumerOp.getNumParallelLoops() ==
               linalgConsumerOp.getNumLoops() ||
           linalgConsumerOp.getMatchingIndexingMap(fusedOperand)
               .isPermutation();
  }

  // All other cases dont fuse.
  return false;
}

namespace {

struct FuseElementwiseOpsWithMultipleUses
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  static const char *getConsumerAttributeName() {
    return "__fusable_conumer__";
  }
  static const char *getProducerAttributeName() {
    return "__fusable_producer__";
  }

  LogicalResult matchAndRewrite(linalg::GenericOp consumerOp,
                                PatternRewriter &rewriter) const override {
    auto consumerMarker =
        consumerOp->getAttrOfType<IntegerAttr>(getConsumerAttributeName());
    if (!consumerMarker) return failure();

    auto fusedOperandIt =
        llvm::find_if(consumerOp->getOpOperands(), [&](OpOperand &operand) {
          Operation *operandProducer = operand.get().getDefiningOp();
          if (!operandProducer) return false;
          auto producerMarker = operandProducer->getAttrOfType<IntegerAttr>(
              getProducerAttributeName());
          if (!producerMarker) return false;
          return consumerMarker.getValue() == producerMarker.getValue();
        });
    assert(fusedOperandIt != consumerOp->getOpOperands().end() &&
           "expected to find the fusable producer");
    OpOperand *fusedOperand = fusedOperandIt;
    assert(linalg::areElementwiseOpsFusable(fusedOperand) &&
           "expected producer and consumer to be fusable");
    Operation *producerOp = fusedOperand->get().getDefiningOp();

    // Cleanup the markers.
    consumerOp->removeAttr(getConsumerAttributeName());
    producerOp->removeAttr(getProducerAttributeName());

    FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
        linalg::fuseElementwiseOps(rewriter, fusedOperand);
    if (failed(fusionResult)) {
      return rewriter.notifyMatchFailure(consumerOp,
                                         "failed to fuse with producer");
    }
    for (auto replacement : fusionResult->replacements) {
      rewriter.replaceUsesWithIf(
          replacement.first, replacement.second,
          [&](OpOperand &use) { return use.getOwner() != consumerOp; });
    }
    return success();
  }
};

static FailureOr<unsigned> fuseMultiUseProducers(Operation *funcOp,
                                                 MLIRContext *context,
                                                 DominanceInfo &dominanceInfo) {
  // Try fusion of operations when producer has multiple uses.
  // 1. Walk the function in pre-order.
  // 2. Check if a `linalg.generic` op has a consumer `linalg.generic` op
  //    that dominates all uses of the producer op. Then fuse the producer
  //    consumer
  unsigned numCandidates = 0;
  OpBuilder builder(context);
  funcOp->walk<WalkOrder::PreOrder>([&](linalg::GenericOp genericOp) {
    auto consumerAttrName =
        FuseElementwiseOpsWithMultipleUses::getConsumerAttributeName();
    auto producerAttrName =
        FuseElementwiseOpsWithMultipleUses::getProducerAttributeName();
    if (genericOp->hasAttr(consumerAttrName) ||
        genericOp->hasAttr(producerAttrName)) {
      return;
    }

    std::optional<OpOperand *> fusableUse =
        getFusableUse(genericOp, dominanceInfo);
    if (!fusableUse) return;
    if (!linalg::areElementwiseOpsFusable(fusableUse.value())) return;

    auto consumer = dyn_cast<linalg::GenericOp>(fusableUse.value()->getOwner());
    auto isParallelIteratorType = [](Attribute attr) {
      return linalg::isParallelIterator(
          attr.cast<linalg::IteratorTypeAttr>().getValue());
    };
    if (!consumer ||
        !(llvm::all_of(genericOp.getIteratorTypes(), isParallelIteratorType) &&
          llvm::all_of(consumer.getIteratorTypes(), isParallelIteratorType))) {
      return;
    }

    genericOp->setAttr(producerAttrName,
                       builder.getI64IntegerAttr(numCandidates));
    consumer->setAttr(consumerAttrName,
                      builder.getI64IntegerAttr(numCandidates));
    numCandidates++;
    return;
  });
  LLVM_DEBUG({
    llvm::dbgs() << "Num of multiuse fusable candidates : " << numCandidates
                 << "\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
  });
  RewritePatternSet fusionPatterns(context);
  fusionPatterns.insert<FuseElementwiseOpsWithMultipleUses>(context);
  linalg::GenericOp::getCanonicalizationPatterns(fusionPatterns, context);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(fusionPatterns)))) {
    return funcOp->emitOpError("multi use producer -> consumer fusion failed");
  }
  return numCandidates;
}

/// Pass to fuse linalg on tensor operations as well as fusion of hal.interface*
/// operations with linalg.tensor_reshape operation.
struct FusionOfTensorOpsPass
    : public FusionOfTensorOpsBase<FusionOfTensorOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    math::MathDialect>();
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
            if (!producer) return false;
            Operation *consumer = fusedOperand->getOwner();

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
            if (operands.size() >= kIreeMaxOperandCount) return false;

            return areFusableOps(context, fusedOperand);
          };
      linalg::populateElementwiseOpsFusionPatterns(fusionPatterns,
                                                   fuseElementwiseOpsControlFn);

      // Always fold reshape by expansion.
      linalg::ControlFusionFn fuseByExpansionControlFn =
          [](OpOperand *fusedOperand) {
            Operation *producer = fusedOperand->get().getDefiningOp();
            if (!producer) {
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
        auto producer = fusedOperand->get().getDefiningOp();
        return producer && producer->hasOneUse();
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
      memref::populateResolveRankedShapeTypeResultDimsPatterns(fusionPatterns);

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
            auto producer = fusedOperand->get().getDefiningOp();
            if (!producer) {
              return false;
            }

            auto reshapeOp = dyn_cast<tensor::ExpandShapeOp>(producer);
            if (!reshapeOp) return true;

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
      memref::populateResolveRankedShapeTypeResultDimsPatterns(
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

    if (fuseMultiUse) {
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
        if (numOfFusableCandidates.value() == 0) break;
      }
    }
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFusionOfTensorOpsPass(bool fuseMultiUse,
                            unsigned multiUseFusionIteration) {
  return std::make_unique<FusionOfTensorOpsPass>(fuseMultiUse,
                                                 multiUseFusionIteration);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
