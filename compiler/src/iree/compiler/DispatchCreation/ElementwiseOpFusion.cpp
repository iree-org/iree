// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- ElementwiseOpFusion.cpp --- Pass to fuse elementwise ops --------===//
//
// This pass applies the elementwise operation fusion transformation in Linalg
// with a IREE-custom cost function.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/FusionUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-elementwise-op-fusion"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_ELEMENTWISEOPFUSIONPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

struct ElementwiseOpFusionPass final
    : public impl::ElementwiseOpFusionPassBase<ElementwiseOpFusionPass> {
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// GatherFusionPattern
//===----------------------------------------------------------------------===//

// Specific case. The linalg generic implementation of "gather"
// cannot be fused because it there is no producer-consumer
// relationship between the two generics. This is because the indexing
// is not affine (index values come from a tensor).
struct GatherFusionPattern final : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // Check if extractOp is inside a generic op
    auto consumerOp =
        dyn_cast_or_null<linalg::GenericOp>(extractOp->getParentOp());
    if (!consumerOp) {
      return rewriter.notifyMatchFailure(
          extractOp, "expected extract op to be inside a generic op");
    }

    auto producerOp = extractOp.getTensor().getDefiningOp<linalg::GenericOp>();
    if (!producerOp) {
      return rewriter.notifyMatchFailure(
          consumerOp, "expected extract operand to be a generic op");
    }

    // Check if the producerOp is fusible
    if (producerOp.getNumDpsInputs() != 1 || producerOp.getNumResults() != 1 ||
        !isElementwise(producerOp) ||
        !IREE::LinalgExt::isBitExtendOp(producerOp)) {
      return rewriter.notifyMatchFailure(producerOp,
                                         "producer op is not fusible");
    }

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(extractOp);

    // Create a new extract op that extracts from the original tensor
    // (after the original extract). Clone the producerOp's body into the
    // consumerOp, inline the cloned block (erases the block) after the new
    // extract, and clean up.
    auto newExtractOp = rewriter.create<tensor::ExtractOp>(
        extractOp.getLoc(), producerOp.getDpsInputOperand(0)->get(),
        extractOp.getIndices());
    rewriter.cloneRegionBefore(producerOp.getRegion(), consumerOp.getRegion(),
                               consumerOp.getRegion().begin());
    Block &clonedBlock = consumerOp.getRegion().front();
    auto producerTermOp = clonedBlock.getTerminator();

    rewriter.inlineBlockBefore(
        &clonedBlock, extractOp->getNextNode(),
        {newExtractOp.getResult(), newExtractOp.getResult()});

    // Replace the the all references to the original extract result with the
    // result from the inlined producerOp.
    extractOp.getResult().replaceAllUsesWith(producerTermOp->getOperand(0));
    rewriter.eraseOp(producerTermOp);
    rewriter.eraseOp(extractOp);

    return success();
  }
};

} // namespace

void ElementwiseOpFusionPass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet fusionPatterns(context);
  // Only fuse operations where all uses of the producer are generic
  // operations. If an operation is used in a named op, it will be computed
  // anyway, so the consumers can just use that value.
  linalg::ControlFusionFn fuseElementwiseOpsControlFn =
      [&](OpOperand *fusedOperand) {
        Operation *producer = fusedOperand->get().getDefiningOp();
        Operation *consumer = fusedOperand->getOwner();

        if (!IREE::Flow::isNonNullAndOutsideDispatch({producer, consumer})) {
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

        return areFusableAsElementwiseOps(context, fusedOperand,
                                          fuseMultiReduction);
      };
  linalg::populateElementwiseOpsFusionPatterns(fusionPatterns,
                                               fuseElementwiseOpsControlFn);

  linalg::ControlFusionFn foldTransposeControlFn = [](OpOperand *fusedOperand) {
    Operation *producer = fusedOperand->get().getDefiningOp();
    Operation *consumer = fusedOperand->getOwner();

    return IREE::Flow::isNonNullAndOutsideDispatch({producer, consumer});
  };
  IREE::LinalgExt::populateFuseLinalgExtOpsWithTransposes(
      fusionPatterns, foldTransposeControlFn);
  fusionPatterns.insert<GatherFusionPattern>(context);

  GreedyRewriteConfig rewriteConfig;
  rewriteConfig.maxIterations = GreedyRewriteConfig::kNoLimit;
  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), std::move(fusionPatterns), rewriteConfig))) {
    getOperation()->emitOpError("Failed to perform elementwise operations");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
