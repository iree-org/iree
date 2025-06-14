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
} // namespace

template <typename T>
static SmallVector<T> applyProjectedPermutation(const SmallVectorImpl<T> &input,
                                                ArrayRef<int64_t> projPerm) {
  SmallVector<T> result;
  result.reserve(projPerm.size());
  for (int64_t idx : projPerm) {
    result.push_back(input[idx]);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// GatherFusionPattern
//===----------------------------------------------------------------------===//

// Specific case. The linalg generic implementation of "gather"
// cannot be fused because it there is no producer-consumer
// relationship between the two generics. This is because the indexing
// is not affine (index values come from a tensor).
namespace {
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
    if (producerOp.getNumResults() != 1 || !isElementwise(producerOp) ||
        !IREE::LinalgExt::isBitExtendOp(producerOp)) {
      return rewriter.notifyMatchFailure(producerOp,
                                         "producer op is not fusible");
    }

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(extractOp);

    auto result = cast<OpResult>(extractOp.getTensor());
    auto resultMap = producerOp.getIndexingMapMatchingResult(result);
    SmallVector<Value> extractOps;
    for (OpOperand &operand : producerOp->getOpOperands()) {
      auto inputMap = producerOp.getMatchingIndexingMap(&operand);
      auto composedMap = inputMap.compose(inversePermutation(resultMap));
      auto perm = llvm::map_to_vector<4>(
          composedMap.getResults(), [](AffineExpr expr) -> int64_t {
            return cast<AffineDimExpr>(expr).getPosition();
          });
      SmallVector<Value, 4> indices = extractOp.getIndices();
      indices = applyProjectedPermutation(indices, perm);
      auto newExtract = rewriter.create<tensor::ExtractOp>(
          extractOp.getLoc(), operand.get(), indices);
      extractOps.push_back(newExtract);
    }
    rewriter.cloneRegionBefore(producerOp.getRegion(), consumerOp.getRegion(),
                               consumerOp.getRegion().begin());
    Block &clonedBlock = consumerOp.getRegion().front();

    // Replace `linalg.index` ops with the value of the index from `indices`.
    SmallVector<Value, 4> indices = extractOp.getIndices();
    indices = applyPermutationMap(resultMap, ArrayRef(indices));
    SmallVector<linalg::IndexOp> indexOps(
        clonedBlock.getOps<linalg::IndexOp>());
    for (linalg::IndexOp indexOp : indexOps) {
      rewriter.replaceOp(indexOp, indices[indexOp.getDim()]);
    }
    auto producerTermOp = clonedBlock.getTerminator();

    rewriter.inlineBlockBefore(&clonedBlock, extractOp->getNextNode(),
                               extractOps);

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

  RewritePatternSet linalgFusionPatterns(context);
  linalg::populateElementwiseOpsFusionPatterns(linalgFusionPatterns,
                                               fuseElementwiseOpsControlFn);

  GreedyRewriteConfig rewriteConfig;
  rewriteConfig.setMaxIterations(GreedyRewriteConfig::kNoLimit);
  if (failed(applyPatternsGreedily(
          getOperation(), std::move(linalgFusionPatterns), rewriteConfig))) {
    getOperation()->emitOpError(
        "Failed to fuse elementwise ops with upstream patterns.");
    return signalPassFailure();
  }

  // Try fuse with linalgExt patterns.
  linalg::ControlFusionFn foldTransposeControlFn = [](OpOperand *fusedOperand) {
    Operation *producer = fusedOperand->get().getDefiningOp();
    Operation *consumer = fusedOperand->getOwner();

    return IREE::Flow::isNonNullAndOutsideDispatch({producer, consumer});
  };
  RewritePatternSet linalgExtFusionPatterns(context);
  IREE::LinalgExt::populateFuseLinalgExtOpsWithTransposes(
      linalgExtFusionPatterns, foldTransposeControlFn);
  linalgExtFusionPatterns.insert<GatherFusionPattern>(context);
  if (failed(applyPatternsGreedily(
          getOperation(), std::move(linalgExtFusionPatterns), rewriteConfig))) {
    getOperation()->emitOpError(
        "Failed to fuse elementwise ops with linalgExt patterns.");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
