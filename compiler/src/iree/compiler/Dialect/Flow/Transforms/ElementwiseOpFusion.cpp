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

#include "iree/compiler/Dialect/Flow/Transforms/FusionUtils.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-elementwise-op-fusion"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_ELEMENTWISEOPFUSIONPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

class ElementwiseOpFusionPass
    : public impl::ElementwiseOpFusionPassBase<ElementwiseOpFusionPass> {

public:
  using Base::Base;

  void runOnOperation() override;
};

} // namespace

// Indicates whether the given linalg op represents a transpose. In particular,
// it requires a single input where the indexing maps are full permutations and
// non-equal.
static bool isaTransposeOpInterface(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops())
    return false;

  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1)
    return false;
  auto mapRange = linalgOp.getIndexingMapsArray();
  if (!mapRange.front().isPermutation() || !mapRange.back().isPermutation() ||
      mapRange.front() == mapRange.back()) {
    return false;
  }
  return llvm::hasSingleElement(linalgOp.getBlock()->getOperations());
}

// Check if the `op` is the start of a
// `linalg.transpose -> (tensor.collapse_shape ->)? -> linalg_ext.attention`
// chain.
static bool isTransposeOfAttentionOperands(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp && !isaTransposeOpInterface(linalgOp)) {
    return false;
  }
  for (Operation *user : op->getUsers()) {
    Operation *checkOp = user;
    if (isa<tensor::CollapseShapeOp>(checkOp)) {
      if (!checkOp->hasOneUse()) {
        return false;
      }
      checkOp = *checkOp->user_begin();
    }
    if (!isa<IREE::LinalgExt::AttentionOp>(checkOp)) {
      return false;
    }
  }
  return true;
}

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

        if (!areFusableAsElementwiseOps(context, fusedOperand,
                                        fuseMultiReduction)) {
          return false;
        }

        // TODO(MaheshRavishankar): Transpose before attention
        // seems to throw fusion into a bad state. Avoid this for now
        if (isTransposeOfAttentionOperands(consumer)) {
          return false;
        }
        return true;
      };
  linalg::populateElementwiseOpsFusionPatterns(fusionPatterns,
                                               fuseElementwiseOpsControlFn);
  GreedyRewriteConfig rewriteConfig;
  rewriteConfig.maxIterations = GreedyRewriteConfig::kNoLimit;
  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), std::move(fusionPatterns), rewriteConfig))) {
    getOperation()->emitOpError("Failed to perform elementwise operations");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::IREE::Flow
