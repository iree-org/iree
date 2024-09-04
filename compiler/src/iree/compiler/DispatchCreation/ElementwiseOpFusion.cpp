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

  GreedyRewriteConfig rewriteConfig;
  rewriteConfig.maxIterations = GreedyRewriteConfig::kNoLimit;
  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), std::move(fusionPatterns), rewriteConfig))) {
    getOperation()->emitOpError("Failed to perform elementwise operations");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
