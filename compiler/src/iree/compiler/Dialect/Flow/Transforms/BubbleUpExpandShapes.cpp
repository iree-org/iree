// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- BubbleExpandShapes.cpp --- Pass to propagate expand shapes op up -===//
//
// This pass propagates expand_shape operations up the program (and conversely)
// sinks the collapse_shape operations down the program to get the elementwise
// operations into higher dimensionality to get better fusion.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/FusionUtils.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-bubble-up-expand-shapes"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_BUBBLEUPEXPANDSHAPESPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

class BubbleUpExpandShapesPass
    : public impl::BubbleUpExpandShapesPassBase<BubbleUpExpandShapesPass> {
public:
  using Base::Base;

  void runOnOperation() override;
};

} // namespace

void BubbleUpExpandShapesPass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet bubbleExpandShapePatterns(context);
  linalg::ControlFusionFn bubbleUpExpansionControlFn =
      [](OpOperand *fusedOperand) {
        Operation *producer = fusedOperand->get().getDefiningOp();
        Operation *consumer = fusedOperand->getOwner();
        if (!isNonNullAndOutsideDispatch({producer, consumer})) {
          return false;
        }

        // Do not fuse by expand if consumer is dequant.
        if (isBitExtendOp(consumer)) {
          return false;
        }

        // Do not fuse producer generic op if it has more than one user
        // or any reduction iterators.
        if (auto producerGenericOp = dyn_cast<linalg::GenericOp>(producer)) {
          return producerGenericOp->hasOneUse() &&
                 llvm::all_of(producerGenericOp.getIteratorTypesArray(),
                              linalg::isParallelIterator);
        }

        // Do not fuse with any producer linalg named ops for now.
        if (isa<linalg::LinalgOp>(producer)) {
          return false;
        }

        // Do not fuse with consumer linalg named ops or reductions.
        if (auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(consumer)) {
          return isa<linalg::GenericOp>(consumerLinalgOp) &&
                 llvm::all_of(consumerLinalgOp.getIteratorTypesArray(),
                              linalg::isParallelIterator);
        }
        // Fuse in all other cases.
        return true;
      };
  linalg::populateFoldReshapeOpsByExpansionPatterns(bubbleExpandShapePatterns,
                                                    bubbleUpExpansionControlFn);
  // Add patterns to do some additional cleanup (on top of canonicalizations
  // that can be done later) of reshape ops.
  tensor::populateFoldTensorEmptyPatterns(bubbleExpandShapePatterns);

  GreedyRewriteConfig rewriteConfig;
  rewriteConfig.maxIterations = GreedyRewriteConfig::kNoLimit;
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(bubbleExpandShapePatterns),
                                          rewriteConfig))) {
    getOperation()->emitOpError("Failed to perform elementwise operations");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::IREE::Flow
