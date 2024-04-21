// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- SinkReshapes.cpp --- Pass to sink reshapes -----------------------===//
//
// This pass sinks reshapes (tensor.expand_shape/tensor.collapse_shape) that
// block producer-consumer fusion. These reshapes are generally produced by
// the `BubbleExpandShapes.cpp` pass that propagates reshapes towards the
// arguments, but get blocked on named op.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/FusionUtils.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-sink-reshapes"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_SINKRESHAPESPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

class SinkReshapesPass : public impl::SinkReshapesPassBase<SinkReshapesPass> {
public:
  using Base::Base;

  void runOnOperation() override;
};

/// Control function to check if an `tensor.expand_shape` (which is producer of
/// `opOperand`) should be pushed past the `genericOp` (which is the consumer of
/// `opOperand`).
static bool shouldSinkExpandShapeOp(OpOperand *opOperand) {
  auto reshapeOp =
      dyn_cast<tensor::ExpandShapeOp>(opOperand->get().getDefiningOp());
  if (!reshapeOp) {
    return false;
  }
  Operation *consumer = opOperand->getOwner();
  if (!isNonNullAndOutsideDispatch({reshapeOp, consumer})) {
    return false;
  }

  // Do not sink if consumer is a contraction/matmul like op.
  if (auto linalgConsumerOp = dyn_cast<linalg::LinalgOp>(consumer)) {
    if (linalg::isaContractionOpInterface(linalgConsumerOp))
      return false;
  }

  return llvm::isa_and_nonnull<linalg::LinalgOp, tensor::UnPackOp,
                               LinalgExt::UnsetEncodingOp>(
      reshapeOp.getSrc().getDefiningOp());
}

void SinkReshapesPass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet sinkReshapePatterns(context);
  linalg::populateFoldReshapeOpsByCollapsingPatterns(sinkReshapePatterns,
                                                     shouldSinkExpandShapeOp);
  // Add patterns to fold `tensor.empty` and reshape ops.
  tensor::populateFoldTensorEmptyPatterns(sinkReshapePatterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(sinkReshapePatterns)))) {
    getOperation()->emitOpError("failed to sink reshape ops");
    return signalPassFailure();
  }
}

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
