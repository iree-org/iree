// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::iree_compiler::IREE::Flow {

namespace {

class CleanupTensorShapesPass
    : public CleanupTensorShapesBase<CleanupTensorShapesPass> {
  void runOnOperation() override {
    // Walk ops and ensure we no longer have any tensor shape queries.
    // If we come across any shape witness ops we can erase those.
    bool foundBadOps = false;
    getOperation()->walk(
        [&](Operation *op) {
          if (auto tieShapeOp = dyn_cast<IREE::Flow::TensorTieShapeOp>(op)) {
            tieShapeOp.replaceAllUsesWith(tieShapeOp.getOperand());
            tieShapeOp.erase();
          } else if (isa<tensor::DimOp>(op) || isa<tensor::RankOp>(op)) {
            op->emitOpError()
                << "unexpected during shape cleanup; dynamic dimensions must "
                   "have been resolved prior to leaving the flow dialect";
            foundBadOps = true;
          }
        });
    if (foundBadOps)
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createCleanupTensorShapesPass() {
  return std::make_unique<CleanupTensorShapesPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
