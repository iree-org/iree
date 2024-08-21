// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_DATALAYOUTPROPAGATIONPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

struct DataLayoutPropagationPass
    : public impl::DataLayoutPropagationPassBase<DataLayoutPropagationPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    RewritePatternSet patterns(context);
    linalg::populateDataLayoutPropagationPatterns(
        patterns, [](OpOperand *opOperand) {
          Operation *producer = opOperand->get().getDefiningOp();
          Operation *consumer = opOperand->getOwner();
          if (isa<tensor::PackOp>(consumer)) {
            return isa<tensor::CollapseShapeOp>(producer);
          }
          if (isa<tensor::UnPackOp>(producer)) {
            return isa<tensor::ExpandShapeOp>(consumer);
          }
          return false;
        });
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError("folding patterns failed");
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
