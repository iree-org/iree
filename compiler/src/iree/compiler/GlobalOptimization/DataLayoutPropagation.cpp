// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

struct DataLayoutPropagationPass
    : public DataLayoutPropagationBase<DataLayoutPropagationPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    RewritePatternSet patterns(context);
    linalg::populateDataLayoutPropagationPatterns(
        patterns, [](OpOperand *opOperand) {
          Operation *producer = opOperand->get().getDefiningOp();
          Operation *consumer = opOperand->getOwner();
          (void)consumer;
          // Currently only bubble up/push down pack/unpack through
          // collapse/expand shape ops.
          // TODO(#17734): The propagation through expand_shape ops is broken.
          // Enable the propagation once we find it useful and the upstream
          // issue is fixed.
          return isa<tensor::CollapseShapeOp>(producer);
        });
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError("folding patterns failed");
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDataLayoutPropagationPass() {
  return std::make_unique<DataLayoutPropagationPass>();
}
} // namespace mlir::iree_compiler::GlobalOptimization
