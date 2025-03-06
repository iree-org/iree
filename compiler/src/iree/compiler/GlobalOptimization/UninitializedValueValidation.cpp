// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_UNINITIALIZEDVALUEVALIDATIONPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

class UninitializedValueValidationPass final
    : public impl::UninitializedValueValidationPassBase<
          UninitializedValueValidationPass> {
public:
  void runOnOperation() override {
    Operation *op = getOperation();
    op->walk([&](linalg::LinalgOp linalgOp) -> WalkResult {
      Block &body = linalgOp->getRegion(0).front();
      llvm::SmallSet<int, 4> consumedOperands;
      for (Operation &op : body.getOperations()) {
        if (&op == body.getTerminator()) {
          continue;
        }
        for (auto operand : op.getOperands()) {
          for (auto [i, arg] : llvm::enumerate((body.getArguments()))) {
            if (operand == arg) {
              consumedOperands.insert(i);
              break;
            }
          }
        }
      }
      for (int i : consumedOperands) {
        Operation *defOp = linalgOp->getOperands()[i].getDefiningOp();
        if (isa_and_nonnull<tensor::EmptyOp>(defOp)) {
          linalgOp->emitWarning("reads uninitialized values from an operand "
                                "produced by a tensor.empty op");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
