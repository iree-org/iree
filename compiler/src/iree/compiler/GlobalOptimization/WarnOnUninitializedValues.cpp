// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_WARNONUNINITIALIZEDVALUESPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

using OperandsIndicesSet = llvm::SmallSet<int, 8>;

static OperandsIndicesSet getConsumedOperandsIndices(Block &body) {
  OperandsIndicesSet consumedOperandsIndices;
  for (auto [i, arg] : llvm::enumerate(body.getArguments())) {
    for (auto user : arg.getUsers()) {
      if (user != body.getTerminator()) {
        consumedOperandsIndices.insert(i);
      }
    }
  }
  return consumedOperandsIndices;
}

class WarnOnUninitializedValuesPass final
    : public impl::WarnOnUninitializedValuesPassBase<
          WarnOnUninitializedValuesPass> {
public:
  void runOnOperation() override {
    Operation *op = getOperation();
    op->walk([&](linalg::LinalgOp linalgOp) -> WalkResult {
      Block &body = linalgOp->getRegion(0).front();
      OperandsIndicesSet consumedOperandsIndices =
          getConsumedOperandsIndices(body);
      for (int i : consumedOperandsIndices) {
        Operation *defOp = linalgOp->getOperands()[i].getDefiningOp();
        if (isa_and_nonnull<tensor::EmptyOp>(defOp)) {
          linalgOp->emitWarning(
              "reads uninitialized values from an operand "
              "produced by a tensor.empty op. To disable this warning, pass "
              "--iree-global-opt-enable-warn-on-uninitialized-values=false .");
          return WalkResult::advance();
        }
      }
      return WalkResult::advance();
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
