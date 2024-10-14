// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEAGGREGATEOPSPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct DecomposeAggregateOpsPass final
    : impl::DecomposeAggregateOpsPassBase<DecomposeAggregateOpsPass> {
  using Base::Base;

  // Use the initializer to set default options for ListOption.
  LogicalResult initialize(MLIRContext *context) override {
    this->decomposeOps = {IREE::LinalgExt::CustomOp::getOperationName().str()};
    return success();
  }

  void runOnOperation() override;
};

void DecomposeAggregateOpsPass::runOnOperation() {
  SmallVector<linalg::AggregatedOpInterface> aggregateOps;
  llvm::StringSet<> opNamesSet;
  opNamesSet.insert(this->decomposeOps.begin(), this->decomposeOps.end());
  auto walkResult = getOperation()->walk([&](Operation *op) -> WalkResult {
    if (opNamesSet.contains(op->getName().getStringRef())) {
      auto aggregateOp = dyn_cast<linalg::AggregatedOpInterface>(op);
      if (!aggregateOp) {
        return op->emitOpError(
            "expected operation to implement AggregatedOpInterface");
      }
      aggregateOps.push_back(aggregateOp);
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }

  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  for (auto aggregateOp : llvm::make_early_inc_range(aggregateOps)) {
    rewriter.setInsertionPoint(aggregateOp);
    FailureOr<SmallVector<Value>> replacements =
        aggregateOp.decomposeOperation(rewriter);
    if (failed(replacements)) {
      aggregateOp->emitOpError("failed to decompose operation");
      return signalPassFailure();
    }
    rewriter.replaceOp(aggregateOp, replacements.value());
  }
}

} // namespace
} // namespace mlir::iree_compiler::IREE::LinalgExt
