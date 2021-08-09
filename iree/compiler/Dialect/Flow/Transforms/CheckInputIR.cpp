// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
class CheckInputIRPass : public CheckInputIRBase<CheckInputIRPass> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    auto walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
      StringRef opDialectName = op->getDialect()->getNamespace();
      if (opDialectName == "mhlo" || opDialectName == "tosa") {
        return op->emitOpError(
                   "illegal operation in input to iree core compiler. Use "
                   "-iree-input-type=")
               << opDialectName << " to legalize this operation";
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createCheckInputIRPass() {
  return std::make_unique<CheckInputIRPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
