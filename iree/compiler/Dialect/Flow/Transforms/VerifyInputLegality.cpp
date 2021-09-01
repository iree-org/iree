// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
class VerifyInputLegalityPass
    : public VerifyInputLegalityBase<VerifyInputLegalityPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
      StringRef opDialectName = op->getDialect()->getNamespace();

      // Exception: tosa::ApplyScaleOp is lowered through flow for now.
      if (dyn_cast<tosa::ApplyScaleOp>(op)) {
        return WalkResult::advance();
      }

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

std::unique_ptr<OperationPass<mlir::FuncOp>> createVerifyInputLegalityPass() {
  return std::make_unique<VerifyInputLegalityPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
