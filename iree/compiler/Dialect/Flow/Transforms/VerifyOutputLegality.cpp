// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
class VerifyOutputLegalityPass
    : public VerifyOutputLegalityBase<VerifyOutputLegalityPass> {
  void runOnOperation() override {
    auto walkResult = getOperation().walk([&](Operation *op) -> WalkResult {
      // We should generally only be operating on tensor types using flow
      // dialect ops at this point. However, exhaustively allow/deny-listing
      // all ops and dialects would be difficult to maintain, so we just keep a
      // best-effort list.

      // Only test ops in the standard and math dialects.
      auto dialectNamespace = op->getDialect()->getNamespace();
      if (dialectNamespace != StandardOpsDialect::getDialectNamespace() &&
          dialectNamespace != math::MathDialect::getDialectNamespace()) {
        return WalkResult::advance();
      }
      // Certain standard ops for flow control are okay to operate on tensors.
      if (dyn_cast<mlir::ReturnOp>(op) || dyn_cast<mlir::CallOp>(op) ||
          dyn_cast<mlir::BranchOp>(op) || dyn_cast<mlir::CondBranchOp>(op)) {
        return WalkResult::advance();
      }

      for (auto resultType : op->getResultTypes()) {
        if (resultType.isa<TensorType>()) {
          return op->emitOpError("illegal operation returning ")
                 << resultType
                 << " in output from flow transformation, flow dialect "
                    "ops should be used for acting on tensors at this point";
        }
      }
      for (auto operand : op->getOperands()) {
        auto operandType = operand.getType();
        if (operandType.isa<TensorType>()) {
          return op->emitOpError("illegal operand type ")
                 << operandType
                 << " in output from flow transformation, flow dialect "
                    "ops should be used for acting on tensors at this point";
        }
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<mlir::FuncOp>> createVerifyOutputLegalityPass() {
  return std::make_unique<VerifyOutputLegalityPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
