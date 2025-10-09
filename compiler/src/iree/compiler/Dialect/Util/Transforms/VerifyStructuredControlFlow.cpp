// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_VERIFYSTRUCTUREDCONTROLFLOWPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

struct VerifyStructuredControlFlowPass
    : public impl::VerifyStructuredControlFlowPassBase<
          VerifyStructuredControlFlowPass> {
  using Base::Base;
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.empty()) {
      // External function/declaration.
      return;
    }

    // Walk the function and check for any branch operations.
    // BranchOpInterface catches cf.br, cf.cond_br, cf.switch and any other
    // operations that perform branching. Without these operations multi-block
    // regions cannot form a control flow graph so we don't need to check for
    // multiple blocks.
    auto result = funcOp.walk([&](Operation *op) -> WalkResult {
      if (isa<BranchOpInterface>(op)) {
        return op->emitError()
               << "unexpected branch operation in function after structured "
                  "control flow conversion";
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }

    // Preserve all analyses since this is a read-only verification pass.
    markAllAnalysesPreserved();
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
