// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::IREE::VM {

class ValueLivenessTestPass
    : public PassWrapper<ValueLivenessTestPass,
                         OperationPass<IREE::VM::FuncOp>> {
public:
  StringRef getArgument() const override {
    return "test-iree-vm-value-liveness";
  }

  StringRef getDescription() const override {
    return "Test pass used for liveness analysis";
  }

  void runOnOperation() override {
    if (failed(ValueLiveness::annotateIR(getOperation()))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<IREE::VM::FuncOp>> createValueLivenessTestPass() {
  return std::make_unique<ValueLivenessTestPass>();
}

static PassRegistration<ValueLivenessTestPass> pass;

} // namespace mlir::iree_compiler::IREE::VM
