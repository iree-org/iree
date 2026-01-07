// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Analysis/LinearScan/LiveIntervals.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::IREE::VM {

class LiveIntervalsTestPass
    : public PassWrapper<LiveIntervalsTestPass,
                         OperationPass<IREE::VM::FuncOp>> {
public:
  StringRef getArgument() const override {
    return "test-iree-vm-live-intervals";
  }

  StringRef getDescription() const override {
    return "Test pass used for live intervals analysis";
  }

  void runOnOperation() override {
    if (failed(LiveIntervals::annotateIR(getOperation()))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<IREE::VM::FuncOp>> createLiveIntervalsTestPass() {
  return std::make_unique<LiveIntervalsTestPass>();
}

static PassRegistration<LiveIntervalsTestPass> pass;

} // namespace mlir::iree_compiler::IREE::VM
