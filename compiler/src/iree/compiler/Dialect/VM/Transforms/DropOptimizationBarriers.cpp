// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::VM {

#define GEN_PASS_DEF_DROPOPTIMIZATIONBARRIERSPASS
#include "iree/compiler/Dialect/VM/Transforms/Passes.h.inc"

class DropOptimizationBarriersPass
    : public IREE::VM::impl::DropOptimizationBarriersPassBase<
          DropOptimizationBarriersPass> {
  void runOnOperation() override {
    getOperation()->walk([&](IREE::VM::OptimizationBarrierOp op) {
      op.replaceAllUsesWith(op.getOperands());
      op.erase();
    });
  }
};

} // namespace mlir::iree_compiler::IREE::VM
