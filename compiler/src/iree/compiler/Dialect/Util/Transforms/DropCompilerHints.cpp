// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Util {

class DropCompilerHintsPass
    : public DropCompilerHintsBase<DropCompilerHintsPass> {
public:
  void runOnOperation() override {
    // We can't use patterns and applyPatternsAndFoldGreedily because that
    // automatically does canonicalization.
    getOperation()->walk([&](IREE::Util::OptimizationBarrierOp op) {
      op.replaceAllUsesWith(op.getOperands());
      op.erase();
    });
  }
};

std::unique_ptr<OperationPass<void>> createDropCompilerHintsPass() {
  return std::make_unique<DropCompilerHintsPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
