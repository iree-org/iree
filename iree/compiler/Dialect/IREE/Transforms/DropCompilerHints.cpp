// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

class DropCompilerHintsPass
    : public PassWrapper<DropCompilerHintsPass, OperationPass<void>> {
 public:
  void runOnOperation() override {
    // We can't use patterns and applyPatternsAndFoldGreedily because that
    // automatically does canonicalization.
    getOperation()->walk([&](DoNotOptimizeOp op) {
      op.replaceAllUsesWith(op.getOperands());
      op.erase();
    });
  }
};

std::unique_ptr<OperationPass<void>> createDropCompilerHintsPass() {
  return std::make_unique<DropCompilerHintsPass>();
}

static PassRegistration<DropCompilerHintsPass> pass(
    "iree-drop-compiler-hints",
    "Deletes operations that have no runtime equivalent and are only used in "
    "the compiler. This should be performed after all other compiler passes.");

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
