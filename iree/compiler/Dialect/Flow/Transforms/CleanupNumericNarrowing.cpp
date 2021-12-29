// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

class CleanupNumericNarrowingPass
    : public CleanupNumericNarrowingBase<CleanupNumericNarrowingPass> {
  void runOnOperation() override {
    getOperation()->walk([](IREE::Util::NumericOptionalNarrowOp op) {
      op.getResult().replaceAllUsesWith(op.getOperand());
      op->erase();
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createCleanupNumericNarrowingPass() {
  return std::make_unique<CleanupNumericNarrowingPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
