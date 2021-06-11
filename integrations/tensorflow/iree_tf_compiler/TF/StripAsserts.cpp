// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TF/Passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

class StripAssertsPass
    : public PassWrapper<StripAssertsPass, OperationPass<FuncOp>> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    DenseSet<Operation *> assertOps;
    funcOp.walk([&](Operation *op) {
      if (isa<mlir::TF::AssertOp>(op)) {
        assertOps.insert(op);
      }
    });

    for (Operation *assertOp : assertOps) {
      assertOp->erase();
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>> createStripAssertsPass() {
  return std::make_unique<StripAssertsPass>();
}

static PassRegistration<StripAssertsPass> funcPass("iree-tf-strip-asserts",
                                                   "Remove tf.Assert ops");

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
