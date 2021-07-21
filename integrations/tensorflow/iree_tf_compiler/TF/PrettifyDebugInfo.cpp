// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TF/Passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

class PrettifyDebugInfoPass
    : public PassWrapper<PrettifyDebugInfoPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const override {
    return "iree-tf-prettify-debug-info";
  }

  StringRef getDescription() const override {
    return "Simplifies TF debug info to make it easier to look at";
  }

  void runOnOperation() override {
    // TODO: Finish algorithm for simplifying TF debug info.
    // auto moduleOp = getOperation();
    // moduleOp.walk([&](Operation *op) {
    //   Location loc = op->getLoc();
    //   if (auto callSite = loc.dyn_cast<CallSiteLoc>()) {
    //     callSite.getCallee().dump();
    //   }
    // });
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPrettifyDebugInfoPass() {
  return std::make_unique<PrettifyDebugInfoPass>();
}

static PassRegistration<PrettifyDebugInfoPass> modulePass;

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
