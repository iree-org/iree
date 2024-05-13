// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {

struct DropSchedulePass : public PassWrapper<DropSchedulePass, Pass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DropSchedulePass)

  StringRef getArgument() const final {
    return "transform-dialect-drop-schedule";
  }

  StringRef getDescription() const final {
    return "Drop the schedule from the operation";
  }

  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

  void runOnOperation() override {
    SmallVector<Operation *> toDelete;
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
      if (isa<::mlir::transform::TransformOpInterface>(nestedOp)) {
        toDelete.push_back(nestedOp);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    for (auto op : toDelete) {
      op->erase();
    }
    SmallVector<ModuleOp> modulesToDelete;
    // Remove potential empty module after cleanup.
    getOperation()->walk([&](ModuleOp module) {
      if (module.getBodyRegion().hasOneBlock() && module.getBody()->empty()) {
        modulesToDelete.push_back(module);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    for (auto module : modulesToDelete) {
      module->erase();
    }
  }
};
} // namespace

/// Create a Linalg pass to drop the schedule from the module.
std::unique_ptr<Pass> mlir::createDropSchedulePass() {
  return std::make_unique<DropSchedulePass>();
}

/// Registration hook for the Linalg drop schedule from module pass.
void mlir::linalg::transform::registerDropSchedulePass() {
  PassRegistration<DropSchedulePass>();
}
