// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/STLExtras.h"

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
      if (isa<iree_compiler::IREE::LinalgExt::DoNotDCEOperandsOp>(nestedOp)) {
        toDelete.push_back(nestedOp);
      } else if (isa<::mlir::transform::TransformOpInterface>(nestedOp)) {
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
