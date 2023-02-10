// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/MicroKernelOpInterface.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct LowerMicroKernelOpsToCallsPass
    : LowerMicroKernelOpsToCallsBase<LowerMicroKernelOpsToCallsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, func::FuncDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

void LowerMicroKernelOpsToCallsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  SmallVector<Operation *> toDelete;
  Operation *errorOp = nullptr;
  IRRewriter rewriter(context);
  WalkResult result = getOperation().walk(
      [&](IREE::Codegen::MicroKernelOpInterface microKernelOp) -> WalkResult {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(microKernelOp);
        FailureOr<func::CallOp> callOp =
            microKernelOp.lowerToFunctionCall(rewriter);
        if (failed(callOp)) {
          errorOp = microKernelOp;
          return WalkResult::interrupt();
        }
        toDelete.push_back(microKernelOp);
        return WalkResult::advance();
      });
  if (result.wasInterrupted()) {
    errorOp->emitOpError(
        "failed to lower micro kernel operation to function call");
    return signalPassFailure();
  }
  for (auto op : toDelete) {
    op->erase();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createLowerMicroKernelOpsToCallsPass() {
  return std::make_unique<LowerMicroKernelOpsToCallsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
