// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LOWERUKERNELOPSTOCALLSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct LowerUKernelOpsToCallsPass
    : impl::LowerUKernelOpsToCallsPassBase<LowerUKernelOpsToCallsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, func::FuncDialect>();
  }
  void runOnOperation() override;
};
} // namespace

void LowerUKernelOpsToCallsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  llvm::MapVector<IREE::Codegen::UKernelOpInterface, mlir::CallOpInterface>
      toReplace;
  Operation *errorOp = nullptr;
  IRRewriter rewriter(context);
  WalkResult result = getOperation().walk(
      [&](IREE::Codegen::UKernelOpInterface microKernelOp) -> WalkResult {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(microKernelOp);
        FailureOr<mlir::CallOpInterface> callOp =
            microKernelOp.lowerToFunctionCall(rewriter);
        if (failed(callOp)) {
          errorOp = microKernelOp;
          return WalkResult::interrupt();
        }
        toReplace[microKernelOp] = callOp.value();
        return WalkResult::advance();
      });
  if (result.wasInterrupted()) {
    errorOp->emitOpError(
        "failed to lower micro kernel operation to function call");
    return signalPassFailure();
  }
  for (auto r : toReplace) {
    rewriter.replaceOp(r.first, r.second->getResults());
  }
}
} // namespace mlir::iree_compiler
