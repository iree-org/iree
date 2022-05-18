// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct LLVMCPUEliminateVectorTransferOpsPass
    : LLVMCPUEliminateVectorTransferOpsBase<
          LLVMCPUEliminateVectorTransferOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

void LLVMCPUEliminateVectorTransferOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();

  RewritePatternSet patterns(context);
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUEliminateVectorTransferOpsPass() {
  return std::make_unique<LLVMCPUEliminateVectorTransferOpsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
