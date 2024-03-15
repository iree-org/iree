// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-llvmgpu-rewrite-pad-in-dps"

namespace mlir::iree_compiler {

namespace {

class LLVMGPURewritePadInDestinationPassingStylePass
    : public LLVMGPURewritePadInDestinationPassingStyleBase<
          LLVMGPURewritePadInDestinationPassingStylePass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() override;
};

} // namespace

void LLVMGPURewritePadInDestinationPassingStylePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();
  SmallVector<tensor::PadOp> candidates;
  funcOp.walk([&](tensor::PadOp op) { candidates.push_back(op); });
  IRRewriter rewriter(ctx);
  for (auto padOp : candidates) {
    LLVM_DEBUG(llvm::dbgs() << "candidate: " << padOp << "\n");
    rewriter.setInsertionPoint(padOp);
    FailureOr<Operation *> maybeResult =
        linalg::rewriteInDestinationPassingStyle(rewriter, padOp);
    if (failed(maybeResult)) {
      LLVM_DEBUG(llvm::dbgs() << "skip, failed to rewrite the pad op in dps\n");
      continue;
    }
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPURewritePadInDestinationPassingStylePass() {
  return std::make_unique<LLVMGPURewritePadInDestinationPassingStylePass>();
}

} // namespace mlir::iree_compiler
