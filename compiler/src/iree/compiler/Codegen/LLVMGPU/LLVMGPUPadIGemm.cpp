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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-pad-igemm"

namespace mlir::iree_compiler {

namespace {

class LLVMGPUPadIGemmPass : public LLVMGPUPadIGemmBase<LLVMGPUPadIGemmPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() override;
};

} // namespace

void LLVMGPUPadIGemmPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();
  linalg::LinalgOp matmul;
  auto found = funcOp->walk([&](linalg::LinalgOp op) {
    if (op.getNumReductionLoops() == 0) {
      return WalkResult::advance();
    }
    if (op.getNumReductionLoops() != 1) {
      return WalkResult::interrupt();
    }
    if (matmul) {
      return WalkResult::interrupt();
    }
    matmul = op;
    return WalkResult::advance();
  });

  if (found.wasInterrupted()) {
    LLVM_DEBUG(llvm::dbgs() << "skip, expect a single matmul\n");
    return;
  }

  if (matmul.getNumLoops() != 4) {
    LLVM_DEBUG(llvm::dbgs() << "skip, expect matmul is from igemm\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "candidate: " << matmul << "\n");
  IRRewriter rewriter(ctx);
  SmallVector<int64_t> paddingDims = {0, 1, 2, 3};
  SmallVector<bool> packPaddings = {1, 0, 1};
  SmallVector<int64_t> padToMultipleOf(paddingDims.size(), 1);
  SmallVector<Attribute> paddingValueAttributes;
  for (auto &operand : matmul->getOpOperands()) {
    auto elemType = getElementTypeOrSelf(operand.get().getType());
    paddingValueAttributes.push_back(rewriter.getZeroAttr(elemType));
  }

  auto options =
      linalg::LinalgPaddingOptions()
          .setPaddingDimensions(paddingDims)
          .setPaddingValues(paddingValueAttributes)
          .setPadToMultipleOf(padToMultipleOf)
          .setPackPaddings(packPaddings)
          .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);
  linalg::LinalgOp paddedOp;
  SmallVector<Value> replacements;
  SmallVector<tensor::PadOp> newPadOps;
  if (failed(rewriteAsPaddedOp(rewriter, matmul, options, paddedOp,
                               replacements, newPadOps))) {
    LLVM_DEBUG(llvm::dbgs() << "failed to pad op " << matmul << "\n");
    return signalPassFailure();
  }

  // We need to perform our own replacement here because this API is still
  // used in patterns that "pad and hoist", for which the replacement values
  // need to be different.
  // TODO: clean this up and stop "pad and hoist" behavior more globally now
  // that we have more composable abstractions.
  rewriter.replaceOp(matmul, replacements);
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUPadIGemmPass() {
  return std::make_unique<LLVMGPUPadIGemmPass>();
}

} // namespace mlir::iree_compiler
