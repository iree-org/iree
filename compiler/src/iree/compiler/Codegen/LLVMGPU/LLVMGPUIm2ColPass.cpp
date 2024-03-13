// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

class LLVMGPUIm2ColPass : public LLVMGPUIm2ColBase<LLVMGPUIm2ColPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() override;
};

} // namespace

void LLVMGPUIm2ColPass::runOnOperation() {
  auto operation = getOperation();
  SmallVector<Operation *> convOps;
  operation->walk([&](Operation *convOp) {
    if (isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNhwcFhwcOp,
            linalg::DepthwiseConv2DNhwcHwcOp, linalg::Conv2DNchwFchwOp>(
            convOp)) {
      convOps.push_back(convOp);
    }
  });

  IRRewriter rewriter(&getContext());
  for (auto op : convOps) {
    IRRewriter::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(op);
    auto maybeTransformed =
        TypeSwitch<Operation *, FailureOr<std::pair<Operation *, Operation *>>>(
            op)
            .Case([&](linalg::Conv2DNhwcHwcfOp op) {
              return rewriteInIm2Col(rewriter, op);
            })
            .Case([&](linalg::Conv2DNhwcFhwcOp op) {
              return rewriteInIm2Col(rewriter, op);
            })
            .Case([&](linalg::DepthwiseConv2DNhwcHwcOp op) {
              return rewriteInIm2Col(rewriter, op);
            })
            .Case([&](linalg::Conv2DNchwFchwOp op) {
              return rewriteInIm2Col(rewriter, op);
            })
            .Default([&](Operation *op) {
              return rewriter.notifyMatchFailure(op, "not supported");
            });
    if (failed(maybeTransformed)) {
      continue;
    }
  }

  // Bubble collapse

  RewritePatternSet patterns(&getContext());
  linalg::populateFoldReshapeOpsByCollapsingPatterns(
      patterns, [](OpOperand *) { return true; });
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUIm2ColPass() {
  return std::make_unique<LLVMGPUIm2ColPass>();
}

} // namespace mlir::iree_compiler
