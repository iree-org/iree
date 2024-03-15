// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

class LLVMGPUIm2ColPass : public LLVMGPUIm2ColBase<LLVMGPUIm2ColPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() override;
};

} // namespace

template <typename OpTy>
FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter, OpTy op,
                SmallVector<NamedAttribute> &additionalAttributes) {
  additionalAttributes = linalg::getPrunedAttributeList(op);
  return rewriteInIm2Col(rewriter, op);
}

void LLVMGPUIm2ColPass::runOnOperation() {
  auto operation = getOperation();

  IRRewriter rewriter(&getContext());
  operation->walk([&](Operation *op) {
    IRRewriter::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(op);
    SmallVector<NamedAttribute> additionalAttributes;
    auto maybeTransformed =
        TypeSwitch<Operation *, FailureOr<std::pair<Operation *, Operation *>>>(
            op)
            .Case<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNhwcFhwcOp,
                  linalg::DepthwiseConv2DNhwcHwcOp, linalg::Conv2DNchwFchwOp>(
                [&](auto op) {
                  return rewriteInIm2Col(rewriter, op, additionalAttributes);
                })
            .Default([&](Operation *op) {
              return rewriter.notifyMatchFailure(op, "not supported");
            });
    if (failed(maybeTransformed)) {
      return;
    }
    auto matmulOp = cast<tensor::ExpandShapeOp>(maybeTransformed->second)
                        .getSrc()
                        .getDefiningOp();
    for (auto attr : additionalAttributes) {
      matmulOp->setAttr(attr.getName(), attr.getValue());
    }
  });

  // Bubble collapse
  RewritePatternSet patterns(&getContext());
  linalg::populateFoldReshapeOpsByCollapsingPatterns(
      patterns, [](OpOperand *) { return true; });
  populateReshapeToInterfaceTensorPatterns(patterns);
  linalg::FillOp::getCanonicalizationPatterns(patterns, &getContext());
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUIm2ColPass() {
  return std::make_unique<LLVMGPUIm2ColPass>();
}

} // namespace mlir::iree_compiler
