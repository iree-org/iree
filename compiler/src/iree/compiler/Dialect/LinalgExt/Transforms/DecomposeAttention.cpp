// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEATTENTIONPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct DecomposeAttentionPass final
    : impl::DecomposeAttentionPassBase<DecomposeAttentionPass> {
  using impl::DecomposeAttentionPassBase<
      DecomposeAttentionPass>::DecomposeAttentionPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override;
};
} // namespace

void DecomposeAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  getOperation().walk([&](OnlineAttentionOp onlineAtt) {
    rewriter.setInsertionPoint(onlineAtt);
    FailureOr<SmallVector<Value>> results =
        onlineAtt.decomposeOperation(rewriter);
    if (failed(results)) {
      onlineAtt->emitOpError("Could not decompose online attention");
      return signalPassFailure();
    }
    rewriter.replaceOp(onlineAtt, results.value());
  });
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
