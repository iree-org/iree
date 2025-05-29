// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_TESTRESHAPEFUSIONPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {
struct TestReshapeFusionPass final
    : impl::TestReshapeFusionPassBase<TestReshapeFusionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    LinalgExt::populateFoldReshapeOpsByExpansionPatterns(
        patterns, [](OpOperand *) { return true; });
    tensor::populateFoldTensorEmptyPatterns(patterns);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    tensor::populateBubbleUpExpandShapePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::IREE::LinalgExt
