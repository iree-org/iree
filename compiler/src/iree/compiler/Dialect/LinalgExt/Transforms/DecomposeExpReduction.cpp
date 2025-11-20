// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEEXPREDUCTIONPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct DecomposeExpReductionPass final
    : impl::DecomposeExpReductionPassBase<DecomposeExpReductionPass> {
  using DecomposeExpReductionPassBase::DecomposeExpReductionPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

struct DecomposeMultipleResults : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
};

struct DecomposeExpReduction : OpRewritePattern<ExpReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpReductionOp expReductionOp,
                                PatternRewriter &rewriter) const override {
    auto decomposeResults = expReductionOp.decomposeOperation(rewriter);
    if (failed(decomposeResults)) {
      return failure();
    }
    rewriter.replaceOp(expReductionOp,
                       decomposeResults->begin()->getDefiningOp());
    return success();
  }
};

} // namespace

void DecomposeExpReductionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  // patterns.add<DecomposeExpReduction, DecomposeMultipleResults>(context);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    getOperation()->emitOpError("Failed to apply patterns");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::IREE::LinalgExt