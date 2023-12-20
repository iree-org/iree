// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {
struct SimplifyPackUnpackPass
    : public SimplifyPackUnpackBase<SimplifyPackUnpackPass> {

  void runOnOperation() override;
};
} // namespace

void SimplifyPackUnpackPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  tensor::populateSimplifyPackAndUnpackPatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> createSimplifyPackUnpackPass() {
  return std::make_unique<SimplifyPackUnpackPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
