// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {
struct TransposeMatmulPass : public TransposeMatmulBase<TransposeMatmulPass> {
  TransposeMatmulPass(TransposeMatmulOption input) : inputToTranspose(input) {}

  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override {
    if (failed(Pass::initializeOptions(options, errorHandler))) {
      return failure();
    }

    auto selectedInput =
        llvm::StringSwitch<FailureOr<TransposeMatmulOption>>(input)
            .Case("", TransposeMatmulOption::None)
            .Case("lhs", TransposeMatmulOption::Lhs)
            .Case("rhs", TransposeMatmulOption::Rhs)
            .Default(failure());
    if (failed(selectedInput))
      return failure();

    inputToTranspose = *selectedInput;
    return success();
  }

  void runOnOperation() override {
    if (inputToTranspose == TransposeMatmulOption::None)
      return;

    bool transposeLHS = inputToTranspose == TransposeMatmulOption::Lhs;

    RewritePatternSet patterns(&getContext());
    linalg::populateTransposeMatmulPatterns(patterns, transposeLHS);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

private:
  TransposeMatmulOption inputToTranspose = TransposeMatmulOption::None;
};
} // namespace

std::unique_ptr<Pass> createTransposeMatmulPass(TransposeMatmulOption input) {
  return std::make_unique<TransposeMatmulPass>(input);
}

} // namespace mlir::iree_compiler::GlobalOptimization
