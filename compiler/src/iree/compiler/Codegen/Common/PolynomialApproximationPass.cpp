// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

/// Command line to use native hardware operations instead of polynomial
/// approximation.
static llvm::cl::opt<bool> clNativeMathPrecision(
    "iree-codegen-gpu-native-math-precision",
    llvm::cl::desc(
        "Skip polynomial lowering for math op natively available on GPU"),
    llvm::cl::init(false));

namespace {

/// math dialect elementry functions -> polynomial form.
class PolynomialApproximationPass
    : public PolynomialApproximationPassBase<PolynomialApproximationPass> {
  void runOnOperation() override {
    RewritePatternSet mathPatterns(&getContext());
    populateExpandTanPattern(mathPatterns);
    populateExpandExp2FPattern(mathPatterns);
    populateExpandPowFPattern(mathPatterns);

    if (clNativeMathPrecision) {
      mathPatterns.add<math::ErfPolynomialApproximation>(&getContext());
    } else {
      populateMathPolynomialApproximationPatterns(mathPatterns);
      populateExpandRoundEvenPattern(mathPatterns);
    }
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(mathPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<>> createPolynomialApproximationPass() {
  return std::make_unique<PolynomialApproximationPass>();
}

} // namespace iree_compiler
} // namespace mlir
