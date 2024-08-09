// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

/// Command line to use native hardware operations instead of polynomial
/// approximation.
static llvm::cl::opt<bool> clNativeMathPrecision(
    "iree-codegen-gpu-native-math-precision",
    llvm::cl::desc(
        "Skip polynomial lowering for math op natively available on GPU"),
    llvm::cl::init(false));

#define GEN_PASS_DEF_POLYNOMIALAPPROXIMATIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// math dialect elementry functions -> polynomial form.
class PolynomialApproximationPass
    : public impl::PolynomialApproximationPassBase<
          PolynomialApproximationPass> {
  void runOnOperation() override {
    RewritePatternSet mathPatterns(&getContext());
    populateExpandTanPattern(mathPatterns);
    populateExpandSinhPattern(mathPatterns);
    populateExpandCoshPattern(mathPatterns);
    populateExpandAsinhPattern(mathPatterns);
    populateExpandAcoshPattern(mathPatterns);
    populateExpandAtanhPattern(mathPatterns);
    populateExpandPowFPattern(mathPatterns);
    populateExpandFPowIPattern(mathPatterns);

    if (clNativeMathPrecision) {
      mathPatterns.add<math::ErfPolynomialApproximation>(&getContext());
    } else {
      populateExpandExp2FPattern(mathPatterns);
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
} // namespace mlir::iree_compiler
