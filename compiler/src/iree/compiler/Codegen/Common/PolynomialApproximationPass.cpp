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
class PolynomialApproximationPass final
    : public impl::PolynomialApproximationPassBase<
          PolynomialApproximationPass> {

public:
  using impl::PolynomialApproximationPassBase<
      PolynomialApproximationPass>::PolynomialApproximationPassBase;

  void runOnOperation() override {
    RewritePatternSet mathPatterns(&getContext());

    if (isLLVMGPU) {
      populateLLVMGPUPolynomialApproximationPatterns(mathPatterns);
    } else {
      populateGenericPolyApproximationPatterns(mathPatterns);
    }

    if (failed(
            applyPatternsGreedily(getOperation(), std::move(mathPatterns)))) {
      return signalPassFailure();
    }
  }

  /// Only expand math dialect elementry functions not supported by device libs.
  void populateLLVMGPUPolynomialApproximationPatterns(
      RewritePatternSet &mathPatterns) {
    // TODO(lialan): Handle these functions efficiently in ROCDL/NVVM
    // conversion passes. This expansion is likely suboptimal.
    populateExpandPowFPattern(mathPatterns);
    populateExpandFPowIPattern(mathPatterns);

    if (clNativeMathPrecision) {
      mathPatterns.add<math::ErfPolynomialApproximation>(&getContext());
    } else {
      populateExpandExp2FPattern(mathPatterns);
      populateMathPolynomialApproximationPatterns(mathPatterns);
      populateExpandRoundEvenPattern(mathPatterns);
    }
  }

  /// Convert math dialect elementry functions to polynomial form.
  void
  populateGenericPolyApproximationPatterns(RewritePatternSet &mathPatterns) {
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
  }
};

} // namespace
} // namespace mlir::iree_compiler
