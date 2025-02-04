// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
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

static void populateErfPattern(RewritePatternSet &patterns) {
  if (clNativeMathPrecision) {
    patterns.add<math::ErfPolynomialApproximation>(patterns.getContext());
  } else {
    populateExpandExp2FPattern(patterns);
    populateMathPolynomialApproximationPatterns(patterns);
    populateExpandRoundEvenPattern(patterns);
  }
}

/// math dialect elementry functions -> polynomial form.
class PolynomialApproximationPass final
    : public impl::PolynomialApproximationPassBase<
          PolynomialApproximationPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    using PatternFunction = llvm::function_ref<void(RewritePatternSet &)>;
    // Order matters here.
    llvm::SmallVector<std::pair<StringRef, PatternFunction>> patternMap = {
        {"tan", populateExpandTanPattern},
        {"sinh", populateExpandSinhPattern},
        {"cosh", populateExpandCoshPattern},
        {"asinh", populateExpandAsinhPattern},
        {"acosh", populateExpandAcoshPattern},
        {"atanh", populateExpandAtanhPattern},
        {"powf", populateExpandPowFPattern},
        {"fpowi", populateExpandFPowIPattern},
        {"erf", populateErfPattern},
    };

    RewritePatternSet mathPatterns(&getContext());

    for (const auto &[fnName, populateFn] : patternMap) {
      // Skip any ops in the "do not convert" list.
      if (!llvm::is_contained(noApproxOps, fnName)) {
        populateFn(mathPatterns);
      }
    }

    if (failed(
            applyPatternsGreedily(getOperation(), std::move(mathPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
