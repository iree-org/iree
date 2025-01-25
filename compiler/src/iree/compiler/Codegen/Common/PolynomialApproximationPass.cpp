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
    using PatternFunction = std::function<decltype(populateExpandTanPattern)>;
    std::map<std::string, PatternFunction> patternMap = {
        {"tan", populateExpandTanPattern},
        {"sinh", populateExpandSinhPattern},
        {"cosh", populateExpandCoshPattern},
        {"asinh", populateExpandAsinhPattern},
        {"acosh", populateExpandAcoshPattern},
        {"atanh", populateExpandAtanhPattern},
        {"powf", populateExpandPowFPattern},
        {"fpowi", populateExpandFPowIPattern},
        {"erf",
         [&](RewritePatternSet &mathPatterns) {
           this->populateErfPattern(mathPatterns);
         }},
    };

    RewritePatternSet mathPatterns(&getContext());

    std::vector<StringRef> noOps = splitByComma(noApproxOps);
    for (auto &pattern : patternMap) {
      // Skip any ops in the "do not convert" list.
      auto it = std::find(noOps.begin(), noOps.end(), pattern.first);
      if (it != noOps.end()) {
        continue;
      }
      pattern.second(mathPatterns);
    }

    if (failed(
            applyPatternsGreedily(getOperation(), std::move(mathPatterns)))) {
      return signalPassFailure();
    }
  }

  std::vector<StringRef> splitByComma(StringRef str) {
    std::vector<StringRef> result;
    SmallVector<StringRef, 8> parts;
    str.split(parts, ",");
    for (auto part : parts) {
      result.push_back(part.str());
    }
    return result;
  }

  void populateErfPattern(RewritePatternSet &mathPatterns) {
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
