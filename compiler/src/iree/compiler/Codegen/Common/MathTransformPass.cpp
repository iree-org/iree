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

/// Deprecated! This flag had buggy/unintentional semantics.
/// Its original comment said:
/// ""use native hardware operations instead of polynomial approximation".
static llvm::cl::opt<bool> clNativeMathPrecision(
    "iree-codegen-gpu-native-math-precision",
    llvm::cl::desc("Deprecated! This flag had buggy/unintentional semantics. "
                   "Its original description said: \"Skip polynomial lowering "
                   "for math op natively available on GPU.\""),
    llvm::cl::init(false));

#define GEN_PASS_DEF_MATHTRANSFORMPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

static void populateMathFunctionsRewritePatterns(
    RewritePatternSet &patterns,
    const std::function<bool(StringRef)> &predicate) {
  if (predicate(math::TanOp::getOperationName())) {
    populateExpandTanPattern(patterns);
  }
  if (predicate(math::SinhOp::getOperationName())) {
    populateExpandSinhPattern(patterns);
  }
  if (predicate(math::CoshOp::getOperationName())) {
    populateExpandCoshPattern(patterns);
  }
  if (predicate(math::AsinhOp::getOperationName())) {
    populateExpandAsinhPattern(patterns);
  }
  if (predicate(math::AcoshOp::getOperationName())) {
    populateExpandAcoshPattern(patterns);
  }
  if (predicate(math::AtanhOp::getOperationName())) {
    populateExpandAtanhPattern(patterns);
  }
  if (predicate(math::PowFOp::getOperationName())) {
    populateExpandPowFPattern(patterns);
  }
  if (predicate(math::FPowIOp::getOperationName())) {
    populateExpandFPowIPattern(patterns);
  }
  if (predicate(math::Exp2Op::getOperationName())) {
    populateExpandExp2FPattern(patterns);
  }
  if (predicate(math::RoundEvenOp::getOperationName())) {
    populateExpandRoundEvenPattern(patterns);
  }
}

static bool predicateRewrite(StringRef name,
                             IREE::HAL::ExecutableTargetAttr target) {
  if (clNativeMathPrecision) { // Legacy.
    if (name == math::Exp2Op::getOperationName() ||
        name == math::RoundEvenOp::getOperationName()) {
      return false;
    }
  }
  if (isROCMBackend(target)) {
    // On ROCm, we want to use device library functions.
    return false;
  }
  // Currently enable all non-approximative rewrites.
  return true;
}

static bool predicateF32Cast(StringRef name,
                             IREE::HAL::ExecutableTargetAttr target) {
  (void)target;                // Currently unused.
  if (clNativeMathPrecision) { // Legacy.
    return false;
  }
  StringRef atan = math::AtanOp::getOperationName();
  StringRef atan2 = math::Atan2Op::getOperationName();
  StringRef cos = math::CosOp::getOperationName();
  StringRef sin = math::SinOp::getOperationName();
  StringRef tanh = math::TanhOp::getOperationName();
  StringRef log = math::LogOp::getOperationName();
  StringRef log2 = math::Log2Op::getOperationName();
  StringRef log1p = math::Log1pOp::getOperationName();
  StringRef exp = math::ExpOp::getOperationName();
  StringRef expm1 = math::ExpM1Op::getOperationName();
  StringRef cbrt = math::CbrtOp::getOperationName();
  StringRef erf = math::ErfOp::getOperationName();
  return llvm::is_contained(
      {atan, atan2, tanh, log, log2, log1p, erf, exp, expm1, cbrt, sin, cos},
      name);
}

static bool predicateApprox(StringRef name,
                            IREE::HAL::ExecutableTargetAttr target) {
  if (clNativeMathPrecision) { // Legacy.
    if (name == math::ErfOp::getOperationName()) {
      // The legacy implementation had a bug: it always applied polynomial
      // approximation of math.erf, even when clNativeMathPrecision was passed.
      // We actually have CI tests that rely on that bug: they pass
      // clNativeMathPrecision but fail unless math.erf is approximated.
      return true;
    }
    return false;
  }
  if (isROCMBackend(target)) {
    // On ROCm, we want to use device library functions.
    return false;
  }
  StringRef acos = math::AcosOp::getOperationName();
  StringRef asin = math::AsinOp::getOperationName();
  StringRef atan = math::AtanOp::getOperationName();
  StringRef atan2 = math::Atan2Op::getOperationName();
  StringRef cos = math::CosOp::getOperationName();
  StringRef sin = math::SinOp::getOperationName();
  StringRef tanh = math::TanhOp::getOperationName();
  StringRef log = math::LogOp::getOperationName();
  StringRef log2 = math::Log2Op::getOperationName();
  StringRef log1p = math::Log1pOp::getOperationName();
  StringRef exp = math::ExpOp::getOperationName();
  StringRef expm1 = math::ExpM1Op::getOperationName();
  StringRef cbrt = math::CbrtOp::getOperationName();
  StringRef erf = math::ErfOp::getOperationName();
  return llvm::is_contained({atan, atan2, tanh, log, log2, log1p, erf, asin,
                             acos, exp, expm1, cbrt, sin, cos},
                            name);
}

namespace {

class MathTransformPass final
    : public impl::MathTransformPassBase<MathTransformPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    auto target = IREE::HAL::ExecutableTargetAttr::lookup(getOperation());
    if (!target) {
      return signalPassFailure();
    }
    populateMathFunctionsRewritePatterns(patterns, [target](StringRef name) {
      return predicateRewrite(name, target);
    });

    populateMathF32ExpansionPatterns(patterns, [target](StringRef name) {
      return predicateF32Cast(name, target);
    });

    populateMathPolynomialApproximationPatterns(
        patterns,
        [target](StringRef name) { return predicateApprox(name, target); });

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
