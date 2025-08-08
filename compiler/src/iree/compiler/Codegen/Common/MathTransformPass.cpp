// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/FastMathPatterns.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

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
  if (name == math::FPowIOp::getOperationName()) {
    // math.fpowi is a special op: it isn't really a "math function", rather
    // it is generally used with a constant exponent that is a small integer,
    // and it is then a shorthand for a few multiplications. That rewrite needs
    // to happen to prevent falling back on a more expensive, more general
    // implementation like math.powf.
    return true;
  }
  if (isROCMBackend(target)) {
    // On ROCm, we do not need most rewrites as we can generally bottom out on
    // either device library functions, or handling of intrinsics in AMDGPU.
    return false;
  }
  if (isWebGPUBackend(target)) {
    // https://github.com/gpuweb/gpuweb/issues/5109 means we get compilation
    // errors whenever Inf or NaN values arise at compile-time, which is not
    // something that we can really prevent. Avoiding this rewrite helps a bit.
    return false;
  }
  // Currently enable all non-approximative rewrites.
  return true;
}

static bool predicateF32Cast(StringRef name,
                             IREE::HAL::ExecutableTargetAttr target) {
  (void)target; // Currently unused.
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
  if (isROCMBackend(target)) {
    // On ROCm, we do not need most rewrites as we can generally bottom out on
    // either device library functions, or handling of intrinsics in AMDGPU.
    return false;
  }
  if (isWebGPUBackend(target)) {
    // https://github.com/gpuweb/gpuweb/issues/5109 means we get compilation
    // errors whenever Inf or NaN values arise at compile-time, which is not
    // something that we can really prevent. Avoiding this rewrite helps a bit.
    return false;
  }

  // Compute hasFastExp from target attribute.
  bool hasFastExp = isROCMBackend(target);

  // Continue with the existing list for standard approximations.
  StringRef acos = math::AcosOp::getOperationName();
  StringRef asin = math::AsinOp::getOperationName();
  StringRef atan = math::AtanOp::getOperationName();
  StringRef atan2 = math::Atan2Op::getOperationName();
  StringRef cos = math::CosOp::getOperationName();
  StringRef erf = math::ErfOp::getOperationName();
  StringRef sin = math::SinOp::getOperationName();
  StringRef tanh = math::TanhOp::getOperationName();
  StringRef log = math::LogOp::getOperationName();
  StringRef log2 = math::Log2Op::getOperationName();
  StringRef log1p = math::Log1pOp::getOperationName();
  StringRef exp = math::ExpOp::getOperationName();
  StringRef expm1 = math::ExpM1Op::getOperationName();
  StringRef cbrt = math::CbrtOp::getOperationName();

  // List of ops that have specific device library implementations enabled by
  // hasFastExp.
  StringRef opsWithDeviceLibImpl[] = {erf};

  // If hasFastExp is enabled and the op is in our device-lib list,
  // don't apply the standard polynomial approximation.
  if (hasFastExp && llvm::is_contained(opsWithDeviceLibImpl, name)) {
    return false;
  }

  return llvm::is_contained({atan, atan2, tanh, log, log2, log1p, erf, asin,
                             acos, exp, expm1, cbrt, sin, cos},
                            name);
}

// Returns true if the given function should be handled by a fast math pattern.
static bool predicateDeviceLibImpl(StringRef name,
                                   IREE::HAL::ExecutableTargetAttr target) {
  // Compute hasFastExp from target attribute.
  bool hasFastExp = isROCMBackend(target);

  // If fast exp is not available, don't use device-lib implementations.
  if (!hasFastExp)
    return false;

  // Only apply to erf for now.
  StringRef erf = math::ErfOp::getOperationName();
  return llvm::is_contained({erf}, name);
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

    populateFastMathPatterns(patterns, [target](StringRef name) {
      return predicateDeviceLibImpl(name, target);
    });

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
