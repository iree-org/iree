// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MATHTRANSFORMPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

/// Deprecated! This flag had buggy/unintentional semantics.
/// Its original comment said:
/// ""use native hardware operations instead of polynomial approximation".
static llvm::cl::opt<bool> clNativeMathPrecision(
    "iree-codegen-gpu-native-math-precision",
    llvm::cl::desc("Deprecated! This flag had buggy/unintentional semantics. "
                   "Its original description said: \"Skip polynomial lowering "
                   "for math op natively available on GPU.\""),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> clMathTransformTweaks(
    "iree-codegen-math-transform-tweaks",
    llvm::cl::desc(
        "Comma-separated list of tweaks to apply to default math transforms. "
        "Each entry has the form transform:+op or transform:-op to "
        "enable/disable a transform on an op. Allowed values for `op` are the "
        "math dialect ops, math.cos. Allowed values for `transform` "
        "are: rewrite, approx, f32cast. Example: approx:-math.cos ."),
    llvm::cl::init(""));

struct MathTransformTweaks {
  // Ops to enable "rewrite" transformations on. See `predicateRewrite`.
  SmallVector<std::string> enableRewrite;
  // Ops to disable "rewrite" transformations on. See `predicateRewrite`.
  SmallVector<std::string> disableRewrite;
  // Ops to enable "f32cast" transformations on. See `predicateF32Cast`.
  SmallVector<std::string> enableF32Cast;
  // Ops to disable "f32cast" transformations on. See `predicateF32Cast`.
  SmallVector<std::string> disableF32Cast;
  // Ops to enable "approx" transformations on. See `predicateApprox`.
  SmallVector<std::string> enableApprox;
  // Ops to disable "approx" transformations on. See `predicateApprox`.
  SmallVector<std::string> disableApprox;
};

static llvm::Expected<MathTransformTweaks>
getMathTransformTweaksFromString(StringRef tweaksString) {
  auto error = [](auto... args) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   llvm::formatv(args...));
  };
  if (tweaksString.empty()) {
    // Default case, no tweaks, return an default-constructed object.
    return MathTransformTweaks{};
  }
  MathTransformTweaks tweaks;
  SmallVector<StringRef> flags;
  tweaksString.split(flags, ',');
  for (StringRef flag : flags) {
    if (flag.empty()) {
      return error("Empty entry in comma-separated list");
    }
    auto [transform, plusMinusOp] = flag.split(':');
    if (plusMinusOp.empty()) {
      return error("No `:` separator found in `{}`", flag);
    }
    if (plusMinusOp.size() < 2) {
      return error("Expected at least 2 characters to the right of `:` in `{}`",
                   flag);
    }
    StringRef op = plusMinusOp.ltrim("-+");
    if (op.size() + 1 != plusMinusOp.size()) {
      // Check that exactly one left prefix char is '-' or '+'.
      return error(
          "Expected exactly one + or - character on the right of : in `{}`",
          flag);
    }
    char plusMinus = plusMinusOp[0];
    // User input already validated. This assert is just an internal check.
    assert(plusMinus == '+' || plusMinus == '-');
    SmallVector<std::string> *dstVecPtr =
        llvm::StringSwitch<SmallVector<std::string> *>(transform)
            .Case("rewrite", plusMinus == '+' ? &tweaks.enableRewrite
                                              : &tweaks.disableRewrite)
            .Case("f32cast", plusMinus == '+' ? &tweaks.enableF32Cast
                                              : &tweaks.disableF32Cast)
            .Case("approx", plusMinus == '+' ? &tweaks.enableApprox
                                             : &tweaks.disableApprox)
            .Default(nullptr);
    if (!dstVecPtr) {
      return error("Unknown key on left of the : separator in {}", flag);
    }
    dstVecPtr->emplace_back(op);
  }
  return tweaks;
}

static llvm::Expected<MathTransformTweaks>
getMathTransformTweaksFromCLAndReportError() {
  llvm::Expected<MathTransformTweaks> tweaks =
      getMathTransformTweaksFromString(clMathTransformTweaks);
  if (!tweaks) {
    clMathTransformTweaks.error(llvm::toString(tweaks.takeError()));
  }
  return tweaks;
}

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
                             IREE::HAL::ExecutableTargetAttr target,
                             const MathTransformTweaks &tweaks) {
  (void)target;                // Currently unused.
  if (clNativeMathPrecision) { // Legacy.
    if (name == math::Exp2Op::getOperationName() ||
        name == math::RoundEvenOp::getOperationName()) {
      return false;
    }
  }
  if (llvm::is_contained(tweaks.enableRewrite, name)) {
    return true;
  }
  if (llvm::is_contained(tweaks.disableRewrite, name)) {
    return false;
  }
  // Currently enable all non-approximative rewrites.
  return true;
}

static bool predicateF32Cast(StringRef name,
                             IREE::HAL::ExecutableTargetAttr target,
                             const MathTransformTweaks &tweaks) {
  (void)target;                // Currently unused.
  if (clNativeMathPrecision) { // Legacy.
    return false;
  }
  if (llvm::is_contained(tweaks.enableF32Cast, name)) {
    return true;
  }
  if (llvm::is_contained(tweaks.disableF32Cast, name)) {
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
                            IREE::HAL::ExecutableTargetAttr target,
                            const MathTransformTweaks &tweaks) {
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
  if (llvm::is_contained(tweaks.enableApprox, name)) {
    return true;
  }
  if (llvm::is_contained(tweaks.disableApprox, name)) {
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
  if (isROCMBackend(target) && name == erf) {
    return false;
  }
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
    // Parse tweaks from command-line flag. Static initializer is thread-safe
    // and ensures that the parsing and any error-printing is done only once.
    static llvm::Expected<MathTransformTweaks> tweaks =
        getMathTransformTweaksFromCLAndReportError();
    if (!tweaks) {
      // Any error message has already been printed.
      return signalPassFailure();
    }
    RewritePatternSet patterns(&getContext());
    auto target = IREE::HAL::ExecutableTargetAttr::lookup(getOperation());
    if (!target) {
      return signalPassFailure();
    }
    populateMathFunctionsRewritePatterns(patterns, [&](StringRef name) {
      return predicateRewrite(name, target, *tweaks);
    });

    populateMathF32ExpansionPatterns(patterns, [&](StringRef name) {
      return predicateF32Cast(name, target, *tweaks);
    });

    populateMathPolynomialApproximationPatterns(patterns, [&](StringRef name) {
      return predicateApprox(name, target, *tweaks);
    });
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
