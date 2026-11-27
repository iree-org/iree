// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/FastMathPatterns.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MATHTRANSFORMPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

static void populateMathFunctionsRewritePatterns(
    RewritePatternSet &patterns,
    const std::function<bool(StringRef)> &predicate) {
  llvm::SmallVector<StringRef> opNames,
      opFullNames = {math::TanOp::getOperationName(),
                     math::SinhOp::getOperationName(),
                     math::CoshOp::getOperationName(),
                     math::AsinhOp::getOperationName(),
                     math::AcoshOp::getOperationName(),
                     math::AtanhOp::getOperationName(),
                     math::PowFOp::getOperationName(),
                     math::FPowIOp::getOperationName(),
                     math::Exp2Op::getOperationName(),
                     math::RoundEvenOp::getOperationName()};
  size_t prefix = math::MathDialect::getDialectNamespace().size() + 1;
  for (StringRef name : opFullNames) {
    if (predicate(name)) {
      opNames.push_back(name.drop_front(prefix));
    }
  }
  math::populateExpansionPatterns(patterns, opNames);
}

// The LLVM MIPS backend miscompiles the *vectorized* polynomial approximation
// of these transcendentals once MSA (128-bit SIMD) is enabled; the scalar form
// is correct. We scalarize these ops on MIPS (see MathTransformPass below) so
// the approximations expand into correct scalar arithmetic — keeping modules
// hermetic (for the embedded-ELF loader) instead of falling back to libm.
//
// This is the set of base transcendentals whose vectorized approximation was
// empirically confirmed wrong under MSA (differential vs host). It also covers
// the composite ops that decompose into them (e.g. sinh/cosh/tan/asinh/acosh
// -> exp/log, atan2/asin/acos -> atan) because the scalarization shares the
// greedy run with the approximations and runs at a higher benefit, so a
// transcendental produced mid-approximation is scalarized before it is itself
// approximated. cbrt is intentionally excluded: its vectorized approximation
// was verified correct, so it stays fully MSA-vectorized.
static bool isMIPSScalarizedTranscendental(StringRef name) {
  return llvm::is_contained(
      {math::ExpOp::getOperationName(), math::ExpM1Op::getOperationName(),
       math::LogOp::getOperationName(), math::Log2Op::getOperationName(),
       math::Log1pOp::getOperationName(), math::SinOp::getOperationName(),
       math::CosOp::getOperationName(), math::TanhOp::getOperationName(),
       math::ErfOp::getOperationName(), math::AtanOp::getOperationName(),
       math::AsinOp::getOperationName(), math::AcosOp::getOperationName(),
       math::AsinhOp::getOperationName(), math::AcoshOp::getOperationName()},
      name);
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
  if (name == math::RoundEvenOp::getOperationName() &&
      hasNativeRoundEven(target)) {
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
    // cbrt has no WGSL builtin and its expansion uses only finite constants.
    return name == math::CbrtOp::getOperationName();
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

  // NOTE: on MIPS the vectorized form of several of these approximations is
  // miscompiled by the LLVM MSA backend (e.g. exp(1) -> 7.43, erf(1) -> 0.655);
  // we keep the approximation (so modules stay hermetic for the embedded-ELF
  // loader) and instead scalarize the affected ops before this pass runs, where
  // the scalar form is correct. See isMIPSScalarizedTranscendental and
  // MathTransformPass::runOnOperation.
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
  if (!hasFastExp) {
    return false;
  }

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
    auto target = IREE::HAL::ExecutableTargetAttr::lookup(getOperation());

    RewritePatternSet patterns(&getContext());
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

    // On MIPS the LLVM MSA backend miscompiles the *vectorized* polynomial
    // approximation of several transcendentals (the scalar form is correct).
    // Scalarize those ops to single lanes so the approximation above produces
    // correct scalar arithmetic: a <1xf32> math op lowers to scalar (no MSA).
    // This keeps the module hermetic (no external libm — required by the
    // embedded-ELF loader) at the same cost as the old libm fallback, which was
    // already scalar; matmul/conv and all other elementwise ops keep full MSA.
    // The pattern is given a higher benefit and shares this greedy run with the
    // approximations so it also catches transcendentals the approximations
    // *generate* (e.g. atan2 -> atan), scalarizing them before they themselves
    // get approximated into the miscompiled vector form.
    if (target && isMIPS(target.getConfiguration())) {
      vector::UnrollVectorOptions unrollOptions;
      unrollOptions.setNativeShapeFn(
          [](Operation *op) -> std::optional<SmallVector<int64_t>> {
            if (!isMIPSScalarizedTranscendental(op->getName().getStringRef())) {
              return std::nullopt;
            }
            auto vecType = dyn_cast<VectorType>(op->getResult(0).getType());
            if (!vecType || vecType.isScalable() ||
                vecType.getNumElements() == 1) {
              return std::nullopt;
            }
            return SmallVector<int64_t>(vecType.getRank(), 1);
          });
      vector::populateVectorUnrollPatterns(patterns, unrollOptions,
                                           /*benefit=*/10);
    }

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
