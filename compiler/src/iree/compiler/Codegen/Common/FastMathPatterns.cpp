// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/FastMathPatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

namespace {

// Helper to evaluate a polynomial using FMA chains for both scalar and vector
// types.
static Value evalErfPolynomial(Value x, Value t, ArrayRef<Value> coeffs,
                               RewriterBase &rewriter, Location loc) {
  Value acc = coeffs[0];
  for (size_t i = 1; i < coeffs.size(); ++i) {
    acc = math::FmaOp::create(rewriter, loc, t, acc, coeffs[i]);
  }
  return math::FmaOp::create(rewriter, loc, x, acc, x);
}

// Pattern to lower math.erf to its device lib implementation
// (from
// https://github.com/ROCm/llvm-project/blob/amd-staging/amd/device-libs/ocml/src/erfF.cl#L11)
struct FastErfPattern : public OpRewritePattern<math::ErfOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(math::ErfOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getOperand();
    Type resultType = op.getType();

    VectorType resVecType = dyn_cast<VectorType>(resultType);
    if (!(resultType.isF32() ||
          (resVecType && resVecType.getElementType().isF32()))) {
      return rewriter.notifyMatchFailure(
          op, "result only supports f32 or vector types with f32 element type");
    }

    // Helper to create constants for both scalar and vector types.
    auto createConst = [&](float v) -> Value {
      if (resVecType) {
        SmallVector<Attribute> values(resVecType.getNumElements(),
                                      rewriter.getF32FloatAttr(v));
        return arith::ConstantOp::create(
            rewriter, loc, resVecType,
            DenseElementsAttr::get(resVecType, values));
      } else {
        return arith::ConstantOp::create(rewriter, loc, rewriter.getF32Type(),
                                         rewriter.getF32FloatAttr(v));
      }
    };

    Value one = createConst(1.0f);
    Value ax = math::AbsFOp::create(rewriter, loc, input);
    Value cmp = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OLT,
                                      ax, one);

    // Coefficients for |x| < 1.0.
    const SmallVector<Value, 6> coeffs1 = {
        createConst(-0x1.268bc2p-11f), createConst(0x1.420828p-8f),
        createConst(-0x1.b5937p-6f),   createConst(0x1.ce077cp-4f),
        createConst(-0x1.81266p-2f),   createConst(0x1.06eba0p-3f)};
    // Coefficients for |x| >= 1.0.
    const SmallVector<Value, 7> coeffs2 = {
        createConst(0x1.1d3156p-16f), createConst(-0x1.8d129p-12f),
        createConst(0x1.f9a6d2p-9f),  createConst(-0x1.8c3164p-6f),
        createConst(0x1.b4e9c8p-4f),  createConst(0x1.4515fap-1f),
        createConst(0x1.078e50p-3f)};

    // Select between the two results using scf.if.
    Value result;
    if (resultType.isF32()) {
      // For scalar types, use scf.if.
      auto ifOp = scf::IfOp::create(rewriter, loc, resultType, cmp,
                                    /*withElseRegion=*/true);

      // Then region: |x| < 1.0 - evaluate polynomial.
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value t = arith::MulFOp::create(rewriter, loc, ax, ax);
      Value result1 = evalErfPolynomial(ax, t, coeffs1, rewriter, loc);
      scf::YieldOp::create(rewriter, loc, result1);

      // Else region: |x| >= 1.0 - evaluate different polynomial and
      // post-processing.
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      Value p2 = evalErfPolynomial(ax, ax, coeffs2, rewriter, loc);
      Value negP2 = arith::NegFOp::create(rewriter, loc, p2);
      Value expNegP2 = math::ExpOp::create(rewriter, loc, negP2);
      Value result2 = arith::SubFOp::create(rewriter, loc, one, expNegP2);
      scf::YieldOp::create(rewriter, loc, result2);

      rewriter.setInsertionPointAfter(ifOp);
      result = ifOp.getResult(0);
    } else {
      // For vector types, we need to handle conditional logic differently
      // since we can't use scf.if with vector boolean conditions.
      // We'll compute both polynomials but use vector operations to handle
      // the conditional logic element-wise.

      // Compute t = ax * ax for the first polynomial.
      Value t = arith::MulFOp::create(rewriter, loc, ax, ax);

      // Compute first polynomial (for |x| < 1.0).
      Value result1 = evalErfPolynomial(ax, t, coeffs1, rewriter, loc);

      // Compute second polynomial (for |x| >= 1.0).
      Value p2 = evalErfPolynomial(ax, ax, coeffs2, rewriter, loc);
      Value negP2 = arith::NegFOp::create(rewriter, loc, p2);
      Value expNegP2 = math::ExpOp::create(rewriter, loc, negP2);
      Value result2 = arith::SubFOp::create(rewriter, loc, one, expNegP2);

      // Select between the two results based on the condition.
      result = arith::SelectOp::create(rewriter, loc, cmp, result1, result2);
    }

    // Restore the sign.
    Value finalResult = math::CopySignOp::create(rewriter, loc, result, input);
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};
} // anonymous namespace

void populateFastMathPatterns(RewritePatternSet &patterns,
                              const std::function<bool(StringRef)> &predicate) {
  MLIRContext *ctx = patterns.getContext();

  if (predicate(math::ErfOp::getOperationName())) {
    patterns.add<FastErfPattern>(ctx);
  }

  // TODO: Add more device-lib implementations for other math functions as
  // needed. Some candidates:
  //   - math.tan
  //   - math.sinh/cosh/tanh
  //   - math.log/log2/log1p
  //   - math.exp/expm1
}

} // namespace mlir::iree_compiler
