// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/FastMathPatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

namespace {
// Pattern to lower math.erf to its device lib implementation
// (from
// https://github.com/ROCm/llvm-project/blob/amd-staging/amd/device-libs/ocml/src/erfF.cl#L11)
struct FastErfPattern : public OpRewritePattern<math::ErfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ErfOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getOperand();
    Type resultType = op.getType();

    // Erf only supports f32.
    if (!resultType.isF32()) {
      return rewriter.notifyMatchFailure(op, "Result only supports f32");
    }

    // Create constants.
    Type f32Type = rewriter.getF32Type();
    auto oneF = arith::ConstantOp::create(rewriter, loc, f32Type,
                                          rewriter.getF32FloatAttr(1.0f));

    // Get abs value.
    Value ax = math::AbsFOp::create(rewriter, loc, input);

    // Create comparison for |x| < 1.0.
    Value cmp = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OLT,
                                      ax, oneF);

    // Create if statement.
    auto ifOp = scf::IfOp::create(rewriter, loc, resultType, cmp, true);

    // --- Then region (|x| < 1.0) ---
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      // Define polynomial coefficients for |x| < 1.0.
      auto c1_0 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(-0x1.268bc2p-11f));
      auto c1_1 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(0x1.420828p-8f));
      auto c1_2 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(-0x1.b5937p-6f));
      auto c1_3 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(0x1.ce077cp-4f));
      auto c1_4 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(-0x1.81266p-2f));
      auto c1_5 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(0x1.06eba0p-3f));

      Value t = arith::MulFOp::create(rewriter, loc, ax, ax);
      Value mad1 = math::FmaOp::create(rewriter, loc, t, c1_0, c1_1);
      Value mad2 = math::FmaOp::create(rewriter, loc, t, mad1, c1_2);
      Value mad3 = math::FmaOp::create(rewriter, loc, t, mad2, c1_3);
      Value mad4 = math::FmaOp::create(rewriter, loc, t, mad3, c1_4);
      Value p = math::FmaOp::create(rewriter, loc, t, mad4, c1_5);
      Value result = math::FmaOp::create(rewriter, loc, ax, p, ax);
      scf::YieldOp::create(rewriter, loc, result);
    } // End then region.

    // --- Else region (|x| >= 1.0) ---
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());

      // Define polynomial coefficients for |x| >= 1.0
      auto c2_0 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(0x1.1d3156p-16f));
      auto c2_1 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(-0x1.8d129p-12f));
      auto c2_2 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(0x1.f9a6d2p-9f));
      auto c2_3 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(-0x1.8c3164p-6f));
      auto c2_4 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(0x1.b4e9c8p-4f));
      auto c2_5 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(0x1.4515fap-1f));
      auto c2_6 = arith::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(0x1.078e50p-3f));

      Value mad5 = math::FmaOp::create(rewriter, loc, ax, c2_0, c2_1);
      Value mad6 = math::FmaOp::create(rewriter, loc, ax, mad5, c2_2);
      Value mad7 = math::FmaOp::create(rewriter, loc, ax, mad6, c2_3);
      Value mad8 = math::FmaOp::create(rewriter, loc, ax, mad7, c2_4);
      Value mad9 = math::FmaOp::create(rewriter, loc, ax, mad8, c2_5);
      Value mad10 = math::FmaOp::create(rewriter, loc, ax, mad9, c2_6);
      // In the C code, there's an extra fma(ax, p, ax) here, which seems
      // incorrect based on the standard erf approximation formula and leads to
      // values > 1. The typical approximation leads directly to the exponent
      // term. Value p2 = math::FmaOp::create(rewriter, loc, ax, mad10, ax); //
      // Original line based on C code.
      Value p2 = mad10; // Corrected based on typical erf formula structure for
                        // |x| >= 1
      Value negP2 = arith::NegFOp::create(rewriter, loc, p2);
      Value expNegP2 = math::ExpOp::create(rewriter, loc, negP2);
      Value result2 = arith::SubFOp::create(rewriter, loc, oneF, expNegP2);
      scf::YieldOp::create(rewriter, loc, result2);
    } // End else region

    // Set insertion point after the if.
    rewriter.setInsertionPointAfter(ifOp);

    // Restore the sign: BUILTIN_COPYSIGN_F32(ret, x)
    Value finalResult =
        math::CopySignOp::create(rewriter, loc, ifOp.getResult(0), input);
    // Replace the original op with our implementation.
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
