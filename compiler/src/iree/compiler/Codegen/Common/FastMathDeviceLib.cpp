// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/FastMathDeviceLib.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

namespace {
// Pattern to lower math.erf to its device lib implementation (from erfF.cl)
// Pattern to lower math.erf to its device lib implementation (from erfF.cl)
struct ErfDeviceLibPattern : public OpRewritePattern<math::ErfOp> {
  using OpRewritePattern<math::ErfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ErfOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getOperand();
    Type resultType = op.getType();

    // Create constants
    auto f32Type = rewriter.getF32Type();
    auto zeroF = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(0.0f));
    auto oneF = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(1.0f));
    
    // Define polynomial coefficients for |x| < 1.0
    auto c1_0 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(-0x1.268bc2p-11f));
    auto c1_1 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(0x1.420828p-8f));
    auto c1_2 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(-0x1.b5937p-6f));
    auto c1_3 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(0x1.ce077cp-4f));
    auto c1_4 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(-0x1.81266p-2f));
    auto c1_5 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(0x1.06eba0p-3f));
    
    // Define polynomial coefficients for |x| >= 1.0
    auto c2_0 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(0x1.1d3156p-16f));
    auto c2_1 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(-0x1.8d129p-12f));
    auto c2_2 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(0x1.f9a6d2p-9f));
    auto c2_3 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(-0x1.8c3164p-6f));
    auto c2_4 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(0x1.b4e9c8p-4f));
    auto c2_5 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(0x1.4515fap-1f));
    auto c2_6 = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(0x1.078e50p-3f));
    
    // Get abs value
    Value ax = rewriter.create<math::AbsFOp>(loc, input);
    
    // Create comparison for |x| < 1.0
    Value cmp = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, ax, oneF);
    
    // Create if statement
    auto ifOp = rewriter.create<scf::IfOp>(loc, resultType, cmp, true);
    
    // Set up the "then" region for |x| < 1.0
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    
    // Polynomial approximation for |x| < 1.0
    Value t = rewriter.create<arith::MulFOp>(loc, ax, ax);
    
    // First nested MAD: MATH_MAD(t, -0x1.268bc2p-11f, 0x1.420828p-8f)
    Value mad1 = rewriter.create<arith::FmaOp>(loc, t, c1_0, c1_1);
    
    // Second nested MAD: MATH_MAD(t, mad1, -0x1.b5937p-6f)
    Value mad2 = rewriter.create<arith::FmaOp>(loc, t, mad1, c1_2);
    
    // Third nested MAD: MATH_MAD(t, mad2, 0x1.ce077cp-4f)
    Value mad3 = rewriter.create<arith::FmaOp>(loc, t, mad2, c1_3);
    
    // Fourth nested MAD: MATH_MAD(t, mad3, -0x1.81266p-2f)
    Value mad4 = rewriter.create<arith::FmaOp>(loc, t, mad3, c1_4);
    
    // Fifth nested MAD: MATH_MAD(t, mad4, 0x1.06eba0p-3f)
    Value p = rewriter.create<arith::FmaOp>(loc, t, mad4, c1_5);
    
    // Final BUILTIN_FMA_F32(ax, p, ax)
    Value result = rewriter.create<arith::FmaOp>(loc, ax, p, ax);
    
    rewriter.create<scf::YieldOp>(loc, result);
    
    // Set up the "else" region for |x| >= 1.0
    rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
    
    // Complex polynomial approximation for |x| >= 1.0
    // Start with MATH_MAD(ax, 0x1.1d3156p-16f, -0x1.8d129p-12f)
    Value mad5 = rewriter.create<arith::FmaOp>(loc, ax, c2_0, c2_1);
    
    // MATH_MAD(ax, mad5, 0x1.f9a6d2p-9f)
    Value mad6 = rewriter.create<arith::FmaOp>(loc, ax, mad5, c2_2);
    
    // MATH_MAD(ax, mad6, -0x1.8c3164p-6f)
    Value mad7 = rewriter.create<arith::FmaOp>(loc, ax, mad6, c2_3);
    
    // MATH_MAD(ax, mad7, 0x1.b4e9c8p-4f)
    Value mad8 = rewriter.create<arith::FmaOp>(loc, ax, mad7, c2_4);
    
    // MATH_MAD(ax, mad8, 0x1.4515fap-1f)
    Value mad9 = rewriter.create<arith::FmaOp>(loc, ax, mad8, c2_5);
    
    // MATH_MAD(ax, mad9, 0x1.078e50p-3f)
    Value mad10 = rewriter.create<arith::FmaOp>(loc, ax, mad9, c2_6);
    
    // BUILTIN_FMA_F32(ax, mad10, ax)
    Value p2 = rewriter.create<arith::FmaOp>(loc, ax, mad10, ax);
    
    // Negate p2
    Value negP2 = rewriter.create<arith::NegFOp>(loc, p2);
    
    // Compute exp(-p2)
    Value expNegP2 = rewriter.create<math::ExpOp>(loc, negP2);
    
    // 1.0f - exp(-p2)
    Value result2 = rewriter.create<arith::SubFOp>(loc, oneF, expNegP2);
    
    rewriter.create<scf::YieldOp>(loc, result2);
    
    // Set insertion point after the if
    rewriter.setInsertionPointAfter(ifOp);
    
    // Restore the sign: BUILTIN_COPYSIGN_F32(ret, x)
    Value finalResult = rewriter.create<math::CopySignOp>(loc, ifOp.getResult(0), input);
    
    // Replace the original op with our implementation
    rewriter.replaceOp(op, finalResult);
    
    return success();
  }
};

} // namespace

void populateDeviceLibMathPatterns(
    RewritePatternSet &patterns,
    const std::function<bool(StringRef)> &predicate) {
  MLIRContext *ctx = patterns.getContext();
  
  // Add the device-lib erf pattern if enabled
  if (predicate(math::ErfOp::getOperationName())) {
    patterns.add<ErfDeviceLibPattern>(ctx);
  }
  
  // TODO: Add more device-lib implementations for other math functions as needed.
  // Some candidates:
  //   - math.tan
  //   - math.sinh/cosh/tanh
  //   - math.log/log2/log1p
  //   - math.exp/expm1
}

} // namespace mlir::iree_compiler
