// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/Check/IR/CheckOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Check {

namespace {
// Rewrites expect_eq_const -> expect_eq
struct ExpectEqConstOpToExpectEqOp : public OpRewritePattern<ExpectEqConstOp> {
  using OpRewritePattern<ExpectEqConstOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ExpectEqConstOp op,
                                PatternRewriter &rewriter) const override {
    auto rhs = rewriter.create<arith::ConstantOp>(op.getLoc(), op.getValue());
    rewriter.replaceOpWithNewOp<ExpectEqOp>(op, op.getDevice(), op.getLhs(),
                                            rhs);
    return success();
  }
};

// Rewrites expect_almost_eq_const -> expect_almost_eq
struct ExpectAlmostEqConstOpToExpectAlmostEqOp
    : public OpRewritePattern<ExpectAlmostEqConstOp> {
  using OpRewritePattern<ExpectAlmostEqConstOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ExpectAlmostEqConstOp op,
                                PatternRewriter &rewriter) const override {
    auto rhs = rewriter.create<arith::ConstantOp>(op.getLoc(), op.getValue());
    rewriter.replaceOpWithNewOp<ExpectAlmostEqOp>(
        op, op.getDevice(), op.getLhs(), rhs, op.getToleranceAttr());
    return success();
  }
};

static constexpr char kToleranceKeyword[] = "tolerance";
static constexpr float kToleranceDefaultValue = 1e-4f;

static ParseResult parseOptionalFloatTolerance(OpAsmParser &parser,
                                               FloatAttr &tolerance) {
  float toleranceValue = kToleranceDefaultValue;
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseKeyword(kToleranceKeyword))) {
      return parser.emitError(parser.getCurrentLocation(),
                              llvm::Twine("Expected keyword: ") +
                                  kToleranceKeyword);
    }
    llvm::APFloat parsedTolerance(APFloat::IEEEsingle());
    if (failed(parser.parseFloat(parsedTolerance.getSemantics(),
                                 parsedTolerance))) {
      return parser.emitError(parser.getCurrentLocation(),
                              "Failed to parse optional float tolerance.");
    }
    toleranceValue = parsedTolerance.convertToFloat();
  }
  tolerance = parser.getBuilder().getF32FloatAttr(toleranceValue);
  return success();
}

static void printOptionalFloatTolerance(OpAsmPrinter &p, Operation *op,
                                        FloatAttr tolerance) {
  float toleranceValue = tolerance.getValue().convertToFloat();
  if (toleranceValue != kToleranceDefaultValue) {
    p << ", " << kToleranceKeyword << " " << toleranceValue;
  }
}

} // namespace

void ExpectEqConstOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<ExpectEqConstOpToExpectEqOp>(context);
}

void ExpectAlmostEqConstOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ExpectAlmostEqConstOpToExpectAlmostEqOp>(context);
}

} // namespace mlir::iree_compiler::IREE::Check

#define GET_OP_CLASSES
#include "iree/compiler/Modules/Check/IR/CheckOps.cpp.inc"
