// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/Check/IR/CheckOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Check {

namespace {
template <typename SrcOp, typename DstOp>
struct ExpandAttributeToConst : public OpRewritePattern<SrcOp> {
  using OpRewritePattern<SrcOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    auto rhs = rewriter.create<arith::ConstantOp>(op.getLoc(), op.value());
    rewriter.replaceOpWithNewOp<DstOp>(op, op.lhs(), rhs);
    return success();
  }
};
}  // namespace

void ExpectEqConstOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<ExpandAttributeToConst<ExpectEqConstOp, ExpectEqOp>>(context);
}

void ExpectAlmostEqConstOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results
      .insert<ExpandAttributeToConst<ExpectAlmostEqConstOp, ExpectAlmostEqOp>>(
          context);
}

}  // namespace Check
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Modules/Check/IR/CheckOps.cpp.inc"
