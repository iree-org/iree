// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_PATTERNUTILS_H_
#define IREE_COMPILER_UTILS_PATTERNUTILS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Generic patterns that can either be greedy rewrite patterns or conversion
// patterns. This allows patterns that operate within supported behavior for
// the conversion framework to use the subset of facilities of the
// PatternRewriter while being compatible with type conversion.
//
// TODO(laurenzo): Start a discussion upstream about the best way to be doing
// this.
//===----------------------------------------------------------------------===//

template <typename OpTy>
using GenericOpRewritePattern = LogicalResult (*)(
    OpTy op, typename OpTy::Adaptor operands, PatternRewriter &rewriter);

template <typename OpTy>
static void insertGreedyPattern(RewritePatternSet &patterns,
                                MLIRContext *context,
                                GenericOpRewritePattern<OpTy> f,
                                PatternBenefit benefit = 1) {
  struct Pattern : public OpRewritePattern<OpTy> {
    Pattern(MLIRContext *context, GenericOpRewritePattern<OpTy> f,
            PatternBenefit benefit)
        : OpRewritePattern<OpTy>(context, benefit), f(f) {}
    LogicalResult matchAndRewrite(OpTy op,
                                  PatternRewriter &rewriter) const override {
      // TODO(laurenzo): It would be nice if the operand adaptors did not
      // have a dependency on ArrayRef as it requires doing this copy. In
      // practice for this level of IR, this is sub-optimal but not the end
      // of the world.
      SmallVector<Value> operands;
      for (unsigned i = 0, e = op.getOperation()->getNumOperands(); i < e;
           ++i) {
        operands.push_back(op.getOperation()->getOperand(i));
      }
      return f(op, typename OpTy::Adaptor(operands), rewriter);
    }
    GenericOpRewritePattern<OpTy> f;
  };
  patterns.insert<Pattern>(context, f, benefit);
}

template <typename OpTy>
static void insertConversionPattern(RewritePatternSet &patterns,
                                    MLIRContext *context,
                                    GenericOpRewritePattern<OpTy> f,
                                    PatternBenefit benefit = 1) {
  struct Pattern : public OpConversionPattern<OpTy> {
    Pattern(MLIRContext *context, GenericOpRewritePattern<OpTy> f,
            PatternBenefit benefit)
        : OpConversionPattern<OpTy>(context, benefit), f(f) {}
    LogicalResult
    matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      return f(op, adaptor, rewriter);
    }
    GenericOpRewritePattern<OpTy> f;
  };
  patterns.insert<Pattern>(context, f, benefit);
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_PATTERNUTILS_H_
