// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_COMPILER_UTILS_PATTERNUTILS_H_
#define IREE_COMPILER_UTILS_PATTERNUTILS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

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
static void insertGreedyPattern(OwningRewritePatternList &patterns,
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
      SmallVector<Value, 4> operands;
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
static void insertConversionPattern(OwningRewritePatternList &patterns,
                                    MLIRContext *context,
                                    GenericOpRewritePattern<OpTy> f,
                                    PatternBenefit benefit = 1) {
  struct Pattern : public OpConversionPattern<OpTy> {
    Pattern(MLIRContext *context, GenericOpRewritePattern<OpTy> f,
            PatternBenefit benefit)
        : OpConversionPattern<OpTy>(context, benefit), f(f) {}
    LogicalResult matchAndRewrite(
        OpTy op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
      return f(op, typename OpTy::Adaptor(operands), rewriter);
    }
    GenericOpRewritePattern<OpTy> f;
  };
  patterns.insert<Pattern>(context, f, benefit);
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_PATTERNUTILS_H_
