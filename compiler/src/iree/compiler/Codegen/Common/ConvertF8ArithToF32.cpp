// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- BubbleUpOrdinalOpPass.cpp -----------------------------------------===//
//
// The workgroup count computation when using slices needs the ordinal
// annotation ops to be bubbled up as much as possible. This pass implements
// patterns to bubble these operations up.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTF8ARITHTOF32PASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Replace the following sequence
///
/// ```mlir
/// %1 = arith.negf %input : vector<1x2x1x1x1x1xf8E4M3FNUZ>
/// ```
///
/// with
///
/// ```mlir
/// %0 = arith.extf %input : f8E4M3FNUZ to f32
/// %1 = arith.negf %0 : vector<1x2x1x1x1x1xf32>
/// %2 = arith.truncf %1 : vector<1x2x1x1x1x1xf8E4M3FNUZ>
/// ```
///
/// to make all the uses flow through `flow.dispatch.workload.ordinal` ops.
template <typename OpTy>
struct F8ArithToF32CastOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {

    auto inputType = op.getOperand().getType().cast<VectorType>();
    if (inputType.getElementType().isF8E4M3FNUZ()) {
      // Extend the input to f32
      auto extendedType = inputType.clone(rewriter.getF32Type());
      auto extended = rewriter.create<arith::ExtFOp>(op.getLoc(), extendedType,
                                                     op.getOperand());

      // Negate the extended value
      auto negated = rewriter.create<OpTy>(op.getLoc(), extended);

      // Truncate back to f8E4M3FNUZ
      auto truncated =
          rewriter.create<arith::TruncFOp>(op.getLoc(), inputType, negated);

      // Replace the original operation
      rewriter.replaceOp(op, truncated.getResult());
      return success();
    }
    return failure();
  }
};

struct ConvertF8ArithToF32Pass final
    : impl::ConvertF8ArithToF32PassBase<ConvertF8ArithToF32Pass> {
  void runOnOperation() override;
};
} // namespace

void ConvertF8ArithToF32Pass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<F8ArithToF32CastOp<arith::NegFOp>>(context);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler
