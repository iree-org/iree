// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_EXPANDF16OPTOF32PASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

/// A pattern that expands floating-point arithmetic/math operations with f16
/// operands to f32 operands. It performs the expansion by extending the
/// f16 operands to f32, performing the arithmetic operation on the extended
/// operands, and then truncating the result back to f16.
template <typename Op>
struct ExpandF16OpToF32Pattern : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto isElemF16Type = [](Type t) { return getElementTypeOrSelf(t).isF16(); };
    Type resultType = op.getResult().getType();
    if (!isElemF16Type(resultType)) {
      return failure();
    }

    Location loc = op.getLoc();
    Type f32Type = rewriter.getF32Type();
    SmallVector<Value> operands;
    for (auto operand : op.getOperands()) {
      if (!isElemF16Type(operand.getType())) {
        operands.push_back(operand);
        continue;
      }
      Value ext = rewriter.create<arith::ExtFOp>(loc, f32Type, operand);
      operands.push_back(ext);
    }
    Value newOp = rewriter.create<Op>(loc, f32Type, operands);

    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, resultType, newOp);
    return success();
  }
};

struct ExpandF16OpToF32Pass
    : public impl::ExpandF16OpToF32PassBase<ExpandF16OpToF32Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ExpandF16OpToF32Pattern<arith::MaximumFOp>>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
