// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

/// A pattern that expands floating-point arithmetic operations with f16
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
    : public ExpandArithF16ToF32Base<ExpandF16OpToF32Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ExpandF16OpToF32Pattern<arith::MaximumFOp>>(context);
    // TODO(#15661): Remove the expansion for math.powf op after fixing
    // approximation issue.
    patterns.insert<ExpandF16OpToF32Pattern<math::PowFOp>>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> createExpandF16OpToF32Pass() {
  return std::make_unique<ExpandF16OpToF32Pass>();
}

} // namespace mlir::iree_compiler
