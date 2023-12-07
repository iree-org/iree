// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
    Type resultType = op.getLhs().getType();
    if (getElementTypeOrSelf(resultType).getIntOrFloatBitWidth() != 16) {
      return failure();
    }

    Location loc = op.getLoc();

    Type wideType = rewriter.getF32Type();
    if (auto vecTy = resultType.dyn_cast<VectorType>()) {
      wideType = VectorType::get(vecTy.getShape(), wideType);
    }

    Value lhsExt = rewriter.create<arith::ExtFOp>(loc, wideType, op.getLhs());
    Value rhsExt = rewriter.create<arith::ExtFOp>(loc, wideType, op.getRhs());
    Value maxExt = rewriter.create<Op>(loc, wideType, lhsExt, rhsExt);

    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, resultType, maxExt);
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
