// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/LLVMCPUPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct ExpandF16MaxFToF32Pattern : public OpRewritePattern<arith::MaxFOp> {
  public:
    using OpRewritePattern<arith::MaxFOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::MaxFOp op, 
                                  PatternRewriter &rewriter) const override{
      Type resultType = op.getLhs().getType();
      if (getElementTypeOrSelf(resultType).getIntOrFloatBitWidth() != 16) {
        return failure();
      }

      Location loc = op.getLoc();

      Type wideType = rewriter.getF32Type();
      if(auto vecTy = resultType.dyn_cast<VectorType>()) {
        wideType = VectorType::get(vecTy.getShape(), wideType);
      }

      Value lhsExt = rewriter.create<arith::ExtFOp>(loc, wideType, op.getLhs());
      Value rhsExt = rewriter.create<arith::ExtFOp>(loc, wideType, op.getRhs());
      Value maxExt = 
          rewriter.create<arith::MaxFOp>(loc, wideType, lhsExt, rhsExt);
      Value result = rewriter.create<arith::TruncFOp>(loc, resultType, maxExt);

      rewriter.replaceOp(op, result);
      return success();
    }
};

struct ExpandF16MaxFToF32Pass
    : public ExpandArithF16ToF32Base<ExpandF16MaxFToF32Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ExpandF16MaxFToF32Pattern>(
        context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createExpandF16MaxFToF32Pass() {
  return std::make_unique<ExpandF16MaxFToF32Pass>();
}

}  // namespace iree_compiler
}  // namespace mlir
