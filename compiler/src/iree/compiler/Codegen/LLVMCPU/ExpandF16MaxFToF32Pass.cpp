// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#include "iree/compiler/Codegen/LLVMCPU/LLVMCPUPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
      llvm::outs()<<"\n\n Added \n\n";
      if (getElementTypeOrSelf(resultType).getIntOrFloatBitWidth() != 16) {
        return failure();
      }
      llvm::outs()<<"\n\n Pass \n\n";

      Location loc = op.getLoc();

      Type wideType = rewriter.getF32Type();
      llvm::outs()<<"helloA!\n";
      if(auto vecTy = resultType.dyn_cast<VectorType>()) {
        wideType = VectorType::get(vecTy.getShape(), wideType);
      }
      llvm::outs()<<"helloB!\n";

      Value lhsExt = rewriter.create<arith::ExtFOp>(loc, wideType, op.getLhs());
      Value rhsExt = rewriter.create<arith::ExtFOp>(loc, wideType, op.getRhs());
      Value maxExt = 
          rewriter.create<arith::MaxFOp>(loc, wideType, lhsExt, rhsExt);
      Value result = rewriter.create<arith::TruncFOp>(loc, resultType, maxExt);
      llvm::outs()<<"helloC!\n";
      // llvm::outs()<<"op found:"<<op<<"\n";
      // llvm::outs()<<"lhs:"<<lhsExt<<"\n";
      // llvm::outs()<<"rhs:"<<rhsExt<<"\n";
      // llvm::outs()<<"max:"<<maxExt<<"\n";
      // llvm::outs()<<"\nres:"<<result<<"\n";

      rewriter.replaceOp(op, result);
      return success();
    }
};

struct ExpandF16MaxFToF32Pass
    : public ExpandF16MaxFToF32Base<
          ExpandF16MaxFToF32Pass> {
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
