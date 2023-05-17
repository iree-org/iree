// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/LLVMCPUPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Rewrites llvm.intr.fma as its un-fuse version.
// TODO(ataei): Upstream this pattern if needed ?
class UnfusedFMAOpsPassConversion : public OpRewritePattern<LLVM::FMAOp> {
 public:
  using OpRewritePattern<LLVM::FMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::FMAOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mulPart = rewriter.create<LLVM::FMulOp>(loc, op.getResult().getType(),
                                                 op.getA(), op.getB());
    auto fmaResult = rewriter.create<LLVM::FAddOp>(
        loc, mulPart.getResult().getType(), mulPart.getResult(), op.getC());
    rewriter.replaceOp(op, fmaResult.getResult());
    return success();
  }
};
}  // namespace

namespace {
struct LLVMCPUUnfuseFMAOpsPass
    : LLVMCPUUnfuseFMAOpsBase<LLVMCPUUnfuseFMAOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

void populateUnfusedFMAOpsPassPatterns(MLIRContext *context,
                                       RewritePatternSet &patterns) {
  patterns.insert<UnfusedFMAOpsPassConversion>(context);
}

void LLVMCPUUnfuseFMAOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  auto context = funcOp.getContext();
  RewritePatternSet patterns(&getContext());
  populateUnfusedFMAOpsPassPatterns(context, patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUUnfuseFMAOpsPass() {
  return std::make_unique<LLVMCPUUnfuseFMAOpsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
