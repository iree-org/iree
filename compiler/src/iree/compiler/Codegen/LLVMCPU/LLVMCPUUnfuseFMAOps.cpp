// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUUNFUSEFMAOPSPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

// Rewrites llvm.intr.fma as its un-fuse version.
// TODO(ataei): Upstream this pattern if needed ?
class UnfusedFMAOpsPassConversion : public OpRewritePattern<LLVM::FMAOp> {
public:
  using OpRewritePattern::OpRewritePattern;

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
} // namespace

namespace {
struct LLVMCPUUnfuseFMAOpsPass
    : impl::LLVMCPUUnfuseFMAOpsPassBase<LLVMCPUUnfuseFMAOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() override;
};
} // namespace

void populateUnfusedFMAOpsPassPatterns(MLIRContext *context,
                                       RewritePatternSet &patterns) {
  patterns.insert<UnfusedFMAOpsPassConversion>(context);
}

void LLVMCPUUnfuseFMAOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  auto context = funcOp.getContext();
  RewritePatternSet patterns(&getContext());
  populateUnfusedFMAOpsPassPatterns(context, patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler
