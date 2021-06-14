// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/PassDetail.h"
#include "iree/compiler/Conversion/Passes.h"
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
                                                 op.a(), op.b());
    auto fmaResult = rewriter.create<LLVM::FAddOp>(
        loc, mulPart.getResult().getType(), mulPart.getResult(), op.c());
    rewriter.replaceOp(op, fmaResult.getResult());
    return success();
  }
};
}  // namespace

namespace {
struct LinalgToLLVMUnfuseFMAOpsPass
    : LinalgToLLVMUnfuseFMAOpsBase<LinalgToLLVMUnfuseFMAOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

void populateUnfusedFMAOpsPassPatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns) {
  patterns.insert<UnfusedFMAOpsPassConversion>(context);
}

void LinalgToLLVMUnfuseFMAOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  auto context = funcOp.getContext();
  OwningRewritePatternList patterns(&getContext());
  populateUnfusedFMAOpsPassPatterns(context, patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<OperationPass<FuncOp>> createLinalgToLLVMUnfuseFMAOpsPass() {
  return std::make_unique<LinalgToLLVMUnfuseFMAOpsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
