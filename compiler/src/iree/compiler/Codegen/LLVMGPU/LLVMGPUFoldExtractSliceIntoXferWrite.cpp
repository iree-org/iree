// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {
namespace {

class FoldExtractSliceIntoXferWrite final
    : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    if (extractSliceOp.getDroppedDims().any()) {
      return failure();
    }

    auto result = dyn_cast<OpResult>(extractSliceOp.getSource());
    if (!result) {
      return failure();
    }
    if (!result.hasOneUse()) {
      return failure();
    }

    auto xferOp =
        extractSliceOp.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!xferOp) {
      return failure();
    }
    if (!xferOp.getSource().getDefiningOp<tensor::EmptyOp>()) {
      return failure();
    }
    if (xferOp.getMask()) {
      return failure();
    }
    if (!xferOp.getInBounds()) {
      return failure();
    }

    Location loc = extractSliceOp.getLoc();
    SmallVector<OpFoldResult> mixedSizes = extractSliceOp.getMixedSizes();
    auto init = rewriter.create<tensor::EmptyOp>(
        loc, mixedSizes, extractSliceOp.getType().getElementType());

    SmallVector<bool> inBounds;
    inBounds.resize(mixedSizes.size());
    for (auto [idx, vecSize, destSize] :
         llvm::zip_equal(llvm::seq<int64_t>(0, inBounds.size()),
                         xferOp.getVectorType().getShape(), mixedSizes)) {
      auto maybeCst = getConstantIntValue(destSize);
      if (!maybeCst) {
        inBounds[idx] = false;
        continue;
      }
      if (*maybeCst == vecSize) {
        inBounds[idx] = false;
      } else {
        inBounds[idx] = true;
      }
    }

    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        extractSliceOp, xferOp.getVector(), init, xferOp.getIndices(),
        xferOp.getPermutationMap(), inBounds);

    return success();
  }
};

struct LLVMGPUFoldExtractSliceIntoXferWritePass
    : public LLVMGPUFoldExtractSliceIntoXferWriteBase<
          LLVMGPUFoldExtractSliceIntoXferWritePass> {
  void getDependentDialects(DialectRegistry &registry) const override {}
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto funcOp = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<FoldExtractSliceIntoXferWrite>(ctx);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUFoldExtractSliceIntoXferWritePass() {
  return std::make_unique<LLVMGPUFoldExtractSliceIntoXferWritePass>();
}

} // namespace mlir::iree_compiler
