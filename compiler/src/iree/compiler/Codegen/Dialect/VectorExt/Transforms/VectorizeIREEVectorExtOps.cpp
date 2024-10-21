// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::VectorExt {

#define GEN_PASS_DEF_VECTORIZEIREEVECTOREXTOPSPASS
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h.inc"

namespace {

struct VectorizeToLayoutOpPattern final
    : OpRewritePattern<IREE::VectorExt::ToLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::VectorExt::ToLayoutOp toLayoutOp,
                                PatternRewriter &rewriter) const override {
    if (!toLayoutOp.hasTensorSemantics()) {
      return failure();
    }
    if (!toLayoutOp.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(toLayoutOp,
                                         "non-static shape for vectorization");
    }

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(toLayoutOp);

    Location loc = toLayoutOp.getLoc();
    ShapedType inputTy = toLayoutOp.getType();

    // Construct the (never used) zero padding value for input.
    auto padValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(inputTy.getElementType()));

    auto newInput = vector::createReadOrMaskedRead(
        rewriter, loc, toLayoutOp.getInput(), inputTy.getShape(), padValue,
        /*useInBoundsInsteadOfMasking=*/true);

    // Create the toLayout operation but with vector types instead.
    auto newLayoutOp = rewriter.create<IREE::VectorExt::ToLayoutOp>(
        loc, newInput, toLayoutOp.getLayout(), toLayoutOp.getMmaKindAttr(),
        toLayoutOp.getSharedMemoryConversion());

    // Create the write back to a tensor.
    int64_t rank = inputTy.getRank();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto empty = rewriter.create<tensor::EmptyOp>(loc, inputTy, ValueRange());
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        toLayoutOp,
        /*vector=*/newLayoutOp,
        /*source=*/empty,
        /*indices=*/SmallVector<Value>(rank, zero),
        /*inBounds=*/SmallVector<bool>(rank, true));
    return success();
  }
};

} // namespace

namespace {
struct VectorizeIREEVectorExtOpsPass final
    : impl::VectorizeIREEVectorExtOpsPassBase<VectorizeIREEVectorExtOpsPass> {
  void runOnOperation() override {

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<VectorizeToLayoutOpPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::VectorExt
