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

  vector::TransferReadOp
  createReadOp(PatternRewriter &rewriter,
               IREE::VectorExt::ToLayoutOp toLayoutOp) const {
    Location loc = toLayoutOp.getLoc();
    ShapedType inputTy = toLayoutOp.getType();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto identityMap = rewriter.getMultiDimIdentityMap(inputTy.getRank());
    SmallVector<int64_t> readShape =
        toLayoutOp.getLayout().getUndistributedShape();
    Value mask = nullptr;
    if (!toLayoutOp.getType().hasStaticShape()) {
      SmallVector<OpFoldResult> mixedSourceDims =
          tensor::getMixedSizes(rewriter, loc, toLayoutOp.getInput());
      auto maskType = VectorType::get(readShape, rewriter.getI1Type());
      mask =
          rewriter.create<vector::CreateMaskOp>(loc, maskType, mixedSourceDims);
    }
    VectorType vectorType =
        VectorType::get(readShape, inputTy.getElementType());
    auto inBounds = rewriter.getBoolArrayAttr(
        SmallVector<bool>(vectorType.getRank(), true));
    auto padValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(inputTy.getElementType()));
    auto read = rewriter.create<vector::TransferReadOp>(
        loc,
        /*type=*/vectorType,
        /*source=*/toLayoutOp.getInput(),
        /*indices=*/ValueRange{SmallVector<Value>(readShape.size(), zero)},
        /*permutation_map=*/identityMap,
        /*padding=*/padValue,
        /*mask=*/mask,
        /*in_bounds=*/inBounds);
    return read;
  }

  vector::TransferWriteOp
  createWriteOp(PatternRewriter &rewriter,
                IREE::VectorExt::ToLayoutOp tensorLayoutOp,
                Value vectorLayoutOp, Value mask) const {
    Location loc = tensorLayoutOp.getLoc();
    ShapedType tensorTy = tensorLayoutOp.getType();
    auto resType =
        RankedTensorType::get(tensorTy.getShape(), tensorTy.getElementType());
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    int64_t rank = tensorTy.getShape().size();
    auto inBounds = rewriter.getBoolArrayAttr(SmallVector<bool>(rank, true));
    auto identityMap = rewriter.getMultiDimIdentityMap(tensorTy.getRank());
    auto empty = rewriter.create<tensor::EmptyOp>(
        loc, tensor::getMixedSizes(rewriter, loc, tensorLayoutOp.getInput()),
        tensorTy.getElementType());
    return rewriter.create<vector::TransferWriteOp>(
        loc,
        /*result=*/resType,
        /*vector=*/vectorLayoutOp,
        /*source=*/empty,
        /*indices=*/ValueRange{SmallVector<Value>(rank, zero)},
        /*permutation_map=*/identityMap,
        /*mask=*/mask,
        /*inBounds=*/inBounds);
  }

  LogicalResult matchAndRewrite(IREE::VectorExt::ToLayoutOp toLayoutOp,
                                PatternRewriter &rewriter) const override {
    if (!toLayoutOp.hasTensorSemantics()) {
      return failure();
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(toLayoutOp);
    Location loc = toLayoutOp.getLoc();
    vector::TransferReadOp readOp = createReadOp(rewriter, toLayoutOp);
    // Create the toLayout operation but with vector types instead.
    auto newLayoutOp = rewriter.create<IREE::VectorExt::ToLayoutOp>(
        loc, readOp, toLayoutOp.getLayout(), toLayoutOp.getMmaKindAttr(),
        toLayoutOp.getSharedMemoryConversion());
    // Create the write back to a tensor.
    vector::TransferWriteOp writeOp =
        createWriteOp(rewriter, toLayoutOp, newLayoutOp, readOp.getMask());
    rewriter.replaceOp(toLayoutOp, writeOp);
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
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::VectorExt
