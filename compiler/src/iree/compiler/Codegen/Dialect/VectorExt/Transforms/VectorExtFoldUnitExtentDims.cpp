// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::VectorExt {

#define GEN_PASS_DEF_VECTOREXTFOLDUNITEXTENTDIMSPASS
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h.inc"

namespace {

struct DropToLayoutUnitDims final
    : OpRewritePattern<IREE::VectorExt::ToLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::VectorExt::ToLayoutOp toLayoutOp,
                                PatternRewriter &rewriter) const override {
    if (!toLayoutOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(toLayoutOp,
                                         "requires tensor semanticS");
    }

    Location loc = toLayoutOp.getLoc();
    ShapedType inputTy = toLayoutOp.getType();
    ArrayRef<int64_t> shape = inputTy.getShape();

    // Find list of dims to drop and the target shape.
    SmallVector<bool> unitDims(shape.size(), false);
    SmallVector<int64_t> targetShape;
    bool hasUnitDims = false;
    for (auto [idx, size] : llvm::enumerate(shape)) {
      if (size == 1) {
        unitDims[idx] = true;
        hasUnitDims = true;
        continue;
      }
      targetShape.push_back(size);
    }

    if (!hasUnitDims) {
      return rewriter.notifyMatchFailure(toLayoutOp, "no unit dims present");
    }

    // Drop unit dims using extract_slice.
    FailureOr<Value> rankReducingExtract =
        tensor::ExtractSliceOp::rankReduceIfNeeded(
            rewriter, loc, toLayoutOp.getInput(), targetShape);
    assert(succeeded(rankReducingExtract) && "not a unit-extent collapse");

    // Find the rank reduced layout.
    VectorLayoutInterface newLayout = toLayoutOp.getLayout().project(unitDims);

    Value rankReducedValue = rankReducingExtract.value();
    auto newToLayoutOp = rewriter.create<IREE::VectorExt::ToLayoutOp>(
        loc, rankReducedValue.getType(), rankReducedValue, newLayout,
        toLayoutOp.getSharedMemoryConversion(), toLayoutOp.getMmaKindAttr());

    // Expand to preserve output shape using insert_slice.
    // Here, since the shape comes from the result of a to_layout op, it will
    // always be static.
    Value dest =
        rewriter.create<tensor::EmptyOp>(loc, shape, inputTy.getElementType());

    int64_t rank = inputTy.getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, dest);
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        toLayoutOp, newToLayoutOp.getResult(), dest, offsets, sizes, strides);

    return success();
  }
};

} // namespace

namespace {
struct VectorExtFoldUnitExtentDimsPass final
    : impl::VectorExtFoldUnitExtentDimsPassBase<
          VectorExtFoldUnitExtentDimsPass> {
  void runOnOperation() override {

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DropToLayoutUnitDims>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::VectorExt
