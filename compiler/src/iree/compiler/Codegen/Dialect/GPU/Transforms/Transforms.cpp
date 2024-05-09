// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-codegen-gpu-transforms"

namespace mlir::iree_compiler::IREE::GPU {

LogicalResult vectorizeStaticMultiMmaOp(RewriterBase &rewriter,
                                        IREE::GPU::MultiMmaOp mmaOp) {
  if (!mmaOp.hasTensorSemantics()) {
    return failure();
  }
  if (!mmaOp.getLhsType().hasStaticShape() ||
      !mmaOp.getRhsType().hasStaticShape() ||
      !mmaOp.getAccType().hasStaticShape()) {
    return rewriter.notifyMatchFailure(mmaOp,
                                       "non-static shape for vectorization");
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(mmaOp);

  Location loc = mmaOp.getLoc();

  // Construct the (never used) zero padding value for each operand.
  auto lhsPadValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(mmaOp.getLhsType().getElementType()));
  auto rhsPadValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(mmaOp.getRhsType().getElementType()));
  Type resultElementType = mmaOp.getResultType().getElementType();
  auto accPadValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(resultElementType));

  auto lhs = vector::createReadOrMaskedRead(
      rewriter, loc, mmaOp.getLhs(), mmaOp.getLhsType().getShape(), lhsPadValue,
      /*useInBoundsInsteadOfMasking=*/true);
  auto rhs = vector::createReadOrMaskedRead(
      rewriter, loc, mmaOp.getRhs(), mmaOp.getRhsType().getShape(), rhsPadValue,
      /*useInBoundsInsteadOfMasking=*/true);
  auto acc = vector::createReadOrMaskedRead(
      rewriter, loc, mmaOp.getAcc(), mmaOp.getAccType().getShape(), accPadValue,
      /*useInBoundsInsteadOfMasking=*/true);
  auto newMmaOp = rewriter.create<IREE::GPU::MultiMmaOp>(
      loc, lhs, rhs, acc, mmaOp.getIndexingMaps(), mmaOp.getIteratorTypes(),
      mmaOp.getKind());

  // Create the write back to a tensor.
  int64_t rank = mmaOp.getResultType().getRank();
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
      mmaOp,
      /*vector=*/newMmaOp,
      /*source=*/mmaOp.getAcc(),
      /*indices=*/SmallVector<Value>(rank, zero),
      /*inBounds=*/SmallVector<bool>(rank, true));
  return success();
}

namespace {
struct VectorizeStaticMultiMmaOpPattern
    : public OpRewritePattern<IREE::GPU::MultiMmaOp> {
  using OpRewritePattern<IREE::GPU::MultiMmaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::GPU::MultiMmaOp mmaOp,
                                PatternRewriter &rewriter) const override {
    return vectorizeStaticMultiMmaOp(rewriter, mmaOp);
  }
};
} // namespace

void populateIREEGPUVectorizationPatterns(RewritePatternSet &patterns) {
  patterns.add<VectorizeStaticMultiMmaOpPattern>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::GPU
