// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Transforms/Patterns.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#define DEBUG_TYPE "iree-codegen-patterns"

namespace mlir::iree_compiler {

LogicalResult
FoldFillIntoPad::matchAndRewrite(tensor::PadOp padOp,
                                 PatternRewriter &rewriter) const {

  Operation *currentOp = padOp.getSource().getDefiningOp();
  auto maybeExtractSlice = dyn_cast_or_null<tensor::ExtractSliceOp>(currentOp);
  while (currentOp && maybeExtractSlice) {
    currentOp = maybeExtractSlice.getSource().getDefiningOp();
    maybeExtractSlice = dyn_cast_or_null<tensor::ExtractSliceOp>(currentOp);
  }
  auto fillOp = dyn_cast_or_null<linalg::FillOp>(currentOp);
  if (!fillOp) {
    return rewriter.notifyMatchFailure(
        padOp, "not coming from a linalg.fill op via tensor.extract_slice*");
  }

  Value padValue = padOp.getConstantPaddingValue();
  RankedTensorType resultType = padOp.getResultType();
  if (!padValue || getAsOpFoldResult(padValue) !=
                       getAsOpFoldResult(fillOp.getDpsInputOperand(0)->get())) {
    return rewriter.notifyMatchFailure(
        padOp, "not a constant value matching the fill value");
  }

  Location loc = padOp.getLoc();
  auto emptyOp = rewriter.create<tensor::EmptyOp>(
      loc, tensor::getMixedSizes(rewriter, loc, padOp),
      resultType.getElementType());
  rewriter.replaceOpWithNewOp<linalg::FillOp>(padOp, padValue,
                                              emptyOp.getResult());

  return success();
}

LogicalResult
EmptyTensorLoweringPattern::matchAndRewrite(tensor::EmptyOp op,
                                            PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<bufferization::AllocTensorOp>(
      op, op.getType(), op.getDynamicSizes());
  return success();
}

} // namespace mlir::iree_compiler
