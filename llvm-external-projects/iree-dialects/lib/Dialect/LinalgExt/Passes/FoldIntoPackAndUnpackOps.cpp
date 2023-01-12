// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE::LinalgExt;

//===---------------------------------------------------------------------===//
// Patterns to fold operationsinto pack/unpack ops.
//===---------------------------------------------------------------------===//

namespace {

static bool areAllConstantIntValue(ArrayRef<OpFoldResult> ofrs, int64_t value) {
  return llvm::all_of(
      ofrs, [&](OpFoldResult ofr) { return isConstantIntValue(ofr, value); });
}

/// Fold a `pad` -> `pack` into `pack` if they have the same padding values and
/// the pad op has zero low paddings.
struct FoldPadWithPackOp : public OpRewritePattern<PackOp> {
  using OpRewritePattern<PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto padOp = packOp.getInput().getDefiningOp<tensor::PadOp>();

    if (!padOp || padOp.getNofold() || !padOp.hasZeroLowPad())
      return failure();

    Value constantPaddingValue = padOp.getConstantPaddingValue();
    if (!constantPaddingValue)
      return failure();

    if (auto paddingValue = packOp.getPaddingValue())
      if (!isEqualConstantIntOrValue(paddingValue, constantPaddingValue))
        return failure();

    rewriter.replaceOpWithNewOp<PackOp>(
        packOp, padOp.getSource(), packOp.getOutput(), packOp.getInnerDimsPos(),
        packOp.getMixedTiles(), constantPaddingValue,
        packOp.getOuterDimsPerm());
    return success();
  }
};

/// Fold a `unpack` -> `extract_slice` into the `unpack` since it already
/// has extract_slice semantics.
struct FoldUnpackWithExtractSliceOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto unpackOp = sliceOp.getSource().getDefiningOp<UnPackOp>();
    if (!unpackOp)
      return failure();

    // Check all offsets are zeros, and all strides are 1.
    if (!areAllConstantIntValue(sliceOp.getMixedOffsets(), 0) ||
        !areAllConstantIntValue(sliceOp.getMixedStrides(), 1)) {
      return rewriter.notifyMatchFailure(
          sliceOp, "expects offsets to be 0s and strides to be 1s");
    }

    // Create a new empty output tensor.
    Type elementType = unpackOp.getOutput()
                           .getType()
                           .cast<RankedTensorType>()
                           .getElementType();
    Value output = rewriter.create<tensor::EmptyOp>(
        sliceOp.getLoc(), sliceOp.getMixedSizes(), elementType);
    rewriter.replaceOpWithNewOp<UnPackOp>(
        sliceOp, unpackOp.getInput(), output, unpackOp.getInnerDimsPos(),
        unpackOp.getMixedTiles(), unpackOp.getOuterDimsPerm());
    return success();
  }
};
} // namespace

//===---------------------------------------------------------------------===//
// Pass to fold operations into pack and unpack operations.
//===---------------------------------------------------------------------===//

namespace {
struct FoldIntoPackAndUnpackOpsPass
    : public FoldIntoPackAndUnpackOpsBase<FoldIntoPackAndUnpackOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    return;
  }

  void runOnOperation() override;
};
} // namespace

void FoldIntoPackAndUnpackOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateFoldIntoPackAndUnpackOpsPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

void populateFoldIntoPackAndUnpackOpsPatterns(RewritePatternSet &patterns) {
  patterns.insert<FoldUnpackWithExtractSliceOp, FoldPadWithPackOp>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> createFoldIntoPackAndUnpackOps() {
  return std::make_unique<FoldIntoPackAndUnpackOpsPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
