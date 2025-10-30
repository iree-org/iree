// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_SINKTRANSPOSETHROUGHPADPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc"

static Value createTransposeInit(OpBuilder &builder, Value source,
                                 ArrayRef<int64_t> perm) {
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, source.getLoc(), source);
  applyPermutationToVector(mixedSizes, perm);
  Type elemType = cast<RankedTensorType>(source.getType()).getElementType();
  Value empty =
      tensor::EmptyOp::create(builder, source.getLoc(), mixedSizes, elemType)
          .getResult();
  return empty;
}

static Value createTranspose(OpBuilder &builder, Value source,
                             ArrayRef<int64_t> perm) {
  Value empty = createTransposeInit(builder, source, perm);
  return linalg::TransposeOp::create(builder, source.getLoc(), source, empty,
                                     perm)
      ->getResult(0);
}

// Sinks a transpose through a tensor.pad
class SinkTransposeThroughPadOp : public OpRewritePattern<tensor::PadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(padOp)) {
      return failure();
    }
    Value source = padOp.getSource();
    auto transposeOp = source.getDefiningOp<linalg::TransposeOp>();
    if (!transposeOp) {
      return failure();
    }

    Block &block = padOp.getRegion().front();
    if (llvm::any_of(block.getArguments(), [](BlockArgument blockArg) {
          return blockArg.getNumUses();
        })) {
      return failure();
    }

    auto invPerm = invertPermutationVector(transposeOp.getPermutation());
    SmallVector<OpFoldResult> lowSizes = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> highSizes = padOp.getMixedHighPad();
    applyPermutationToVector(lowSizes, invPerm);
    applyPermutationToVector(highSizes, invPerm);

    RankedTensorType oldPaddedType = cast<RankedTensorType>(padOp.getType());
    RankedTensorType newPaddedType = oldPaddedType.clone(
        applyPermutation(oldPaddedType.getShape(), invPerm));
    auto newPadOp = tensor::PadOp::create(
        rewriter, padOp.getLoc(), newPaddedType, transposeOp.getInput(),
        lowSizes, highSizes, padOp.getNofold());
    rewriter.cloneRegionBefore(padOp.getRegion(), newPadOp.getRegion(),
                               newPadOp.getRegion().begin());
    Value newTransposeOp =
        createTranspose(rewriter, newPadOp, transposeOp.getPermutation());
    rewriter.replaceOp(padOp, newTransposeOp);
    return success();
  }
};

namespace {
struct SinkTransposeThroughPadPass
    : public impl::SinkTransposeThroughPadPassBase<
          SinkTransposeThroughPadPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<SinkTransposeThroughPadOp>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      getOperation().emitError(getPassName()) << " failed to converge.";
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::Preprocessing
