// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_DECOMPOSECONCATPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

static Value createTranspose(OpBuilder &builder, Value source,
                             SmallVector<int64_t> perm) {
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, source.getLoc(), source);
  applyPermutationToVector(mixedSizes, perm);
  Type elemType = cast<RankedTensorType>(source.getType()).getElementType();
  Value empty =
      builder.create<tensor::EmptyOp>(source.getLoc(), mixedSizes, elemType)
          .getResult();
  return builder
      .create<linalg::TransposeOp>(source.getLoc(), source, empty, perm)
      ->getResult(0);
}

// Transposes the concatenation dimension to happen along the outer most
// non-unit dim of the inputs. The idea is that outer dim concatentations
// can lower to `flow.tensor.update` and ideally disappear, in the worst case
// becoming a sequence of copies. The hope then is that the transposes on the
// inputs and output is then fusable with surrounding operations.
struct TransposeInnerConcatenation : public OpRewritePattern<tensor::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    // Get the outer most non-unit dim to transpose to.
    RankedTensorType concatType = concatOp.getResultType();
    ArrayRef<int64_t> concatShape = concatType.getShape();
    int64_t outerMostNonUnitDim = 0;
    while (outerMostNonUnitDim < concatOp.getRank()) {
      if (concatShape[outerMostNonUnitDim] != 1)
        break;
      outerMostNonUnitDim++;
    }

    // Nothing to do if the concat is already the outer most non-unit
    int64_t dim = concatOp.getDim();
    if (dim <= outerMostNonUnitDim) {
      return failure();
    }

    SmallVector<int64_t> permutation = computePermutationVector(
        concatOp.getRank(), {dim}, {outerMostNonUnitDim});
    SmallVector<Value> transposedInputs;
    for (auto input : concatOp.getInputs()) {
      transposedInputs.push_back(createTranspose(rewriter, input, permutation));
    }

    SmallVector<int64_t> newShape = applyPermutation(concatShape, permutation);
    auto newConcatType = RankedTensorType::get(
        newShape, concatOp.getResultType().getElementType());
    Value newConcat = rewriter.create<tensor::ConcatOp>(
        concatOp.getLoc(), newConcatType, /*dim=*/outerMostNonUnitDim,
        transposedInputs);
    auto invPerm = invertPermutationVector(permutation);
    Value transposedConcat = createTranspose(rewriter, newConcat, invPerm);
    rewriter.replaceOp(concatOp, transposedConcat);
    return success();
  }
};

/// Only decompose the concats that are not outer most here to see if they get
/// folded into other dispatches. Outerdim concats get lowered to
/// `flow.tensor.update` on conversion to Flow and then they get modified to be
/// in-place.
struct DecomposeNonOuterDimConcats : public OpRewritePattern<tensor::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    if (concatOp.getDim() == 0) {
      return rewriter.notifyMatchFailure(
          concatOp, "non-outer dim concats are not decomposed");
    }

    FailureOr<SmallVector<Value>> decomposed =
        concatOp.decomposeOperation(rewriter);
    if (failed(decomposed)) {
      return rewriter.notifyMatchFailure(concatOp,
                                         "failed to decompose concat op");
    }

    rewriter.replaceOp(concatOp, decomposed.value()[0]);
    return success();
  }
};

struct DecomposeConcatPass
    : public impl::DecomposeConcatPassBase<DecomposeConcatPass> {
  using impl::DecomposeConcatPassBase<
      DecomposeConcatPass>::DecomposeConcatPassBase;
  explicit DecomposeConcatPass(bool enableConcatTransposition) {
    this->enableConcatTransposition = enableConcatTransposition;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  DecomposeConcatPass(const DecomposeConcatPass &pass)
      : DecomposeConcatPass(pass.enableConcatTransposition) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.insert<DecomposeNonOuterDimConcats>(context);
    if (enableConcatTransposition) {
      patterns.insert<TransposeInnerConcatenation>(context, /*benefit=*/2);
    }
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass>
createDecomposeConcatPass(bool enableConcatTransposition) {
  return std::make_unique<DecomposeConcatPass>(enableConcatTransposition);
}

} // namespace mlir::iree_compiler::GlobalOptimization
