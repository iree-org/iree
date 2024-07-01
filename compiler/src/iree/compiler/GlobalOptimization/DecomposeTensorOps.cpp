// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {
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

static int64_t findOuterMostNonUnitDim(ArrayRef<int64_t> &shape) {
  int64_t outerMostNonUnitDim = 0;
  while (outerMostNonUnitDim < shape.size()) {
    if (shape[outerMostNonUnitDim] != 1)
      break;
    outerMostNonUnitDim++;
  }
  return outerMostNonUnitDim;
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
    int64_t outerMostNonUnitDim = findOuterMostNonUnitDim(concatShape);

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

// Decompose `tensor.extract_slice` into a `linalg.transpose` +
// `tensor.extract_slice` + (potentially a second transpose). The slice op must
// only extract a subtensor along the innermost dimension. The goal is make sure
// the extraction happens along the innermost dimension.
struct TransposeInnerExtractSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {

    if (sliceOp.getSourceType().getNumDynamicDims() ||
        sliceOp.getResultType().getNumDynamicDims()) {
      return failure();
    }

    ArrayRef<int64_t> sourceShape = sliceOp.getSourceType().getShape();
    ArrayRef<int64_t> resultShape = sliceOp.getResultType().getShape();
    SmallVector<int64_t> sizes(sliceOp.getStaticSizes());

    int64_t numReducedDims = 0;
    for (auto [size, length] : llvm::zip(sizes, sourceShape)) {
      numReducedDims += (size != length);
      if (numReducedDims > 1) {
        return rewriter.notifyMatchFailure(sliceOp,
                                           "slice reduces dim size on >1 dim");
      }
    }

    // There should be exactly 1 dim reduction, and it should be
    // the innermost
    if (numReducedDims == 0 || sizes.back() >= sourceShape.back())
      return failure();

    SmallVector<int64_t> perm = computePermutationVector(
        sourceShape.size(), {static_cast<long>(sourceShape.size() - 1)}, {0});
    SmallVector<int64_t> transposedDims = applyPermutation(sourceShape, perm);
    Value transpose = createTranspose(rewriter, sliceOp.getSource(), perm);
    bool isRankReduction = sourceShape.size() > resultShape.size();

    // When `extract_slice` causes rank reduction, the ordering of dim lengths
    // stays the same (unit dim gets omitted).
    SmallVector<int64_t> newResultShape(resultShape);
    if (!isRankReduction)
      applyPermutationToVector(newResultShape, perm);

    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp.getLoc(),
        RankedTensorType::get(newResultShape,
                              sliceOp.getResultType().getElementType()),
        transpose, applyPermutation(sliceOp.getMixedOffsets(), perm),
        applyPermutation(sliceOp.getMixedSizes(), perm),
        sliceOp.getMixedStrides());

    // When it's a rank reducing slice, there is no need to undo the transpose
    // since unit dim get erased and new/old shapes are automatically equal
    if (isRankReduction) {
      rewriter.replaceOp(sliceOp, newSliceOp);
      return success();
    }

    auto invertedPerm = invertPermutationVector(perm);
    Value transposedExtract =
        createTranspose(rewriter, newSliceOp, invertedPerm);
    rewriter.replaceOp(sliceOp, transposedExtract);
    return success();
  }
};

struct DecomposeTensorOpsPass
    : public DecomposeTensorOpsBase<DecomposeTensorOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  DecomposeTensorOpsPass(bool enableTransposition) {
    this->enableTransposition = enableTransposition;
  }
  DecomposeTensorOpsPass(const DecomposeTensorOpsPass &pass)
      : DecomposeTensorOpsPass(pass.enableTransposition) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    if (enableTransposition) {
      patterns.insert<TransposeInnerConcatenation, TransposeInnerExtractSlice>(
          context, /*benefit=*/2);
    }
    tensor::populateDecomposeTensorConcatPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createDecomposeTensorOpsPass(bool enableTransposition) {
  return std::make_unique<DecomposeTensorOpsPass>(enableTransposition);
}

} // namespace mlir::iree_compiler::GlobalOptimization
