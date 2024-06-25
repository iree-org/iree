// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Unit.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-preprocessing-transpose-extract-concat-pass"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_TRANSPOSEEXTRACTCONCATPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

static Value createTransposeOp(OpBuilder &builder, Value source,
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

// Constructs a transpose of the given tensor and permutation.
static Value createPermutedInit(OpBuilder &builder, Value oldInit,
                                ArrayRef<int64_t> permutation) {
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, oldInit.getLoc(), oldInit);
  applyPermutationToVector(mixedSizes, permutation);
  Type elemType = cast<RankedTensorType>(oldInit.getType()).getElementType();
  Value empty =
      builder.create<tensor::EmptyOp>(oldInit.getLoc(), mixedSizes, elemType)
          .getResult();
  return empty;
}

// Get the operand the operand representing the next op to follow.
// This checks to make sure the current op only has 1 tensor operand and
// possibly other single element tensor operands
static FailureOr<Value> getIntermediateOperand(RewriterBase &rewriter,
                                               linalg::GenericOp genericOp) {
  Value *maybeOperand = nullptr;
  SmallVector<Value> operands = genericOp.getDpsInputs();
  for (Value &currOperand : operands) {
    auto operandType = dyn_cast<RankedTensorType>(currOperand.getType());
    if (!operandType)
      return failure();

    // Single element tensor used as constant is ok
    if (operandType.getRank() == 0)
      continue;

    if (maybeOperand)
      return failure();
    maybeOperand = &currOperand;
  }
  if (!maybeOperand)
    return failure();
  return *maybeOperand;
}

static FailureOr<tensor::ExtractSliceOp>
findConcatToExtratChain(RewriterBase &rewriter, Operation *op) {
  // Iterate up until finding a extract_slice op
  while (!isa<tensor::ExtractSliceOp>(op)) {
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp || !linalg::isElementwise(genericOp) ||
        genericOp.getNumResults() != 1 || !genericOp.getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(op,
                                         "op is not a elementwise generic op");

    auto maybeOperand = getIntermediateOperand(rewriter, genericOp);
    if (failed(maybeOperand))
      return rewriter.notifyMatchFailure(genericOp,
                                         "did not match generic op's operand");

    Value operand = *maybeOperand;
    if (!operand.getDefiningOp()) {
      return rewriter.notifyMatchFailure(
          genericOp, "generic op operand has no defining op");
    }
    op = operand.getDefiningOp();
  }
  return cast<tensor::ExtractSliceOp>(op);
}

static void transposeExtractChain(PatternRewriter &rewriter,
                                  SmallVector<int64_t> &perm, Value &transpose,
                                  tensor::ExtractSliceOp &sliceOp) {
  // Transpose the extract
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(sliceOp);
  auto resultType = RankedTensorType::get(
      applyPermutation(sliceOp.getResultType().getShape(), perm),
      sliceOp.getResultType().getElementType());
  auto newSliceOp = rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
      sliceOp.getOperation(), resultType, transpose,
      applyPermutation(sliceOp.getMixedOffsets(), perm),
      applyPermutation(sliceOp.getMixedSizes(), perm),
      applyPermutation(sliceOp.getMixedStrides(), perm));

  // Transpose from extract until the concat
  Operation *currOp = *newSliceOp.getResult().getUsers().begin();
  while (!isa<tensor::ConcatOp>(currOp)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(currOp);

    linalg::GenericOp genericOp = llvm::cast<linalg::GenericOp>(currOp);

    Value newInit = createPermutedInit(
        rewriter, genericOp.getDpsInitOperand(0)->get(), perm);
    SmallVector<Value> newOperands(genericOp.getOperands());
    newOperands.back() = newInit;
    auto newGenericOp =
        mlir::clone(rewriter, genericOp, newInit.getType(), newOperands);
    rewriter.replaceOp(genericOp, newGenericOp);
    currOp = *newGenericOp->getUsers().begin();
  }
}

class TransposeExtractConcat : public OpRewritePattern<tensor::ConcatOp> {
public:
  using OpRewritePattern<tensor::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    if (concatOp.getRank() - 1 != concatOp.getDim()) {
      return rewriter.notifyMatchFailure(concatOp,
                                         "concat dim is not last dim");
    }

    // Collect all extract slices
    SmallVector<tensor::ExtractSliceOp> sliceOps;
    for (Value operand : concatOp.getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (!definingOp) {
        return rewriter.notifyMatchFailure(
            concatOp, "concat op's operand has no defining op");
      }

      FailureOr<tensor::ExtractSliceOp> maybeSlice =
          findConcatToExtratChain(rewriter, operand.getDefiningOp());
      if (failed(maybeSlice))
        return failure();
      sliceOps.push_back(*maybeSlice);
    }

    // Ensure extract ops have the same source
    Value source = sliceOps[0].getSource();
    for (auto sliceOp : sliceOps) {
      if (sliceOp.getSource() != source)
        return rewriter.notifyMatchFailure(
            concatOp, "not all slice ops share the same source");
    }

    int64_t dim = concatOp.getDim();
    RankedTensorType outerShape = concatOp.getResultType();
    SmallVector<int64_t> perm =
        computePermutationVector(outerShape.getRank(), {dim}, {0});

    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfterValue(source);
      auto transpose = createTransposeOp(rewriter, source, perm);

      for (auto sliceOp : sliceOps) {
        transposeExtractChain(rewriter, perm, transpose, sliceOp);
      }
    }

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(concatOp);
    SmallVector<int64_t> newConcatShape =
        applyPermutation(concatOp.getResultType().getShape(), perm);
    auto newConcatType = RankedTensorType::get(
        newConcatShape, concatOp.getResultType().getElementType());
    auto newConcatOp = rewriter.create<tensor::ConcatOp>(
        concatOp.getLoc(), newConcatType, 0, concatOp.getInputs());

    auto invPerm = invertPermutationVector(perm);
    auto transposed =
        createTransposeOp(rewriter, newConcatOp.getResult(), invPerm);
    rewriter.replaceOp(concatOp, transposed);

    return success();
  }
};

namespace {
struct TransposeExtractConcatPass
    : public impl::TransposeExtractConcatPassBase<TransposeExtractConcatPass> {
  using impl::TransposeExtractConcatPassBase<
      TransposeExtractConcatPass>::TransposeExtractConcatPassBase;

  void runOnOperation() override {
    Operation *funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<TransposeExtractConcat>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();
  }

private:
};
} // namespace

} // namespace mlir::iree_compiler::Preprocessing
