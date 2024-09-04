// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

static bool isaTranspose(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops())
    return false;

  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1)
    return false;

  SmallVector<AffineMap> mapRange = linalgOp.getIndexingMapsArray();
  if (mapRange.size() != 2 || !mapRange.front().isPermutation() ||
      !mapRange.back().isPermutation() || mapRange.front() == mapRange.back()) {
    return false;
  }
  return llvm::hasSingleElement(linalgOp.getBlock()->getOperations());
}

static SmallVector<int64_t> getPermutation(linalg::LinalgOp linalgOp) {
  assert(isaTranspose(linalgOp) && "linalgOp must be a transpose");
  SmallVector<AffineMap> mapRange = linalgOp.getIndexingMapsArray();
  AffineMap outMap = mapRange.back();
  AffineMap inMap = mapRange.front();

  // To get the permutation, look at each output index and find which
  // dimension in the input we're reading from for that index.
  return llvm::map_to_vector(outMap.getResults(), [&](AffineExpr expr) {
    return static_cast<int64_t>(inMap.getResultPosition(expr).value());
  });
}

namespace {

struct FuseTransposeWithAttentionOp final
    : public OpRewritePattern<LinalgExt::AttentionOp> {
  FuseTransposeWithAttentionOp(MLIRContext *context,
                               linalg::ControlFusionFn controlFn,
                               PatternBenefit benefit = 1)
      : OpRewritePattern<LinalgExt::AttentionOp>(context, benefit),
        controlFn(controlFn) {}

  LogicalResult matchAndRewrite(LinalgExt::AttentionOp attentionOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *transposeOperand = nullptr;
    linalg::LinalgOp transposeOp;
    for (OpOperand *input : attentionOp.getDpsInputOperands()) {
      if (controlFn && !controlFn(input)) {
        continue;
      }

      auto maybeTransposeOp = input->get().getDefiningOp<linalg::LinalgOp>();
      if (maybeTransposeOp && isaTranspose(maybeTransposeOp) &&
          maybeTransposeOp->hasOneUse()) {
        transposeOp = maybeTransposeOp;
        transposeOperand = input;
        break;
      }
    }
    if (!transposeOperand) {
      return rewriter.notifyMatchFailure(attentionOp, "no transpose operand");
    }

    int64_t inputIndex = transposeOperand->getOperandNumber();
    SmallVector<int64_t> perm = getPermutation(transposeOp);
    auto invPerm = invertPermutationVector(perm);

    rewriter.modifyOpInPlace(attentionOp, [&]() {
      SmallVector<AffineMap> newIndexingMaps =
          attentionOp.getIndexingMapsArray();
      AffineMap inputMap = attentionOp.getMatchingIndexingMap(transposeOperand);
      SmallVector<AffineExpr> newExprs =
          applyPermutation(inputMap.getResults(), invPerm);
      AffineMap transposedMap =
          AffineMap::get(inputMap.getNumDims(), inputMap.getNumSymbols(),
                         newExprs, rewriter.getContext());
      newIndexingMaps[inputIndex] = transposedMap;
      attentionOp.setIndexingMapsAttr(
          rewriter.getAffineMapArrayAttr(newIndexingMaps));
      attentionOp.setOperand(inputIndex, transposeOp.getDpsInputs()[0]);
    });

    return success();
  }

private:
  linalg::ControlFusionFn controlFn;
};
} // namespace

void populateFuseLinalgExtOpsWithTransposes(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFusionFn) {
  patterns.add<FuseTransposeWithAttentionOp>(patterns.getContext(),
                                             controlFusionFn);
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
