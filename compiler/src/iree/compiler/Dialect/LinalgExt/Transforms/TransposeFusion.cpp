// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
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
    OpOperand *operand = nullptr;
    linalg::LinalgOp producer;
    for (OpOperand *input : attentionOp.getDpsInputOperands()) {
      if (controlFn && !controlFn(input)) {
        continue;
      }

      auto maybeProducer = input->get().getDefiningOp<linalg::GenericOp>();
      if (maybeProducer && maybeProducer.isSingleYieldOp()) {
        producer = maybeProducer;
        operand = input;
        break;
      }
    }
    if (!operand) {
      return rewriter.notifyMatchFailure(attentionOp, "no operand found");
    }

    int64_t inputIndex = operand->getOperandNumber();

    auto producerMaps = producer.getIndexingMapsArray();
    AffineMap producerInputMap = producerMaps[0];
    AffineMap producerResultMap = producerMaps[1];
    if (!producerInputMap.isProjectedPermutation() ||
        !producerResultMap.isPermutation()) {
      return failure();
    }

    rewriter.modifyOpInPlace(attentionOp, [&]() {
      SmallVector<AffineMap> newIndexingMaps =
          attentionOp.getIndexingMapsArray();
      AffineMap consumerInputMap = attentionOp.getMatchingIndexingMap(operand);
      AffineMap composedMap =
          producerInputMap.compose(inversePermutation(producerResultMap));
      newIndexingMaps[inputIndex] = composedMap.compose(consumerInputMap);
      attentionOp.setIndexingMapsAttr(
          rewriter.getAffineMapArrayAttr(newIndexingMaps));
      attentionOp.setOperand(inputIndex, producer.getDpsInputs()[0]);
    });

    return success();
  }

private:
  linalg::ControlFusionFn controlFn;
};

// Bubbles transpose-V out of attention to expose the more performant
// attention-transposeV.
struct BubbleTransposeVFromAttentionOp
    : public OpRewritePattern<LinalgExt::AttentionOp> {
  BubbleTransposeVFromAttentionOp(MLIRContext *context,
                                  linalg::ControlFusionFn controlFn,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern<LinalgExt::AttentionOp>(context, benefit),
        controlFn(controlFn) {}

  LogicalResult matchAndRewrite(LinalgExt::AttentionOp attentionOp,
                                PatternRewriter &rewriter) const override {
    // Only checking for V because we are only bubbling transpose-V.
    OpOperand *valueOpOperand = &attentionOp.getValueMutable();
    if (controlFn && !controlFn(valueOpOperand)) {
      return rewriter.notifyMatchFailure(
          attentionOp, "Expected attentionOp and producer of V to be non-null "
                       "and outside dispatch.");
    }
    // Extract Attention indexing information.
    AffineMap qMap = attentionOp.getQueryMap();
    AffineMap kMap = attentionOp.getKeyMap();
    AffineMap vMap = attentionOp.getValueMap();
    AffineMap oMap = attentionOp.getOutputMap();
    FailureOr<AttentionOpDetail> maybeOpInfo =
        AttentionOpDetail::get(qMap, kMap, vMap, oMap);
    if (failed(maybeOpInfo)) {
      return failure();
    }

    // Only handle single dim for K2 and N for now.
    if (maybeOpInfo->getK2Dims().size() != 1 ||
        maybeOpInfo->getNDims().size() != 1) {
      return failure();
    }
    // Check that V has standard map/non transposed V.
    AffineExpr k2Dim =
        rewriter.getAffineDimExpr(maybeOpInfo->getK2Dims().back());
    AffineExpr nDim = rewriter.getAffineDimExpr(maybeOpInfo->getNDims().back());
    int64_t vRank = vMap.getNumResults();
    // TODO: This check is quite conservative, in the future we should simply
    //       do vMap.getResultPosition(k2Dim) > vMap.getResultPosition(nDim).
    if (vMap.getResult(vRank - 1) != nDim ||
        vMap.getResult(vRank - 2) != k2Dim) {
      return failure();
    }

    // Get dimension positions to prepare for transpose.
    std::optional<int64_t> maybeK2Pos = vMap.getResultPosition(k2Dim);
    std::optional<int64_t> maybeNPos = vMap.getResultPosition(nDim);
    assert(maybeK2Pos.has_value() && maybeNPos.has_value() &&
           "Expected K2 dim and N dim to be in V-map.");
    int64_t k2Pos = maybeK2Pos.value();
    int64_t nPos = maybeNPos.value();
    SmallVector<int64_t> perm = llvm::to_vector(llvm::seq<int64_t>(0, vRank));
    std::swap(perm[k2Pos], perm[nPos]);

    // Expose transposeOp for V.
    Location loc = attentionOp.getLoc();
    Value value = attentionOp.getValue();
    auto valueType = dyn_cast<ShapedType>(value.getType());
    auto valueElType = valueType.getElementType();
    SmallVector<OpFoldResult> transVShape =
        tensor::getMixedSizes(rewriter, loc, value);
    applyPermutationToVector(transVShape, perm);
    Value initTransV =
        rewriter.create<tensor::EmptyOp>(loc, transVShape, valueElType)
            .getResult();
    Value transposeV =
        rewriter.create<linalg::TransposeOp>(loc, value, initTransV, perm)
            ->getResult(0);

    // Generate transpose V map.
    SmallVector<AffineExpr> newExprs =
        applyPermutation(vMap.getResults(), perm);
    AffineMap transposedVMap =
        AffineMap::get(vMap.getNumDims(), vMap.getNumSymbols(), newExprs,
                       rewriter.getContext());

    // Modify attention to have transposed V inputs and mapping.
    int64_t valueIndex = valueOpOperand->getOperandNumber();
    rewriter.modifyOpInPlace(attentionOp, [&]() {
      SmallVector<AffineMap> newIndexingMaps =
          attentionOp.getIndexingMapsArray();
      newIndexingMaps[valueIndex] = transposedVMap;
      attentionOp.setIndexingMapsAttr(
          rewriter.getAffineMapArrayAttr(newIndexingMaps));
      attentionOp.setOperand(valueIndex, transposeV);
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

void populateBubbleTransposeFromLinalgExtOps(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFusionFn) {
  patterns.add<BubbleTransposeVFromAttentionOp>(patterns.getContext(),
                                                controlFusionFn);
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
