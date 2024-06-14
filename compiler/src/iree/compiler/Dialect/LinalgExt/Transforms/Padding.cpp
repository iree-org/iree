// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

namespace {

OpFoldResult getPadding(RewriterBase &rewriter, Location loc,
                        OpFoldResult bound, int64_t padMultiple) {
  AffineExpr s0;
  bindSymbols(rewriter.getContext(), s0);
  AffineExpr padByExpr = (s0).ceilDiv(padMultiple) * padMultiple - s0;
  return affine::makeComposedFoldedAffineApply(rewriter, loc, padByExpr,
                                               {bound});
}

static Value
getPaddedValue(RewriterBase &rewriter, Location loc, Value padSource,
               ArrayRef<OpFoldResult> padding,
               std::optional<TypedAttr> padValueAttr = std::nullopt) {
  auto sourceType = cast<RankedTensorType>(padSource.getType());
  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  auto paddedShape =
      llvm::map_to_vector(llvm::zip_equal(sourceShape, padding), [](auto it) {
        std::optional<int64_t> padInt = getConstantIntValue(std::get<1>(it));
        if (ShapedType::isDynamic(std::get<0>(it)) || !padInt) {
          return ShapedType::kDynamic;
        }
        return std::get<0>(it) + padInt.value();
      });
  auto paddedResultType =
      RankedTensorType::get(paddedShape, sourceType.getElementType());
  auto zero = rewriter.getZeroAttr(sourceType.getElementType());
  Value paddingValue =
      rewriter.create<arith::ConstantOp>(loc, padValueAttr.value_or(zero));
  SmallVector<OpFoldResult> low(padding.size(), rewriter.getIndexAttr(0));
  Value paddedResult = rewriter.create<tensor::PadOp>(
      loc, paddedResultType, padSource, low, padding, paddingValue);
  return paddedResult;
}

struct PadAttentionPass : public PadAttentionBase<PadAttentionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

} // namespace

/// Pads iree_linalg_ext.attention.
LogicalResult padAttention(IREE::LinalgExt::AttentionOp attnOp,
                           SmallVectorImpl<Operation *> &ops,
                           RewriterBase &rewriter,
                           ArrayRef<int64_t> padToMultipleOf) {
  SmallVector<AffineMap> maps = attnOp.getIndexingMapsArray();
  FailureOr<IREE::LinalgExt::AttentionOpDetail> maybeOpInfo =
      IREE::LinalgExt::AttentionOpDetail::get(maps);
  if (failed(maybeOpInfo)) {
    // failed to infer attention dims
    return failure();
  }
  auto opInfo = maybeOpInfo.value();
  Location loc = attnOp.getLoc();
  rewriter.setInsertionPoint(attnOp);

  int64_t domainRank = maps[0].getNumDims();
  if (domainRank != 5) {
    // Currently only support base-case of attention dims.
    return failure();
  }
  if (padToMultipleOf.size() != domainRank) {
    // Expects pad_to_multiple to have same rank as dimensions of attention.
    return failure();
  }

  bool hasValidPadding = llvm::none_of(
      padToMultipleOf, [](int64_t padMultiple) { return padMultiple < 0; });
  if (!hasValidPadding) {
    // pad-multiple-of cannot be a negative value.
    return failure();
  }

  SmallVector<Range> bounds = attnOp.getIterationDomain(rewriter);

  int64_t batchIdx = opInfo.getBatchDims().back();
  int64_t mIdx = opInfo.getMDims().back();
  int64_t k1Idx = opInfo.getK1Dims().back();
  int64_t k2Idx = opInfo.getK2Dims().back();
  int64_t nIdx = opInfo.getNDims().back();

  // Padding in K2 dimension requires to fill those K2 dimensions as -Inf during
  // softmax(Q.KT), preemptively padding it with -Inf may cause NaNs during
  // matmul of Q.KT.
  if (padToMultipleOf[k2Idx] > 1) {
    return failure();
  }

  SmallVector<OpFoldResult> padValues(domainRank, rewriter.getIndexAttr(0));
  for (auto [idx, bound] : enumerate(bounds)) {
    if (padToMultipleOf[idx] != 0) {
      padValues[idx] =
          getPadding(rewriter, loc, bound.size, padToMultipleOf[idx]);
    }
  }

  Value paddedQuery = attnOp.getQuery();
  Value paddedKey = attnOp.getKey();
  Value paddedValue = attnOp.getValue();
  Value paddedAcc = attnOp.getOutput();
  Value scale = attnOp.getScale();

  OpFoldResult zero = rewriter.getIndexAttr(0);

  // Pad Q-tensor if any of its' dims needs padding.
  if (!isConstantIntValue(padValues[batchIdx], 0) ||
      !isConstantIntValue(padValues[mIdx], 0) ||
      !isConstantIntValue(padValues[k1Idx], 0)) {
    paddedQuery = getPaddedValue(
        rewriter, loc, paddedQuery,
        {padValues[batchIdx], padValues[mIdx], padValues[k1Idx]});
  }

  // Pad K-tensor if any non-K1 dims needs padding.
  if (!isConstantIntValue(padValues[batchIdx], 0) ||
      !isConstantIntValue(padValues[k1Idx], 0)) {
    paddedKey = getPaddedValue(rewriter, loc, paddedKey,
                               {padValues[batchIdx], zero, padValues[k1Idx]});
  }

  // Pad V-tensor if any of its' dims needs padding.
  if (!isConstantIntValue(padValues[batchIdx], 0) ||
      !isConstantIntValue(padValues[nIdx], 0)) {
    paddedValue = getPaddedValue(rewriter, loc, paddedValue,
                                 {padValues[batchIdx], zero, padValues[nIdx]});
  }

  // Pad Acc-tensor if any of its' dims needs padding.
  if (!isConstantIntValue(padValues[batchIdx], 0) ||
      !isConstantIntValue(padValues[mIdx], 0) ||
      !isConstantIntValue(padValues[nIdx], 0)) {
    if (llvm::dyn_cast_or_null<tensor::EmptyOp>(paddedAcc.getDefiningOp())) {
      SmallVector<OpFoldResult> paddedQueryShape =
          tensor::getMixedSizes(rewriter, loc, paddedQuery);
      SmallVector<OpFoldResult> paddedValueShape =
          tensor::getMixedSizes(rewriter, loc, paddedValue);
      SmallVector<OpFoldResult> paddedOutputShape = {
          paddedQueryShape[0], paddedQueryShape[1], paddedValueShape[2]};
      paddedAcc = rewriter.create<tensor::EmptyOp>(
          loc, paddedOutputShape, attnOp.getOutputType().getElementType());
    } else {
      paddedAcc = getPaddedValue(
          rewriter, loc, paddedAcc,
          {padValues[batchIdx], padValues[mIdx], padValues[nIdx]});
    }
  }

  // Generate padded attention op.
  auto paddedAttnOp = rewriter.create<IREE::LinalgExt::AttentionOp>(
      loc, paddedAcc.getType(),
      SmallVector<Value>{paddedQuery, paddedKey, paddedValue, scale},
      paddedAcc);

  ops.push_back(paddedAttnOp);

  // Extract subtensor result.
  IntegerAttr one = rewriter.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> offsets(3, zero);
  SmallVector<OpFoldResult> strides(3, one);
  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(rewriter, loc, attnOp.getOutput());
  Operation *extracted = rewriter.create<tensor::ExtractSliceOp>(
      loc, paddedAttnOp->getResults()[0], offsets, sizes, strides);
  ops.push_back(extracted);

  rewriter.replaceOp(attnOp, extracted);

  return success();
}

void PadAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  getOperation().walk([&](AttentionOp attnOp) {
    SmallVector<Operation *> ops;
    (void)padAttention(attnOp, ops, rewriter, padToMultipleOf);
  });
}

std::unique_ptr<Pass> createPadAttentionPass() {
  return std::make_unique<PadAttentionPass>();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
