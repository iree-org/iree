// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler::IREE::VectorExt {

namespace {

static FailureOr<int64_t> getNonInnermostDimToUnroll(TransferGatherOp xferOp) {
  for (AffineExpr expr : xferOp.getPermutationMap().getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr) {
      continue;
    }

    int64_t dim = dimExpr.getPosition();
    // Do not unroll the innermost dimension.
    if (dim + 1 == xferOp.getIndices().size()) {
      continue;
    }

    return dim;
  }

  return failure();
}

void doit(AffineMap map, int64_t domainDim,
          function_ref<void(int64_t resultDim)> fun) {
  std::optional<int64_t> resultDim =
      map.getResultPosition(getAffineDimExpr(domainDim, map.getContext()));
  if (resultDim.has_value()) {
    fun(resultDim.value());
  }
}

SmallVector<int64_t> getIndexedIndices(VectorType vectorTy, AffineMap map,
                                       int64_t dim, int64_t iv) {
  SmallVector<int64_t> indices(vectorTy.getRank(), 0);
  doit(map, dim, [&](int64_t resultDim) { indices[resultDim] = iv; });
  return indices;
}

SmallVector<int64_t> getIndexedShape(VectorType vectorTy, AffineMap map,
                                     int64_t dim) {
  SmallVector<int64_t> shape(vectorTy.getShape());
  doit(map, dim, [&](int64_t resultDim) { shape[resultDim] = 1; });
  return shape;
}

AffineMap getIndexedMap(AffineMap map, int64_t dim) {
  doit(map, dim, [&](int64_t resultDim) { map = map.dropResult(resultDim); });
  return map;
}

Value loadIndexedOperand(Location loc, RewriterBase &rewriter, Value val,
                         AffineMap map, int64_t dim, int64_t iv) {
  VectorType vectorTy = cast<VectorType>(val.getType());

  SmallVector<int64_t> offsets(vectorTy.getRank(), 0);
  SmallVector<int64_t> sizes(vectorTy.getShape());
  SmallVector<int64_t> strides(offsets.size(), 1);

  doit(map, dim, [&](int64_t resultDim) {
    offsets[resultDim] = iv;
    sizes[resultDim] = 1;

    val = rewriter.create<vector::ExtractStridedSliceOp>(loc, val, offsets,
                                                         sizes, strides);

    sizes.erase(sizes.begin() + resultDim);
    val = rewriter.create<vector::ShapeCastOp>(
        loc, VectorType::get(sizes, getElementTypeOrSelf(val)), val);
  });

  return val;
}

Value storeIndexedOperand(Location loc, RewriterBase &rewriter, Value val,
                          Value dest, AffineMap map, int64_t dim, int64_t iv) {
  VectorType vectorTy = cast<VectorType>(dest.getType());

  SmallVector<int64_t> offsets(vectorTy.getRank(), 0);
  SmallVector<int64_t> sizes(vectorTy.getShape());
  SmallVector<int64_t> strides(offsets.size(), 1);

  doit(map, dim, [&](int64_t resultDim) {
    offsets[resultDim] = iv;
    sizes[resultDim] = 1;

    val = rewriter.create<vector::ShapeCastOp>(
        loc, VectorType::get(sizes, getElementTypeOrSelf(val)), val);
  });

  Value inserted = rewriter.create<vector::InsertStridedSliceOp>(
      loc, val, dest, offsets, strides);

  return inserted;
}

struct UnrollOuterGatherDimensions : public OpRewritePattern<TransferGatherOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TransferGatherOp xferOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<int64_t> maybeDim = getNonInnermostDimToUnroll(xferOp);
    if (failed(maybeDim)) {
      return failure();
    }
    int64_t unrollDim = maybeDim.value();

    int64_t resultDim =
        xferOp.getPermutationMap()
            .getResultPosition(rewriter.getAffineDimExpr(unrollDim))
            .value();
    int64_t dimSize = xferOp.getVectorType().getDimSize(resultDim);

    // Mask not handled yet.
    if (xferOp.getMask()) {
      return failure();
    }

    Location loc = xferOp.getLoc();
    Value dest = rewriter.create<vector::SplatOp>(loc, xferOp.getVectorType(),
                                                  xferOp.getPadding());
    for (int64_t i = 0; i < dimSize; ++i) {
      // Extract the indexed operands.
      SmallVector<Value> newIndexVecs;
      for (auto [val, map] :
           llvm::zip(xferOp.getIndexVecs(), xferOp.getIndexedMapsArray())) {
        Value newVal =
            loadIndexedOperand(loc, rewriter, val, map, unrollDim, i);
        newIndexVecs.push_back(newVal);
      }

      SmallVector<AffineMap> newIndexedMaps;
      for (AffineMap map : xferOp.getIndexedMapsArray()) {
        newIndexedMaps.push_back(getIndexedMap(map, unrollDim));
      }
      AffineMap newPermMap =
          getIndexedMap(xferOp.getPermutationMap(), unrollDim);

      VectorType resultType = xferOp.getVectorType();
      SmallVector<bool> inBounds = xferOp.getInBoundsValues();

      doit(xferOp.getPermutationMap(), unrollDim, [&](int64_t resultDim) {
        SmallVector<int64_t> sizes(resultType.getShape());
        sizes.erase(sizes.begin() + resultDim);
        resultType = VectorType::get(sizes, getElementTypeOrSelf(resultType));
        inBounds.erase(inBounds.begin() + resultDim);
      });

      // Create the new TransferGatherOp.
      auto gather = rewriter.create<TransferGatherOp>(
          loc, resultType, xferOp.getSource(), xferOp.getIndices(),
          newIndexVecs, xferOp.getIndexed(),
          rewriter.getAffineMapArrayAttr(newIndexedMaps), newPermMap,
          xferOp.getPadding(), Value(), rewriter.getBoolArrayAttr(inBounds));

      // Insert the result of the operand.
      dest = storeIndexedOperand(loc, rewriter, gather, dest,
                                 xferOp.getPermutationMap(), unrollDim, i);
    }

    rewriter.replaceOp(xferOp, dest);

    return success();
  }
};

} // namespace

void populateVectorTransferGatherLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<UnrollOuterGatherDimensions>(patterns.getContext());
}
}; // namespace mlir::iree_compiler::IREE::VectorExt
