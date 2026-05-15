// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

namespace {

/// Remove dim 0 from an AffineMap by:
/// 1. Replacing AffineDimExpr(0) with AffineConstantExpr(0)
/// 2. Renumbering AffineDimExpr(k) where k > 0 to AffineDimExpr(k-1)
/// 3. Reducing numDims by 1
static AffineMap removeDim0FromMap(AffineMap map) {
  MLIRContext *ctx = map.getContext();
  SmallVector<AffineExpr> newResults;
  for (AffineExpr expr : map.getResults()) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      unsigned pos = dimExpr.getPosition();
      if (pos == 0) {
        newResults.push_back(getAffineConstantExpr(0, ctx));
      } else {
        newResults.push_back(getAffineDimExpr(pos - 1, ctx));
      }
    } else {
      newResults.push_back(expr);
    }
  }
  return AffineMap::get(map.getNumDims() - 1, map.getNumSymbols(), newResults,
                        ctx);
}

/// Remove dim 0 references from an index vec map. Returns the new map with
/// results that referenced dim 0 dropped, and the axis positions in the index
/// vec that need to be sliced.
static AffineMap removeDim0FromIndexVecMap(AffineMap map,
                                           SmallVectorImpl<int64_t> &axes) {
  MLIRContext *ctx = map.getContext();
  SmallVector<AffineExpr> newResults;
  for (auto [resultIdx, expr] : llvm::enumerate(map.getResults())) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      unsigned pos = dimExpr.getPosition();
      if (pos == 0) {
        axes.push_back(resultIdx);
        continue;
      }
      newResults.push_back(getAffineDimExpr(pos - 1, ctx));
    } else {
      newResults.push_back(expr);
    }
  }
  return AffineMap::get(map.getNumDims() - 1, map.getNumSymbols(), newResults,
                        ctx);
}

/// Extract a slice from a vector at position `idx` along the given `axis`.
/// For a vector<4x8xindex>, extracting axis=0, idx=2 gives vector<8xindex>.
static Value extractVecSlice(OpBuilder &b, Location loc, Value vec,
                             int64_t axis, int64_t idx) {
  auto vecType = cast<VectorType>(vec.getType());
  int64_t rank = vecType.getRank();

  if (axis == 0) {
    // Extracting from rank-1 along axis 0 gives a scalar.
    return vector::ExtractOp::create(b, loc, vec, int64_t{idx});
  }

  // General case: use extract_strided_slice.
  SmallVector<int64_t> offsets(rank, 0);
  SmallVector<int64_t> sizes(vecType.getShape());
  SmallVector<int64_t> strides(rank, 1);
  offsets[axis] = idx;
  sizes[axis] = 1;
  Value slice = vector::ExtractStridedSliceOp::create(b, loc, vec, offsets,
                                                      sizes, strides);
  // Drop the unit dim.
  SmallVector<int64_t> newShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != axis) {
      newShape.push_back(vecType.getShape()[i]);
    }
  }
  auto newType = VectorType::get(newShape, vecType.getElementType());
  return vector::ShapeCastOp::create(b, loc, newType, slice);
}

//===----------------------------------------------------------------------===//
// Shared unroll helpers
//===----------------------------------------------------------------------===//

/// Compute dim-0-removed indexing maps for unrolling. Populates the new base
/// map, per-index-vec maps and axes, mask map and axes, base dims using dim 0,
/// and the combined new indexing maps array.
static void
computeUnrollDim0Maps(ArrayRef<AffineMap> indexingMaps, int64_t numIndexVecs,
                      bool hasMask, AffineMap baseMap,
                      SmallVectorImpl<AffineMap> &newAllMaps,
                      SmallVectorImpl<SmallVector<int64_t>> &indexVecAxes,
                      SmallVectorImpl<int64_t> &maskAxes,
                      SmallVectorImpl<int64_t> &baseDimsUsingDim0) {
  AffineMap newBaseMap = removeDim0FromMap(baseMap);
  newAllMaps.push_back(newBaseMap);

  for (int64_t i = 0; i < numIndexVecs; ++i) {
    SmallVector<int64_t> axes;
    AffineMap newMap = removeDim0FromIndexVecMap(indexingMaps[1 + i], axes);
    newAllMaps.push_back(newMap);
    indexVecAxes.push_back(std::move(axes));
  }

  if (hasMask) {
    AffineMap newMaskMap =
        removeDim0FromIndexVecMap(indexingMaps.back(), maskAxes);
    newAllMaps.push_back(newMaskMap);
  }

  for (auto [j, expr] : llvm::enumerate(baseMap.getResults())) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      if (dimExpr.getPosition() == 0) {
        baseDimsUsingDim0.push_back(j);
      }
    }
  }
}

/// Extract sliced index vecs and mask for iteration `i` of dim-0 unrolling.
/// Returns the sliced mask (or nullptr if no mask).
static Value
extractSlicesForIteration(OpBuilder &rewriter, Location loc, int64_t i,
                          OperandRange indexVecs, int64_t numIndexVecs,
                          ArrayRef<SmallVector<int64_t>> indexVecAxes,
                          Value mask, ArrayRef<int64_t> maskAxes,
                          SmallVectorImpl<Value> &newIndexVecs) {
  for (int64_t k = 0; k < numIndexVecs; ++k) {
    Value idxVec = indexVecs[k];
    if (!indexVecAxes[k].empty()) {
      for (int64_t axis : indexVecAxes[k]) {
        idxVec = extractVecSlice(rewriter, loc, idxVec, axis, i);
      }
    }
    newIndexVecs.push_back(idxVec);
  }

  if (!mask) {
    return nullptr;
  }
  Value m = mask;
  for (int64_t axis : maskAxes) {
    m = extractVecSlice(rewriter, loc, m, axis, i);
  }
  return m;
}

/// Update base offsets for dim-0 iteration `i`.
static SmallVector<Value>
computeNewOffsets(OpBuilder &rewriter, Location loc, ValueRange offsets,
                  int64_t i, ArrayRef<int64_t> baseDimsUsingDim0) {
  SmallVector<Value> newOffsets(offsets);
  for (int64_t baseDim : baseDimsUsingDim0) {
    Value offset = newOffsets[baseDim];
    Value iVal = arith::ConstantIndexOp::create(rewriter, loc, i);
    newOffsets[baseDim] = arith::AddIOp::create(rewriter, loc, offset, iVal);
  }
  return newOffsets;
}

//===----------------------------------------------------------------------===//
// UnrollTransferGatherDim / UnrollTransferScatterDim
//===----------------------------------------------------------------------===//

/// Unrolls dim 0 of a transfer_gather, reducing vector rank by 1 each
/// application. Sub-gathers are assembled into the result via
/// insert_strided_slice. Stops at rank 1.
struct UnrollTransferGatherDim final : OpRewritePattern<TransferGatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferGatherOp op,
                                PatternRewriter &rewriter) const override {
    VectorType vectorType = op.getVector().getType();
    int64_t rank = vectorType.getRank();
    if (rank <= 1) {
      return rewriter.notifyMatchFailure(op, "already rank <= 1");
    }

    Location loc = op.getLoc();
    int64_t dim0Size = vectorType.getShape()[0];
    SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
    OperandRange indexVecs = op.getIndexVecs();
    int64_t numIndexVecs = indexVecs.size();
    Value mask = op.getMask();

    SmallVector<AffineMap> newAllMaps;
    SmallVector<SmallVector<int64_t>> indexVecAxes;
    SmallVector<int64_t> maskAxes, baseDimsUsingDim0;
    computeUnrollDim0Maps(indexingMaps, numIndexVecs, !!mask, indexingMaps[0],
                          newAllMaps, indexVecAxes, maskAxes,
                          baseDimsUsingDim0);

    SmallVector<int64_t> newShape(vectorType.getShape().drop_front());
    auto newVectorType = VectorType::get(newShape, vectorType.getElementType());

    Value acc = ub::PoisonOp::create(rewriter, loc, vectorType);

    for (int64_t i = 0; i < dim0Size; ++i) {
      SmallVector<Value> newOffsets = computeNewOffsets(
          rewriter, loc, op.getOffsets(), i, baseDimsUsingDim0);

      SmallVector<Value> newIndexVecs;
      Value newMask =
          extractSlicesForIteration(rewriter, loc, i, indexVecs, numIndexVecs,
                                    indexVecAxes, mask, maskAxes, newIndexVecs);

      auto subGather = TransferGatherOp::create(
          rewriter, loc, newVectorType, op.getBase(), newOffsets, newIndexVecs,
          rewriter.getAffineMapArrayAttr(newAllMaps), op.getPadding(), newMask);

      SmallVector<int64_t> offsets(rank, 0);
      offsets[0] = i;
      SmallVector<int64_t> strides(newShape.size(), 1);
      acc = vector::InsertStridedSliceOp::create(
          rewriter, loc, subGather.getResult(), acc, offsets, strides);
    }

    rewriter.replaceOp(op, acc);
    return success();
  }
};

/// Unrolls dim 0 of a transfer_scatter, reducing vector rank by 1 each
/// application. For tensor semantics, sub-scatters are chained via SSA
/// results. For memref semantics, sub-scatters write in-place. Stops at
/// rank 1.
struct UnrollTransferScatterDim final : OpRewritePattern<TransferScatterOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferScatterOp op,
                                PatternRewriter &rewriter) const override {
    VectorType vectorType = op.getVectorType();
    int64_t rank = vectorType.getRank();
    if (rank <= 1) {
      return rewriter.notifyMatchFailure(op, "already rank <= 1");
    }

    Location loc = op.getLoc();
    int64_t dim0Size = vectorType.getShape()[0];
    SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
    OperandRange indexVecs = op.getIndexVecs();
    int64_t numIndexVecs = indexVecs.size();
    Value mask = op.getMask();

    SmallVector<AffineMap> newAllMaps;
    SmallVector<SmallVector<int64_t>> indexVecAxes;
    SmallVector<int64_t> maskAxes, baseDimsUsingDim0;
    computeUnrollDim0Maps(indexingMaps, numIndexVecs, !!mask, indexingMaps[0],
                          newAllMaps, indexVecAxes, maskAxes,
                          baseDimsUsingDim0);

    Value dest = op.getBase();

    for (int64_t i = 0; i < dim0Size; ++i) {
      SmallVector<Value> newOffsets = computeNewOffsets(
          rewriter, loc, op.getOffsets(), i, baseDimsUsingDim0);

      SmallVector<Value> newIndexVecs;
      Value newMask =
          extractSlicesForIteration(rewriter, loc, i, indexVecs, numIndexVecs,
                                    indexVecAxes, mask, maskAxes, newIndexVecs);

      Value vecSlice =
          vector::ExtractOp::create(rewriter, loc, op.getVector(), int64_t{i});

      if (op.hasTensorSemantics()) {
        auto subScatter = TransferScatterOp::create(
            rewriter, loc, dest.getType(), dest, vecSlice, newOffsets,
            newIndexVecs, rewriter.getAffineMapArrayAttr(newAllMaps), newMask);
        dest = subScatter.getResult();
        continue;
      }
      TransferScatterOp::create(rewriter, loc, /*resultTypes=*/TypeRange{},
                                dest, vecSlice, newOffsets, newIndexVecs,
                                rewriter.getAffineMapArrayAttr(newAllMaps),
                                newMask);
    }

    if (op.hasTensorSemantics()) {
      rewriter.replaceOp(op, dest);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

} // namespace

namespace mlir::iree_compiler::IREE::VectorExt {

void populateVectorTransferGatherScatterLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<UnrollTransferGatherDim, UnrollTransferScatterDim>(
      patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::VectorExt
