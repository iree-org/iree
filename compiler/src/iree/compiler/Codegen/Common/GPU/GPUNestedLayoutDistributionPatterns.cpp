// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/PatternApplicator.h"

namespace mlir::iree_compiler {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

/// Helper to linearize the given |ids| with maximum values given as |sizes|.
/// Gets the element ID in terms of |elementCount| and adds the element
/// |offset|. For example,
///
/// IDs = [d0, d1, d2, d3]
/// sizes = [s0, s1, s2, s3]
/// linear_index = d0 * (s1 * s2 * s3)
///              + d1 * (s2 * s3)
///              + d2 * (s3)
///              + d3
/// return element_index = linear_index * |elementCount| + |offset|;
static Value linearizeIndex(OpBuilder &builder, Value offset,
                            ArrayRef<OpFoldResult> ids, ArrayRef<int64_t> sizes,
                            int64_t elementCount) {
  SmallVector<AffineExpr> exprs(ids.size() + 1);
  bindSymbolsList(builder.getContext(), MutableArrayRef{exprs});
  AffineExpr idExpr = builder.getAffineConstantExpr(0);

  for (int i = 0, e = ids.size(); i < e; ++i) {
    if (sizes[i] > 1) {
      // Multiply by the residual threads along this dimension (which must be
      // faster changing than all previous dimensions) and add the id for this
      // dimension.
      idExpr = idExpr * builder.getAffineConstantExpr(sizes[i]) + exprs[i];
    }
  }
  idExpr = idExpr * builder.getAffineConstantExpr(elementCount);
  idExpr = idExpr + exprs.back();
  SmallVector<OpFoldResult> mapArgs(ids);
  mapArgs.push_back(offset);
  return affine::makeComposedAffineApply(
             builder, offset.getLoc(),
             AffineMap::get(0, mapArgs.size(), idExpr), mapArgs)
      .getResult();
}

/// Given a set of base transfer |indices|, |offsets| for the batch/outer
/// dimensions, and distributed warp and thread indices, computes the indices
/// of the distributed transfer operation based on the |vectorLayout|.
static SmallVector<Value> getTransferIndicesFromNestedLayout(
    OpBuilder &b, ValueRange indices, ArrayRef<int64_t> offsets,
    NestedLayoutAttr vectorLayout, AffineMap permutationMap,
    ArrayRef<Value> warpIndices, ArrayRef<Value> threadIndices) {
  auto isBroadcast = [](AffineExpr expr) {
    if (auto constExpr = dyn_cast<AffineConstantExpr>(expr))
      return constExpr.getValue() == 0;
    return false;
  };
  int64_t rank = vectorLayout.getBatchOrder().size();
  // Permute the batch and outer vector offsets to match the order of
  // the vector dimensions using the inverse of the batch/offset order.
  SmallVector<int64_t> batchOffsets =
      applyPermutation(ArrayRef<int64_t>(offsets.begin(), rank),
                       invertPermutationVector(vectorLayout.getBatchOrder()));
  SmallVector<int64_t> outerVectorOffsets =
      applyPermutation(ArrayRef<int64_t>(offsets.begin() + rank, rank),
                       invertPermutationVector(vectorLayout.getOuterOrder()));

  SmallVector<Value> slicedIndices(indices.begin(), indices.end());
  for (const auto &[i, dim] : llvm::enumerate(permutationMap.getResults())) {
    // Broadcasted dimension offsets can be used as-is; the read index is
    // invariant of the thread in such cases (and illegal for writes).
    if (isBroadcast(dim)) {
      continue;
    }
    unsigned pos = cast<AffineDimExpr>(dim).getPosition();
    SmallVector<OpFoldResult> ids = {
        warpIndices[i], b.getIndexAttr(batchOffsets[i]),
        b.getIndexAttr(outerVectorOffsets[i]), threadIndices[i]};
    // The order in which a vector dimension is "tiled" is
    // subgroups -> batches -> outer vectors -> threads -> elements
    SmallVector<int64_t> sizes = {vectorLayout.getSubgroupsPerWorkgroup()[i],
                                  vectorLayout.getBatchesPerSubgroup()[i],
                                  vectorLayout.getOutersPerBatch()[i],
                                  vectorLayout.getThreadsPerOuter()[i]};
    slicedIndices[pos] = linearizeIndex(b, indices[pos], ids, sizes,
                                        vectorLayout.getElementsPerThread()[i]);
  }
  return slicedIndices;
}

static SmallVector<int64_t> getLoopOrder(NestedLayoutAttr vectorLayout) {
  int64_t rank = vectorLayout.getBatchOrder().size();
  // Let the unroll order first unroll the batch dimensions, then the
  // outer vector dimensions. We unroll in the order specified by the
  // layout.
  SmallVector<int64_t> loopOrder;
  int64_t base = 0;
  for (auto b : vectorLayout.getBatchOrder()) {
    loopOrder.push_back(base + b);
  }
  base += rank;
  // We must unroll along the outer dimensions as well to match the rank
  // requirements of vector transfer ops (<= memref rank up to broadcasts).
  for (auto o : vectorLayout.getOuterOrder()) {
    loopOrder.push_back(base + o);
  }
  base += rank;
  for (int i = 0, e = rank; i < e; ++i) {
    loopOrder.push_back(base + i);
  }
  return loopOrder;
}

static SmallVector<int64_t>
getElementVectorTileShape(NestedLayoutAttr vectorLayout) {
  int64_t rank = vectorLayout.getBatchOrder().size();
  SmallVector<int64_t> tileShape = vectorLayout.getDistributedShape();
  // We tile to a vector with BATCH, OUTER, and ELEMENT dimensions. So to access
  // the subvector only containing elements, we need indices in all BATCH and
  // OUTER (rank * 2) dimensions to have tile size 1.
  for (int i = 0, e = rank * 2; i < e; ++i) {
    tileShape[i] = 1;
  }
  return tileShape;
}

/// Computes the warp and thread indices for the given vector layout from a
/// single linearized thread ID.
static void populateWarpAndThreadIndices(RewriterBase &rewriter, Value threadId,
                                         NestedLayoutAttr vectorLayout,
                                         SmallVector<Value> &warpIndices,
                                         SmallVector<Value> &threadIndices) {
  int64_t rank = vectorLayout.getBatchOrder().size();
  // The delinearized thread IDs are returned from outer most to inner most,
  // i.e. before applying the layout described dimensions ordering.
  SmallVector<Value> threadIds =
      vectorLayout.computeThreadIds(threadId, rewriter);

  // Subgroup and thread (lane) indices normalized to the order in which
  // they are used by each dimension.
  warpIndices =
      llvm::to_vector(llvm::map_range(vectorLayout.getSubgroupOrder(),
                                      [&](int64_t i) { return threadIds[i]; }));
  threadIndices = llvm::to_vector(
      llvm::map_range(vectorLayout.getThreadOrder(),
                      [&](int64_t i) { return threadIds[i + rank]; }));
}

namespace {

/// Pattern to distribute `vector.transfer_read` ops with nested layouts.
struct DistributeTransferReadNestedLayoutAttr final
    : OpDistributionPattern<vector::TransferReadOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferReadNestedLayoutAttr(MLIRContext *context, Value threadId)
      : OpDistributionPattern(context), threadId(threadId) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // TODO: Support masking.
    if (readOp.getMask()) {
      return failure();
    }
    NestedLayoutAttr vectorLayout =
        dyn_cast<NestedLayoutAttr>(signature[readOp.getResult()]);
    if (!vectorLayout) {
      return failure();
    }

    // Guard on memrefs for distribution. In isolation this pattern is agnostic
    // to tensors or memrefs.
    if (!isa<MemRefType>(readOp.getSource().getType())) {
      return failure();
    }

    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    SmallVector<int64_t> tileShape = getElementVectorTileShape(vectorLayout);
    SmallVector<int64_t> loopOrder = getLoopOrder(vectorLayout);
    int64_t rank = vectorLayout.getBatchOrder().size();

    Type elementType = readOp.getSource().getType().getElementType();
    auto vectorType = VectorType::get(distShape, elementType);
    // The shape of the vector we read is pre-permutation. The permutation is
    // a transpose on the resulting read vector.
    auto innerVectorType =
        VectorType::get(vectorLayout.getElementsPerThread(), elementType);

    // Initialize the full distributed vector for unrolling the batch/outer
    // vector dimensions.
    Value zero = rewriter.create<arith::ConstantOp>(
        readOp.getLoc(), vectorType, rewriter.getZeroAttr(vectorType));
    VectorValue acc = cast<VectorValue>(zero);

    SmallVector<Value> warpIndices, threadIndices;
    populateWarpAndThreadIndices(rewriter, threadId, vectorLayout, warpIndices,
                                 threadIndices);

    ValueRange indices = readOp.getIndices();
    SmallVector<int64_t> strides(rank, 1);
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distShape, tileShape, loopOrder)) {
      SmallVector<Value> slicedIndices = getTransferIndicesFromNestedLayout(
          rewriter, indices, offsets, vectorLayout, readOp.getPermutationMap(),
          warpIndices, threadIndices);

      Value slicedRead = rewriter.create<vector::TransferReadOp>(
          readOp.getLoc(), innerVectorType, readOp.getSource(), slicedIndices,
          readOp.getPermutationMapAttr(), readOp.getPadding(), readOp.getMask(),
          readOp.getInBoundsAttr());
      // Transpose to the element order.
      if (!isIdentityPermutation(vectorLayout.getElementOrder())) {
        slicedRead = rewriter.create<vector::TransposeOp>(
            slicedRead.getLoc(), slicedRead, vectorLayout.getElementOrder());
      }

      acc = rewriter.create<vector::InsertStridedSliceOp>(
          readOp.getLoc(), slicedRead, acc, offsets, strides);
    }

    replaceOpWithDistributedValues(rewriter, readOp, acc);
    return success();
  }

  Value threadId;
};

/// Pattern to distribute `vector.transfer_write` ops with nested layouts.
struct DistributeTransferWriteNestedLayoutAttr final
    : OpDistributionPattern<vector::TransferWriteOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferWriteNestedLayoutAttr(MLIRContext *context, Value threadId)
      : OpDistributionPattern(context), threadId(threadId) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // TODO: Support masking.
    if (writeOp.getMask()) {
      return failure();
    }
    NestedLayoutAttr vectorLayout =
        dyn_cast<NestedLayoutAttr>(signature[writeOp.getVector()]);
    if (!vectorLayout) {
      return failure();
    }

    if (!isa<MemRefType>(writeOp.getSource().getType())) {
      return failure();
    }

    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    SmallVector<int64_t> tileShape = getElementVectorTileShape(vectorLayout);
    SmallVector<int64_t> loopOrder = getLoopOrder(vectorLayout);
    int64_t rank = vectorLayout.getBatchOrder().size();

    SmallVector<Value> warpIndices, threadIndices;
    populateWarpAndThreadIndices(rewriter, threadId, vectorLayout, warpIndices,
                                 threadIndices);

    Value distributedVector =
        getDistributed(rewriter, writeOp.getVector(), vectorLayout);

    ValueRange indices = writeOp.getIndices();
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distShape, tileShape, loopOrder)) {
      SmallVector<Value> slicedIndices = getTransferIndicesFromNestedLayout(
          rewriter, indices, offsets, vectorLayout, writeOp.getPermutationMap(),
          warpIndices, threadIndices);

      // Extract the "element vector" from the inner most dimensions. All outer
      // dimensions are either unrolled or distributed such that this is a
      // contiguous slice.
      ArrayRef<int64_t> offsetArray(offsets);
      Value slicedVector = rewriter.create<vector::ExtractOp>(
          writeOp.getLoc(), distributedVector,
          offsetArray.take_front(rank * 2));
      // Transpose to the native dimension order.
      if (!isIdentityPermutation(vectorLayout.getElementOrder())) {
        slicedVector = rewriter.create<vector::TransposeOp>(
            slicedVector.getLoc(), slicedVector,
            invertPermutationVector(vectorLayout.getElementOrder()));
      }
      rewriter.create<vector::TransferWriteOp>(
          writeOp.getLoc(), slicedVector, writeOp.getSource(), slicedIndices,
          writeOp.getPermutationMapAttr(), writeOp.getMask(),
          writeOp.getInBoundsAttr());
    }

    rewriter.eraseOp(writeOp);
    return success();
  }

  Value threadId;
};

struct DistributeBroadcastNestedLayoutAttr final
    : OpDistributionPattern<vector::BroadcastOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcastOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {

    VectorValue source = dyn_cast<VectorValue>(broadcastOp.getSource());
    if (!source) {
      // TODO: Add support for scalar broadcasting.
      return failure();
    }
    NestedLayoutAttr sourceLayout =
        dyn_cast<NestedLayoutAttr>(signature[source]);
    if (!sourceLayout) {
      return failure();
    }

    VectorValue vector = broadcastOp.getVector();
    NestedLayoutAttr vectorLayout =
        dyn_cast<NestedLayoutAttr>(signature[vector]);
    if (!vectorLayout) {
      return failure();
    }

    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    Type elementType =
        llvm::cast<ShapedType>(vector.getType()).getElementType();
    auto vectorType = VectorType::get(distShape, elementType);
    Location loc = broadcastOp.getLoc();
    Value accumulator = rewriter.create<arith::ConstantOp>(
        loc, vectorType, rewriter.getZeroAttr(vectorType));

    int64_t rank = vectorLayout.getBatchOrder().size();
    int64_t sourceRank = sourceLayout.getBatchOrder().size();
    // We unroll along the outer dimensions as well for a similar reason to the
    // transfer ops. `vector.broadcast` can only broadcast along outer dims, so
    // mixing broadcasted and un-broadcasted element/outer dims can't be
    // represented with a broadcast.
    SmallVector<int64_t> tileShape = getElementVectorTileShape(vectorLayout);
    SmallVector<int64_t> loopOrder = getLoopOrder(vectorLayout);

    Value distributedSource = getDistributed(rewriter, source, sourceLayout);

    VectorType tileType =
        VectorType::get(applyPermutation(vectorLayout.getElementsPerThread(),
                                         vectorLayout.getElementOrder()),
                        elementType);
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distShape, tileShape, loopOrder)) {
      ArrayRef<int64_t> offsetsRef(offsets);
      // Invert the permutations on the batch/outer offsets to get the offsets
      // in the order of the vector dimensions.
      SmallVector<int64_t> permutedBatchOffsets = applyPermutation(
          offsetsRef.slice(0, rank),
          invertPermutationVector(vectorLayout.getBatchOrder()));
      SmallVector<int64_t> permutedOuterOffsets = applyPermutation(
          offsetsRef.slice(rank, rank),
          invertPermutationVector(vectorLayout.getOuterOrder()));

      // Slice out the last |sourceRank| dimensions which is the inner
      // broadcasted shape.
      ArrayRef<int64_t> batchSourceOffsets =
          ArrayRef<int64_t>(permutedBatchOffsets)
              .slice(rank - sourceRank, sourceRank);
      ArrayRef<int64_t> outerSourceOffsets =
          ArrayRef<int64_t>(permutedOuterOffsets)
              .slice(rank - sourceRank, sourceRank);

      // Construct the list of source offsets based on the batch order of the
      // broadcasted vector.
      SmallVector<int64_t> sourceOffsets;
      sourceOffsets.append(
          applyPermutation(batchSourceOffsets, sourceLayout.getBatchOrder()));
      sourceOffsets.append(
          applyPermutation(outerSourceOffsets, sourceLayout.getOuterOrder()));

      // Extract a slice of the input to be broadcasted.
      Value slice = rewriter.create<vector::ExtractOp>(loc, distributedSource,
                                                       sourceOffsets);
      // TODO: Support non-trivial element orders.
      if (vector::isBroadcastableTo(slice.getType(), tileType) !=
          vector::BroadcastableToResult::Success) {
        return failure();
      }
      Value broadcastedSlice =
          rewriter.create<vector::BroadcastOp>(loc, tileType, slice);
      // Insert into the broadcasted destination vector.
      accumulator = rewriter.create<vector::InsertOp>(
          loc, broadcastedSlice, accumulator, offsetsRef.take_front(rank * 2));
    }

    replaceOpWithDistributedValues(rewriter, broadcastOp, accumulator);
    return success();
  }
};

} // namespace

void populateGPUDistributeNestedLayoutAttrPatterns(
    Value threadId, RewritePatternSet &patterns) {
  patterns.add<DistributeTransferReadNestedLayoutAttr,
               DistributeTransferWriteNestedLayoutAttr>(patterns.getContext(),
                                                        threadId);
  patterns.add<DistributeBroadcastNestedLayoutAttr>(patterns.getContext());
}

}; // namespace mlir::iree_compiler
