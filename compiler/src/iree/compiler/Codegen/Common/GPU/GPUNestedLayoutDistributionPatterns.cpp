// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Utils/Permutation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
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
  int64_t rank = vectorLayout.getRank();
  // Permute the batch and outer vector offsets to match the order of
  // the vector dimensions using the inverse of the batch/offset order.
  ArrayRef<int64_t> batchOffsets(offsets.begin(), rank);
  ArrayRef<int64_t> outerVectorOffsets(offsets.begin() + rank, rank);

  SmallVector<Value> slicedIndices(indices.begin(), indices.end());
  for (const auto &[i, dim] : llvm::enumerate(permutationMap.getResults())) {
    // Broadcasted dimension offsets can be used as-is; the read index is
    // invariant of the thread in such cases (and illegal for writes).
    if (isBroadcast(dim)) {
      continue;
    }
    unsigned pos = cast<AffineDimExpr>(dim).getPosition();
    Value offset = indices[pos];
    int64_t elementCount = vectorLayout.getElementTile()[i];
    Location loc = offset.getLoc();
    SmallVector<Value> ids = {
        warpIndices[i], b.create<arith::ConstantIndexOp>(loc, batchOffsets[i]),
        b.create<arith::ConstantIndexOp>(loc, outerVectorOffsets[i]),
        threadIndices[i], offset};
    // The order in which a vector dimension is "tiled" is
    // subgroups -> batches -> outer vectors -> threads -> elements
    SmallVector<int64_t> sizes = {
        vectorLayout.getSubgroupTile()[i], vectorLayout.getBatchTile()[i],
        vectorLayout.getOuterTile()[i], vectorLayout.getThreadTile()[i],
        elementCount};
    // The offset is often not an offset within `elementCount`, so, in general,
    // we can't mark this `disjoint`. However, if `offset` is known to be
    // a constant less than `elementCount`, we can do this, unlocking
    // potential optimizations.
    bool disjoint = false;
    if (std::optional<int64_t> offsetConst = getConstantIntValue(offset))
      disjoint = *offsetConst < elementCount;
    slicedIndices[pos] =
        b.create<affine::AffineLinearizeIndexOp>(loc, ids, sizes, disjoint);
  }
  return slicedIndices;
}

static SmallVector<int64_t>
getElementVectorTileShape(NestedLayoutAttr vectorLayout) {
  int64_t rank = vectorLayout.getRank();
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
static LogicalResult populateWarpAndThreadIndices(
    RewriterBase &rewriter, Value threadId, int64_t subgroupSize,
    NestedLayoutAttr vectorLayout, SmallVector<Value> &warpIndices,
    SmallVector<Value> &threadIndices) {
  // The delinearized thread IDs are returned from outer most to inner most,
  // i.e. before applying the layout described dimensions ordering.
  int64_t rank = vectorLayout.getRank();
  SmallVector<Value> threadIds =
      vectorLayout.computeThreadIds(threadId, subgroupSize, rewriter);
  if (threadIds.empty() && rank != 0)
    return failure();
  warpIndices = SmallVector<Value>(threadIds.begin(), threadIds.begin() + rank);
  threadIndices = SmallVector<Value>(threadIds.begin() + rank,
                                     threadIds.begin() + 2 * rank);
  return success();
}

namespace {

/// Pattern to distribute `vector.transfer_read` ops with nested layouts.
struct DistributeTransferRead final
    : OpDistributionPattern<vector::TransferReadOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferRead(MLIRContext *context, Value threadId,
                         int64_t subgroupSize)
      : OpDistributionPattern(context), threadId(threadId),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // TODO: Support masking.
    if (readOp.getMask()) {
      return rewriter.notifyMatchFailure(readOp, "unimplemented: masked read");
    }
    NestedLayoutAttr vectorLayout =
        dyn_cast<NestedLayoutAttr>(signature[readOp.getResult()]);
    if (!vectorLayout) {
      return rewriter.notifyMatchFailure(readOp,
                                         "non-nested transfer_read layout");
    }

    // Guard on memrefs for distribution. In isolation this pattern is agnostic
    // to tensors or memrefs.
    if (!isa<MemRefType>(readOp.getSource().getType())) {
      return rewriter.notifyMatchFailure(readOp,
                                         "distribution expects memrefs");
    }

    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    SmallVector<int64_t> tileShape = getElementVectorTileShape(vectorLayout);
    int64_t rank = vectorLayout.getRank();

    Type elementType = readOp.getSource().getType().getElementType();
    auto vectorType = VectorType::get(distShape, elementType);
    // The shape of the vector we read is pre-permutation. The permutation is
    // a transpose on the resulting read vector.
    auto innerVectorType =
        VectorType::get(vectorLayout.getElementTile(), elementType);

    // Initialize the full distributed vector for unrolling the batch/outer
    // vector dimensions.
    Value zero = rewriter.create<arith::ConstantOp>(
        readOp.getLoc(), vectorType, rewriter.getZeroAttr(vectorType));
    VectorValue acc = cast<VectorValue>(zero);

    SmallVector<Value> warpIndices, threadIndices;
    if (failed(populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
                                            vectorLayout, warpIndices,
                                            threadIndices))) {
      return rewriter.notifyMatchFailure(
          readOp, "warp or thread tiles have overlapping strides");
    }

    ValueRange indices = readOp.getIndices();
    SmallVector<int64_t> strides(rank, 1);
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distShape, tileShape)) {
      SmallVector<Value> slicedIndices = getTransferIndicesFromNestedLayout(
          rewriter, indices, offsets, vectorLayout, readOp.getPermutationMap(),
          warpIndices, threadIndices);

      VectorValue slicedRead = rewriter.create<vector::TransferReadOp>(
          readOp.getLoc(), innerVectorType, readOp.getSource(), slicedIndices,
          readOp.getPermutationMapAttr(), readOp.getPadding(), readOp.getMask(),
          readOp.getInBoundsAttr());

      if (acc.getType().getRank() == 0) {
        // TODO: This should really be a folding pattern in
        // insert_strided_slice, but instead insert_strided_slice just doesn't
        // support 0-d vectors...
        acc = slicedRead;
      } else {
        acc = rewriter.create<vector::InsertStridedSliceOp>(
            readOp.getLoc(), slicedRead, acc, offsets, strides);
      }
    }

    replaceOpWithDistributedValues(rewriter, readOp, acc);
    return success();
  }

  Value threadId;
  int64_t subgroupSize;
};

/// Pattern to distribute `vector.transfer_write` ops with nested layouts.
struct DistributeTransferWrite final
    : OpDistributionPattern<vector::TransferWriteOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferWrite(MLIRContext *context, Value threadId,
                          int64_t subgroupSize)
      : OpDistributionPattern(context), threadId(threadId),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // TODO: Support masking.
    if (writeOp.getMask()) {
      return rewriter.notifyMatchFailure(writeOp,
                                         "unimplemented: masked write");
    }
    NestedLayoutAttr vectorLayout =
        dyn_cast<NestedLayoutAttr>(signature[writeOp.getVector()]);
    if (!vectorLayout) {
      return rewriter.notifyMatchFailure(writeOp,
                                         "non-nested transfer_write layout");
    }

    if (!isa<MemRefType>(writeOp.getSource().getType())) {
      return rewriter.notifyMatchFailure(writeOp,
                                         "distribution expects memrefs");
    }

    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    SmallVector<int64_t> tileShape = getElementVectorTileShape(vectorLayout);
    int64_t rank = vectorLayout.getRank();

    SmallVector<Value> warpIndices, threadIndices;
    if (failed(populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
                                            vectorLayout, warpIndices,
                                            threadIndices))) {
      return rewriter.notifyMatchFailure(
          writeOp, "warp or thread tiles have overlapping strides");
    }

    Value distributedVector =
        getDistributed(rewriter, writeOp.getVector(), vectorLayout);

    ValueRange indices = writeOp.getIndices();
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distShape, tileShape)) {
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
      // Promote the slicedVector to 0-d vector if it is a scalar.
      if (!isa<VectorType>(slicedVector.getType())) {
        auto promotedType =
            VectorType::get({}, getElementTypeOrSelf(slicedVector));
        slicedVector = rewriter.create<vector::BroadcastOp>(
            writeOp.getLoc(), promotedType, slicedVector);
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
  int64_t subgroupSize;
};

struct DistributeBroadcast final : OpDistributionPattern<vector::BroadcastOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcastOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Location loc = broadcastOp.getLoc();
    VectorValue dstVector = broadcastOp.getVector();
    auto vectorLayout = dyn_cast<NestedLayoutAttr>(signature[dstVector]);
    if (!vectorLayout) {
      return rewriter.notifyMatchFailure(broadcastOp,
                                         "non-nested result vector layout");
    }
    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    Type elementType =
        llvm::cast<ShapedType>(dstVector.getType()).getElementType();
    auto vectorType = VectorType::get(distShape, elementType);

    VectorValue srcVector = dyn_cast<VectorValue>(broadcastOp.getSource());
    // If the srcVector is a scalar (like f32) we proceed with the scalar
    // distribution branch.
    if (!srcVector) {
      // The way distribution currently works, there is no partial thread
      // distribution, so a scalar is available to all threads. Scalar
      // distribution is simply a broadcast from scalar to the distributed
      // result shape.
      Value source = broadcastOp.getSource();
      VectorValue accumulator =
          rewriter.create<vector::BroadcastOp>(loc, vectorType, source);
      replaceOpWithDistributedValues(rewriter, broadcastOp, accumulator);
      return success();
    }

    auto sourceLayout = dyn_cast<NestedLayoutAttr>(signature[srcVector]);
    if (!sourceLayout) {
      return rewriter.notifyMatchFailure(broadcastOp,
                                         "non-nested source vector layout");
    }

    Value accumulator = rewriter.create<arith::ConstantOp>(
        loc, vectorType, rewriter.getZeroAttr(vectorType));

    int64_t rank = vectorLayout.getRank();
    // We unroll along both the batch and outer dimensions for a similar reason
    // to the transfer ops. `vector.broadcast` can only broadcast along outer
    // dims, so mixing broadcasted and un-broadcasted element/outer dims can't
    // be represented with a single `vector.broadcast`.
    SmallVector<int64_t> resultVectorUnrollShape =
        getElementVectorTileShape(vectorLayout);

    Value distributedSource = getDistributed(rewriter, srcVector, sourceLayout);

    VectorType broadcastTargetType =
        VectorType::get(vectorLayout.getElementTile(), elementType);

    int64_t sourceRank = sourceLayout.getRank();

    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distShape, resultVectorUnrollShape)) {
      ArrayRef<int64_t> offsetsRef(offsets);

      // Slice out the last |sourceRank| dimensions which is the inner
      // broadcasted shape.
      ArrayRef<int64_t> batchSourceOffsets =
          offsetsRef.slice(rank - sourceRank, sourceRank);
      ArrayRef<int64_t> outerSourceOffsets =
          offsetsRef.slice(2 * rank - sourceRank, sourceRank);

      // Construct the list of source offsets based on the batch/outer order of
      // the broadcasted vector. This is because we need to compute the offsets
      // into the distributed source vector with the distributed permutation.
      SmallVector<int64_t> sourceOffsets;
      sourceOffsets.append(batchSourceOffsets.begin(),
                           batchSourceOffsets.end());
      sourceOffsets.append(outerSourceOffsets.begin(),
                           outerSourceOffsets.end());

      // Extract a slice of the input to be broadcasted.
      Value slice = rewriter.create<vector::ExtractOp>(loc, distributedSource,
                                                       sourceOffsets);
      // TODO: Support non-trivial element orders.
      if (vector::isBroadcastableTo(slice.getType(), broadcastTargetType) !=
          vector::BroadcastableToResult::Success) {
        return rewriter.notifyMatchFailure(
            broadcastOp,
            "unimplemented: non-trivial broadcast source element order");
      }
      Value broadcastedSlice =
          rewriter.create<vector::BroadcastOp>(loc, broadcastTargetType, slice);
      // Insert into the broadcasted destination vector.
      accumulator = rewriter.create<vector::InsertOp>(
          loc, broadcastedSlice, accumulator, offsetsRef.take_front(rank * 2));
    }

    replaceOpWithDistributedValues(rewriter, broadcastOp, accumulator);
    return success();
  }
};

static int64_t getShuffleOffset(NestedLayoutAttr layout, int64_t dim) {
  return layout.getThreadStrides()[dim];
}

static int64_t getShuffleWidth(NestedLayoutAttr layout, int64_t dim) {
  return layout.getThreadTile()[dim];
}

/// The lowering for multi_reduction is done in two steps:
///   1. Local Reduce: Each thread reduces all elements carried by it along
///      the reduction dimensions. This is the batch, outer and element dims.
///   2. Thread Reduce: Each thread reduces result of step 1 across threads
///      by doing a butterfly shuffle.
///   3. Accumulator Reduce: Each thread reduces it's intermediate reduced
///      results with the accumulator it holds.
///   4. Subgroup reduce : each subgroup will store the partial reductions
///      to shared memory and will be reloaded into a layout where partial
///      reductions will be placed inside threads.
struct DistributeMultiReduction final
    : OpDistributionPattern<vector::MultiDimReductionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeMultiReduction(MLIRContext *context, int64_t subgroupSize,
                           int64_t maxBitsPerShuffle, int64_t benefit = 1)
      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
        maxBitsPerShuffle(maxBitsPerShuffle) {}

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReduceOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue srcVector = multiReduceOp.getSource();
    Value acc = multiReduceOp.getAcc();
    Value res = multiReduceOp.getResult();
    auto accVector = dyn_cast<VectorValue>(acc);
    auto resVector = dyn_cast<VectorValue>(res);

    auto srcLayout = dyn_cast_or_null<NestedLayoutAttr>(signature[srcVector]);
    if (!srcLayout) {
      return rewriter.notifyMatchFailure(multiReduceOp,
                                         "expected nested layout attr");
    }

    Type elemTy = srcVector.getType().getElementType();
    unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
    if (elemBitwidth > maxBitsPerShuffle) {
      return rewriter.notifyMatchFailure(
          multiReduceOp,
          llvm::formatv("element bitwidth greater than maxBitsPerShuffle",
                        elemBitwidth, maxBitsPerShuffle));
    }

    VectorValue disSrc =
        getDistributed(rewriter, srcVector, signature[srcVector]);

    Value disAcc;
    if (accVector) {
      disAcc = getDistributed(rewriter, accVector, signature[accVector]);
    } else {
      // Scalars are always distributed to all threads already.
      disAcc = multiReduceOp.getAcc();
    }

    Location loc = multiReduceOp.getLoc();
    SmallVector<bool> reducedDims = multiReduceOp.getReductionMask();
    int64_t rank = srcVector.getType().getRank();

    // Do thread local reduce.

    // The distributed reduction mask is simply the same mask appended
    // thrice.
    SmallVector<bool> distributedReductionMask;
    distributedReductionMask.reserve(3 * rank);
    for (int i = 0; i < 3; ++i) {
      distributedReductionMask.append(reducedDims.begin(), reducedDims.end());
    }
    Value localInit = getCombiningIdentityValue(
        loc, rewriter, multiReduceOp.getKind(), disAcc.getType());
    auto localReduction = rewriter.create<vector::MultiDimReductionOp>(
        loc, disSrc, localInit, distributedReductionMask,
        multiReduceOp.getKind());

    VectorValue locallyReduced;
    if (accVector) {
      locallyReduced = dyn_cast<VectorValue>(localReduction.getResult());
    } else {
      // Broadcast scalar accumulator to vector.
      VectorType vecType = VectorType::get(ArrayRef{int64_t(1)}, elemTy);
      locallyReduced = rewriter.create<vector::BroadcastOp>(
          loc, vecType, localReduction.getResult());
    }

    assert(locallyReduced && "result should have been a vector");

    // Flatten the locally reduced value.
    VectorValue threadReduced = locallyReduced;
    VectorType shaped = locallyReduced.getType();
    bool hasThreadReductions =
        llvm::any_of(multiReduceOp.getReductionDims(), [&](int64_t rDim) {
          return srcLayout.getThreadTile()[rDim] > 1;
        });
    if (hasThreadReductions) {
      int64_t numElements = shaped.getNumElements();
      SmallVector<int64_t> flatShape(1, numElements);
      VectorType flatVecType = VectorType::get(flatShape, elemTy);
      VectorValue flat = rewriter.create<vector::ShapeCastOp>(loc, flatVecType,
                                                              locallyReduced);

      // Do inter-thread/warp reduce.
      FailureOr<VectorValue> threadReducedFlat = doThreadReduction(
          rewriter, srcLayout, flat, multiReduceOp.getKind(), reducedDims);
      if (failed(threadReducedFlat)) {
        return failure();
      }

      // Do reduction against accumulator, which needs to be done after thread
      // reduction.
      threadReduced = rewriter.create<vector::ShapeCastOp>(
          loc, shaped, threadReducedFlat.value());
    }

    if (!accVector) {
      // Broadcast the scalar (e.g., f32) to a vector type (e.g., vector<f32>)
      // because the following implementation requires the operand to be a
      // vector.
      disAcc = rewriter.create<vector::BroadcastOp>(loc, shaped, disAcc);
    }

    bool hasSubgroupReductions =
        llvm::any_of(multiReduceOp.getReductionDims(), [&](int64_t rDim) {
          return srcLayout.getSubgroupTile()[rDim] > 1;
        });
    // We can exit here if its just a subgroup reduction.
    if (!hasSubgroupReductions) {
      Value accReduction = vector::makeArithReduction(
          rewriter, loc, multiReduceOp.getKind(), threadReduced, disAcc);
      auto accReduced = dyn_cast<VectorValue>(accReduction);
      if (!accReduced) {
        return failure();
      }
      if (resVector) {
        replaceOpWithDistributedValues(rewriter, multiReduceOp, accReduced);
      } else {
        Value accReducedVal = rewriter.create<vector::ExtractOp>(
            loc, accReduction, ArrayRef{int64_t(0)});
        replaceOpWithDistributedValues(rewriter, multiReduceOp, accReducedVal);
      }
      return success();
    }
    // do inter-subgroup reductions
    Value subgroupReduced = doSubgroupReduction(
        rewriter, loc, srcVector, srcLayout, multiReduceOp.getReductionDims(),
        threadReduced, multiReduceOp.getKind(), acc, signature[resVector]);
    rewriter.replaceOp(multiReduceOp, subgroupReduced);
    return success();
  }

  FailureOr<VectorValue> doThreadReduction(RewriterBase &rewriter,
                                           NestedLayoutAttr layout,
                                           VectorValue flat,
                                           vector::CombiningKind kind,
                                           ArrayRef<bool> reductionMask) const {
    VectorType flatVecType = flat.getType();
    int64_t numElements = flatVecType.getNumElements();
    Location loc = flat.getLoc();

    auto constOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(flatVecType));
    auto res = llvm::cast<VectorValue>(constOp.getResult());

    for (unsigned i = 0; i < numElements; ++i) {
      Value extracted = rewriter.create<vector::ExtractOp>(loc, flat, i);

      // Reduce across all reduction dimensions 1-by-1.
      for (unsigned i = 0, e = reductionMask.size(); i != e; ++i) {
        if (reductionMask[i]) {
          int64_t offset = getShuffleOffset(layout, i);
          int64_t width = getShuffleWidth(layout, i);
          assert(offset <= std::numeric_limits<uint32_t>::max() &&
                 width <= std::numeric_limits<uint32_t>::max());

          extracted = rewriter.create<gpu::SubgroupReduceOp>(
              loc, extracted, combiningKindToAllReduce(kind),
              /*uniform=*/false, /*cluster_size=*/width,
              /*cluster_stride=*/offset);
        }
      }

      res = rewriter.create<vector::InsertOp>(loc, extracted, res, i);
    }
    return res;
  }

  // The reductions across subgroups are performed
  // as follows:
  // 1) Re-cover the subgroup-local result as the same rank as the
  //    input vector
  // 2) Write the subgroup-local reduced vector to shared memory
  // 3) Read the subgroup-local reduced vector where partially reduced
  //    subgroup tile is read as the element tile.
  // 4) Perform a second reduction to complete the reduction.
  Value doSubgroupReduction(PatternRewriter &rewriter, Location loc,
                            VectorValue srcVector, NestedLayoutAttr srcLayout,
                            ArrayRef<int64_t> reductionDims,
                            VectorValue threadReduced,
                            vector::CombiningKind kind, Value acc,
                            VectorLayoutInterface resLayout) const {
    // Subgroup-local / thread-local vector.multi_reduce operations
    // will remove the reduction dimensions by definition.
    // e.g.:
    // p1 x p2 x p3 x r2 x r1 --> p1 x p2 x p3
    // However, the reduction is not complete until inter-subgroup results
    // are combined. Therefore, we need to maintain the rank to get them back to
    // the SIMD domain to re-layout the vector.
    // Thus, we re-insert the reduction dimensions in
    // their original positions as :
    // p1 x p2 x p3 -> p1 x p2 x p3 x 1 x 1
    int64_t rank = srcLayout.getRank();
    SmallVector<int64_t> partialReducedDistributedShape =
        srcLayout.getDistributedShape();
    for (int64_t tileGroupIdx : llvm::seq<int64_t>(3)) {
      int64_t tileGroupOffset = tileGroupIdx * rank;
      for (int64_t rDim : reductionDims) {
        partialReducedDistributedShape[tileGroupOffset + rDim] = 1;
      }
    }
    VectorType partialReducedDistributedType = VectorType::get(
        partialReducedDistributedShape, srcVector.getType().getElementType());
    Value isoRankThreadReduced = rewriter.create<vector::ShapeCastOp>(
        loc, partialReducedDistributedType, threadReduced);

    SmallVector<int64_t> preDistrShape =
        srcLayout.getUndistributedPackedShape();
    SmallVector<int64_t> partialReductionShape =
        llvm::to_vector(srcVector.getType().getShape());
    for (int64_t rDim : reductionDims) {
      // The first #rank elements will form the subgroup tile
      // Here we replace the input shape with subgroup tile
      // because every other tile is reduced except the subgroup
      // tile.
      partialReductionShape[rDim] = preDistrShape[rDim];
    }
    auto workgroupMemoryAddressSpace = Attribute(gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::AddressSpace::Workgroup));
    MemRefType allocType = MemRefType::get(
        partialReductionShape, srcVector.getType().getElementType(),
        AffineMap(), workgroupMemoryAddressSpace);
    auto alloc = rewriter.create<memref::AllocOp>(loc, allocType);
    VectorType unDistributedType = VectorType::get(
        partialReductionShape, srcVector.getType().getElementType());
    Value undistrWrite = rewriter.create<IREE::VectorExt::ToSIMDOp>(
        loc, unDistributedType, isoRankThreadReduced);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(unDistributedType.getRank(), c0);
    SmallVector<bool> inBounds(unDistributedType.getRank(), true);
    // Insert gpu.barrier to make sure previuos iteration
    // of batch loop has fully read the subgroup partial
    // reductions.
    rewriter.create<gpu::BarrierOp>(loc);
    auto write = rewriter.create<vector::TransferWriteOp>(
        loc, undistrWrite, alloc, indices, inBounds);
    // Set layouts signature for write.
    // We need to set the layout on the srcVector/first operand.
    auto unitAttr = UnitAttr::get(rewriter.getContext());
    {
      SmallVector<int64_t> subgroupTileLens =
          llvm::to_vector(srcLayout.getSubgroupTile());
      SmallVector<int64_t> batchTileLens =
          llvm::to_vector(srcLayout.getBatchTile());
      SmallVector<int64_t> outerTileLens =
          llvm::to_vector(srcLayout.getOuterTile());
      SmallVector<int64_t> threadTileLens =
          llvm::to_vector(srcLayout.getThreadTile());
      SmallVector<int64_t> elementTileLens =
          llvm::to_vector(srcLayout.getElementTile());
      SmallVector<int64_t> subgroupStrides =
          llvm::to_vector(srcLayout.getSubgroupStrides());
      SmallVector<int64_t> threadStrides =
          llvm::to_vector(srcLayout.getThreadStrides());
      // Replace the reduced tiles with unit dimension.
      for (int64_t rDim : reductionDims) {
        batchTileLens[rDim] = 1;
        outerTileLens[rDim] = 1;
        threadTileLens[rDim] = 1;
        elementTileLens[rDim] = 1;
        threadStrides[rDim] = 0;
      }
      auto interSubGroupLayout = IREE::VectorExt::NestedLayoutAttr::get(
          rewriter.getContext(), subgroupTileLens, batchTileLens, outerTileLens,
          threadTileLens, elementTileLens, subgroupStrides, threadStrides);
      auto writeAttrs =
          SmallVector<Attribute>(write->getNumOperands(), unitAttr);
      writeAttrs[0] = interSubGroupLayout;
      ArrayAttr writeOperandsAttr =
          ArrayAttr::get(rewriter.getContext(), writeAttrs);
      ArrayAttr writeResultsAttr = ArrayAttr::get(rewriter.getContext(), {});
      setSignatureForRedistribution(rewriter, write.getOperation(),
                                    writeOperandsAttr, writeResultsAttr);
    }
    // Insert gpu.barrier
    rewriter.create<gpu::BarrierOp>(write.getLoc());
    auto read = rewriter.create<vector::TransferReadOp>(
        loc, unDistributedType, alloc, indices, inBounds);
    // Create new layout where subgroup dims are squashed to
    // element tile
    IREE::VectorExt::NestedLayoutAttr intraSubGroupLayout;
    {
      // We intentionally make the subgroup tile to be 1
      SmallVector<int64_t> subgroupTileLens =
          llvm::to_vector(srcLayout.getSubgroupTile());
      SmallVector<int64_t> batchTileLens =
          llvm::to_vector(srcLayout.getBatchTile());
      SmallVector<int64_t> outerTileLens =
          llvm::to_vector(srcLayout.getOuterTile());
      SmallVector<int64_t> threadTileLens =
          llvm::to_vector(srcLayout.getThreadTile());
      SmallVector<int64_t> elementTileLens =
          llvm::to_vector(srcLayout.getElementTile());
      SmallVector<int64_t> subgroupStrides =
          llvm::to_vector(srcLayout.getSubgroupStrides());
      SmallVector<int64_t> threadStrides =
          llvm::to_vector(srcLayout.getThreadStrides());
      for (int64_t rDim : reductionDims) {
        subgroupTileLens[rDim] = 1;
        batchTileLens[rDim] = 1;
        outerTileLens[rDim] = 1;
        threadTileLens[rDim] = 1;
        // the partial reductions that was across subgroups will
        // will be loaded as element tile. We can revisit if this
        // need to be something else such as thread tile.
        elementTileLens[rDim] = srcLayout.getSubgroupTile()[rDim];
        subgroupStrides[rDim] = 0;
        threadStrides[rDim] = 0;
      }
      intraSubGroupLayout = IREE::VectorExt::NestedLayoutAttr::get(
          rewriter.getContext(), subgroupTileLens, batchTileLens, outerTileLens,
          threadTileLens, elementTileLens, subgroupStrides, threadStrides);
      auto readAttrs = SmallVector<Attribute>(read->getNumOperands(), unitAttr);
      ArrayAttr readOperandsAttr =
          ArrayAttr::get(rewriter.getContext(), readAttrs);
      ArrayAttr readResultsAttr =
          ArrayAttr::get(rewriter.getContext(), {intraSubGroupLayout});
      setSignatureForRedistribution(rewriter, read.getOperation(),
                                    readOperandsAttr, readResultsAttr);
    }

    // A newly created reduction to complete the reduction
    // that reduces the data that was otherwise was on
    // different subgroups.
    auto secondReduction = rewriter.create<vector::MultiDimReductionOp>(
        loc, kind, read, acc, reductionDims);
    {
      auto reduceAttrs =
          SmallVector<Attribute>(secondReduction->getNumOperands(), unitAttr);
      reduceAttrs[0] = intraSubGroupLayout;
      ArrayAttr reduceResultsAttr =
          ArrayAttr::get(rewriter.getContext(), {unitAttr});
      if (auto dstLayout = dyn_cast_or_null<NestedLayoutAttr>(resLayout)) {
        reduceAttrs[1] = dstLayout;
        reduceResultsAttr = ArrayAttr::get(rewriter.getContext(), {dstLayout});
      }
      ArrayAttr reduceOperandsAttr =
          ArrayAttr::get(rewriter.getContext(), reduceAttrs);
      setSignatureForRedistribution(rewriter, secondReduction.getOperation(),
                                    reduceOperandsAttr, reduceResultsAttr);
    }
    return secondReduction.getResult();
  }

  int64_t subgroupSize;
  int64_t maxBitsPerShuffle;
};

/// The distribution of contract is performed by doing a local contraction where
/// each thread performs operations on its locally distributed elements. Then,
/// the resulting vector is interpreted in undistributed domain. The said
/// undistributed vector is a partial reduction when contraction has been
/// performed only thread locally. Therefore, a to-be-distributed
/// vector.multi_reduce
////is added to complete the contraction.
struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeContract(MLIRContext *context, int64_t benefit = 1)
      : OpDistributionPattern(context, benefit) {}

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    FailureOr<VectorContractOpInfo> maybeOpInfo =
        VectorContractOpInfo::inferFromIndexingMaps(
            contractOp.getIndexingMapsArray());
    if (failed(maybeOpInfo)) {
      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
    }
    // If mmaAttr exists, defer the lowering to use MMA.
    // Notify failure if the "iree.amdgpu.mma" intrinsic attribute is present.
    auto mmaAttr = contractOp->getAttrOfType<IREE::GPU::MmaInterfaceAttr>(
        "iree.amdgpu.mma");
    if (mmaAttr) {
      return rewriter.notifyMatchFailure(
          contractOp, "iree.amdgpu.mma intrinsic attribute exists");
    }

    auto lhsLayout = dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
    if (!lhsLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing nested layout for contraction lhs");
    }
    auto rhsLayout = dyn_cast<NestedLayoutAttr>(signature[contractOp.getRhs()]);
    if (!rhsLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing nested layout for contraction rhs");
    }
    NestedLayoutAttr resLayout;
    if (auto contractRes = dyn_cast<VectorValue>(contractOp.getResult())) {
      resLayout = dyn_cast<NestedLayoutAttr>(signature[contractRes]);
    } else {
      // Create a zero-d layout because we
      // are going to add reduction dims
      // back to handle the partial reduction
      resLayout = NestedLayoutAttr::get(
          contractOp.getContext(), ArrayRef<int64_t>{}, {}, {}, {}, {}, {}, {});
    }

    Value disLhs = getDistributed(rewriter, contractOp.getLhs(), lhsLayout);
    Value disRhs = getDistributed(rewriter, contractOp.getRhs(), rhsLayout);

    Value acc = contractOp.getAcc();
    Value res = contractOp.getResult();
    auto accVector = dyn_cast<VectorValue>(acc);
    auto resVector = dyn_cast<VectorValue>(res);
    Value disAcc;
    if (accVector) {
      disAcc = getDistributed(rewriter, accVector, signature[accVector]);
    } else {
      disAcc = contractOp.getAcc();
    }

    Type accElemTy = getElementTypeOrSelf(acc.getType());

    MLIRContext *ctx = contractOp.getContext();
    Location loc = contractOp.getLoc();

    // Step 1: local contraction
    Value localInit = getCombiningIdentityValue(
        loc, rewriter, contractOp.getKind(), disAcc.getType());
    vector::ContractionOp localContractOp = doDistributedContraction(
        rewriter, loc, ctx, contractOp, disLhs, disRhs, localInit);

    VectorValue localContractValue;
    if (accVector) {
      localContractValue = dyn_cast<VectorValue>(localContractOp.getResult());
    } else {
      VectorType vecType = VectorType::get(ArrayRef{int64_t(1)}, accElemTy);
      localContractValue = rewriter.create<vector::BroadcastOp>(
          loc, vecType, localContractOp.getResult());
    }

    assert(localContractValue && "result should have been a vector");

    // Identify the reduction dimension and apply it for subgroup reduction.
    auto lhsMap = contractOp.getIndexingMapsArray()[0];
    SmallVector<int64_t> reductionSubGroupTile;
    SmallVector<int64_t> reductionSubGroupStrides;
    SmallVector<int64_t> reductionThreadTile;
    SmallVector<int64_t> reductionThreadStrides;
    SmallVector<int64_t> partialReductionDims;
    for (auto [index, iteratorType] :
         llvm::enumerate(contractOp.getIteratorTypes())) {
      if (vector::isReductionIterator(iteratorType)) {
        int64_t redLhsIdx =
            *(lhsMap.getResultPosition(getAffineDimExpr(index, ctx)));
        partialReductionDims.push_back(resLayout.getRank() +
                                       reductionSubGroupTile.size());
        reductionSubGroupTile.push_back(lhsLayout.getSubgroupTile()[redLhsIdx]);
        reductionSubGroupStrides.push_back(
            lhsLayout.getSubgroupStrides()[redLhsIdx]);
        reductionThreadTile.push_back(lhsLayout.getThreadTile()[redLhsIdx]);
        reductionThreadStrides.push_back(
            lhsLayout.getThreadStrides()[redLhsIdx]);
      }
    }
    SmallVector<int64_t> unitBroadcastTile(reductionThreadTile.size(), 1);

    // Manually infer the layout of partial reduction
    // We do this by appending the reduction dims on
    // subgroup and thread tiles to the layout of the
    // result.
    IREE::VectorExt::NestedLayoutAttr reductionLayout =
        IREE::VectorExt::NestedLayoutAttr::get(
            contractOp.getContext(),
            /*source=*/resLayout,
            /*appendSubGroupLens=*/reductionSubGroupTile,
            /*appendBatchLens=*/unitBroadcastTile,
            /*appendOuterLens=*/unitBroadcastTile,
            /*appendThreadLens=*/reductionThreadTile,
            /*appendElementLens=*/unitBroadcastTile,
            /*appendSubgroupStrides=*/reductionSubGroupStrides,
            /*appendThreadStrides=*/reductionThreadStrides);

    VectorType partialReducedDistributedType =
        VectorType::get(reductionLayout.getDistributedShape(),
                        localContractValue.getType().getElementType());
    Value shapeCasted = rewriter.create<vector::ShapeCastOp>(
        loc, partialReducedDistributedType, localContractValue);
    VectorType unDistributedType =
        VectorType::get(reductionLayout.getUndistributedShape(),
                        localContractValue.getType().getElementType());
    Value undistrLocalReduced = rewriter.create<IREE::VectorExt::ToSIMDOp>(
        loc, unDistributedType, shapeCasted);

    // Create the partial reduction
    auto partialReduction = rewriter.create<vector::MultiDimReductionOp>(
        loc, contractOp.getKind(), undistrLocalReduced, acc,
        partialReductionDims);
    {
      auto unitAttr = UnitAttr::get(rewriter.getContext());
      auto reduceAttrs =
          SmallVector<Attribute>(partialReduction->getNumOperands(), unitAttr);
      reduceAttrs[0] = reductionLayout;
      ArrayAttr reduceResultsAttr =
          ArrayAttr::get(rewriter.getContext(), {unitAttr});
      if (auto dstLayout =
              dyn_cast_or_null<NestedLayoutAttr>(signature[resVector])) {
        reduceAttrs[1] = dstLayout;
        reduceResultsAttr = ArrayAttr::get(rewriter.getContext(), {dstLayout});
      }
      ArrayAttr reduceOperandsAttr =
          ArrayAttr::get(rewriter.getContext(), reduceAttrs);
      setSignatureForRedistribution(rewriter, partialReduction.getOperation(),
                                    reduceOperandsAttr, reduceResultsAttr);
    }
    rewriter.replaceOp(contractOp, partialReduction);
    return success();
  }

  vector::ContractionOp
  doDistributedContraction(RewriterBase &rewriter, Location loc,
                           MLIRContext *ctx, vector::ContractionOp contractOp,
                           Value lhs, Value rhs, Value acc) const {
    SmallVector<AffineMap> maps = contractOp.getIndexingMapsArray();
    ArrayRef<Attribute> iteratorTypes =
        contractOp.getIteratorTypes().getValue();

    // Given that the distribution format is <BATCH x OUTER x ELEMENT>,
    // the iterations and affine maps need to be replicated three times.

    SmallVector<Attribute> newIterators;
    // Replicate the iterators for local vector.contract
    for (int i = 0; i < 3; ++i) {
      newIterators.append(iteratorTypes.begin(), iteratorTypes.end());
    }

    // Replicate the affine maps for local vector.contract
    SmallVector<AffineMap> newMaps;
    for (AffineMap map : maps) {
      int64_t numDims = map.getNumDims();
      int64_t numResults = map.getNumResults();
      SmallVector<AffineExpr> exprs;
      for (int i = 0; i < 3; ++i) {
        AffineMap shiftedMap = map.shiftDims(i * numDims);
        for (int j = 0; j < numResults; ++j) {
          exprs.push_back(shiftedMap.getResult(j));
        }
      }
      AffineMap newMap =
          AffineMap::get(/*dimCount=*/3 * numDims,
                         /*symbolCount=*/map.getNumSymbols(), exprs, ctx);
      newMaps.push_back(newMap);
    }

    Value localInit = getCombiningIdentityValue(
        loc, rewriter, contractOp.getKind(), acc.getType());

    auto localContractOp = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, localInit, rewriter.getAffineMapArrayAttr(newMaps),
        rewriter.getArrayAttr(newIterators), contractOp.getKind());
    localContractOp->setDiscardableAttrs(
        contractOp->getDiscardableAttrDictionary());

    return localContractOp;
  }
};

struct DistributeTranspose final : OpDistributionPattern<vector::TransposeOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue value = transposeOp.getVector();
    VectorLayoutInterface layout = dyn_cast<NestedLayoutAttr>(signature[value]);
    if (!layout) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "layout must be NestedLayoutAttr");
    }

    /// Transpose only changes the notion of where the data carried by each
    /// thread comes from in the transposed SIMD vector. The data carried by
    /// each thread is still the same, transposed as requested by the operation.
    /// So, for distributed dimensions (thread and subgroup) transpose is a
    /// no-op.
    ///
    /// Example (indices [0-3] represent ids of the threads carrying the data):
    ///
    /// input: vector<2x4xf16>
    ///
    /// 0 0 1 1
    /// 2 2 3 3
    ///
    /// after transpose,
    ///
    /// transp: vector<4x2xf16>
    ///
    /// 0 2
    /// 0 2
    /// 1 3
    /// 1 3
    ///
    /// As it can be seen, each thread is still carrying the same data but
    /// just holds a transposed version of it.

    VectorValue input = getDistributed(rewriter, value, layout);
    // Permute batch, outer and element based on the given permutation.
    int64_t rank = value.getType().getRank();
    SmallVector<int64_t> permutation;
    for (int i = 0; i < 3; ++i) {
      for (auto it : transposeOp.getPermutation()) {
        permutation.push_back(it + (i * rank));
      }
    }
    VectorValue transposed = rewriter.create<vector::TransposeOp>(
        transposeOp.getLoc(), input, permutation);
    replaceOpWithDistributedValues(rewriter, transposeOp, transposed);
    return success();
  }
};

struct DistributeBatchOuterToLayoutConversions final
    : OpDistributionPattern<IREE::VectorExt::ToLayoutOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(IREE::VectorExt::ToLayoutOp toLayoutOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Location loc = toLayoutOp.getLoc();
    auto input = cast<VectorValue>(toLayoutOp.getInput());
    auto output = cast<VectorValue>(toLayoutOp.getOutput());
    auto layoutA = dyn_cast<NestedLayoutAttr>(signature[input]);
    auto layoutB = dyn_cast<NestedLayoutAttr>(signature[output]);

    if (!layoutA || !layoutB) {
      return rewriter.notifyMatchFailure(toLayoutOp, "non-nested layout");
    }

    // Check if everything other than batch and outer tile matches.
    if (layoutA.getSubgroupTile() != layoutB.getSubgroupTile()) {
      return failure();
    }
    if (layoutA.getSubgroupStrides() != layoutB.getSubgroupStrides()) {
      return failure();
    }
    if (layoutA.getThreadTile() != layoutB.getThreadTile()) {
      return failure();
    }
    if (layoutA.getThreadStrides() != layoutB.getThreadStrides()) {
      return failure();
    }
    if (layoutA.getElementTile() != layoutB.getElementTile()) {
      return failure();
    }

    auto batchTileA = SmallVector<int64_t>(layoutA.getBatchTile());
    auto outerTileA = SmallVector<int64_t>(layoutA.getOuterTile());
    auto batchTileB = SmallVector<int64_t>(layoutB.getBatchTile());
    auto outerTileB = SmallVector<int64_t>(layoutB.getOuterTile());

    // Check if there is a batch/outer tile mismatch.
    if (batchTileA == batchTileB && outerTileA == outerTileB) {
      return rewriter.notifyMatchFailure(toLayoutOp,
                                         "trivial layout conversion");
    }

    SmallVector<int64_t> shapeA = layoutA.getDistributedShape();
    SmallVector<int64_t> shapeB = layoutB.getDistributedShape();
    int64_t rank = layoutA.getRank();

    // Interleave batch and outer dims by transposing.

    // Build a permutation for interleaving.
    auto interleavePermutation =
        llvm::to_vector(llvm::seq<int64_t>(shapeA.size()));
    for (int i = 0; i < rank; ++i) {
      // Batch tile : [0...rank]
      // OuterTile : [rank+1...2*rank]
      // Interleave : [batch0, outer0, batch1, outer1,...]
      interleavePermutation[2 * i] = i;
      interleavePermutation[2 * i + 1] = i + rank;
    }

    auto interleaved = rewriter.create<vector::TransposeOp>(
        loc, getDistributed(rewriter, input, layoutA), interleavePermutation);

    // Shape cast to match the new layout.

    SmallVector<int64_t> transposedShapeB(shapeB);
    applyPermutationToVector(transposedShapeB, interleavePermutation);
    Type reshapedType = VectorType::get(
        transposedShapeB, interleaved.getResultVectorType().getElementType());

    auto reshaped =
        rewriter.create<vector::ShapeCastOp>(loc, reshapedType, interleaved);

    // Inverse transpose to preserve original order.
    SmallVector<int64_t> invertedPermutation =
        invertPermutationVector(interleavePermutation);

    auto layouted = rewriter.create<vector::TransposeOp>(loc, reshaped,
                                                         invertedPermutation);

    replaceOpWithDistributedValues(rewriter, toLayoutOp, layouted.getResult());
    return success();
  }
};

struct DistributeStep final : OpDistributionPattern<vector::StepOp> {
  using OpDistributionPattern::OpDistributionPattern;

  // This is a helper aggregate
  // to hold the information about
  // a dimension.
  // For e.g. : 3x4x2 shape will
  // have lengths = [3, 4, 2]
  // and strides = [8, 2, 1]
  struct DimInfo {
    std::optional<Value> dimIdx;
    int64_t dimLen;
    int64_t dimStride;
  };

  // This is a helper function to extract the remaining
  // dimensions with their original strides once the
  // distributed dimensions are extracted out
  //         threads
  //          V
  // E.g. 3 x 4 x 2
  // This will return back remaining dimensions that
  // have lengths = [3, 2] and strides = [8, 1]
  SmallVector<DimInfo> getRemainingDims(ArrayRef<DimInfo> distributedStrides,
                                        int64_t originalLen) const {
    SmallVector<DimInfo> remainingDims;
    int64_t currLen = originalLen;
    for (const DimInfo &dInfo : distributedStrides) {
      if (dInfo.dimStride != 0) {
        int64_t dStride = dInfo.dimStride;
        int64_t dLen = dInfo.dimLen;
        int64_t higherStride = dLen * dStride;
        if (higherStride < currLen) {
          remainingDims.push_back(
              {std::nullopt, currLen / higherStride, higherStride});
        }
        currLen = dStride;
      }
    }
    remainingDims.push_back({std::nullopt, currLen, 1});
    return remainingDims;
  }

  // This is a helper to extract lengths of all dimensions
  SmallVector<int64_t> getLens(ArrayRef<DimInfo> dimInfos) const {
    SmallVector<int64_t> lens;
    lens.reserve(dimInfos.size());
    for (const DimInfo &dInfo : dimInfos) {
      lens.push_back(dInfo.dimLen);
    }
    return lens;
  }

  // This is a helper to extract strides from a given shape
  // E.g. : a shape of 2x3x4 will return strides [12, 4, 1]
  SmallVector<int64_t> getStrides(ArrayRef<int64_t> shape) const {
    int64_t elementCount = ShapedType::getNumElements(shape);
    SmallVector<int64_t> strides;
    int64_t currStride = elementCount;
    for (int64_t len : shape) {
      currStride = currStride / len;
      strides.push_back(currStride);
    }
    return strides;
  }

  // Once we are in the realm of remaining dimensions,
  // the strides are not packed. This is a helper to
  // obtain the packed strides of the remaining dimensions.
  // (See above for an example of remaining dimensions under
  //  getRemainingDims)
  SmallVector<int64_t> getPackedStrides(ArrayRef<DimInfo> dims) const {
    SmallVector<int64_t> lens = getLens(dims);
    return getStrides(lens);
  }

  // This function emulates the slicing of otherwise large constant
  // across threads and subgroups.
  VectorValue generateSlicedStep(OpBuilder &builder, Location loc,
                                 ArrayRef<DimInfo> distributedDims,
                                 int64_t distributedLen,
                                 int64_t originalLen) const {
    SmallVector<DimInfo> remainingDims =
        getRemainingDims(distributedDims, originalLen);
    SmallVector<int64_t> remainingPackedStrides =
        getPackedStrides(remainingDims);

    SmallVector<APInt> offsets;
    offsets.reserve(distributedLen);
    // As for a complex example what the following
    // maths would achieve:
    //    wave
    //     |   threads
    //     V   V
    // 2 x 3 x 4 x 2 = 0 1 2 .... 48
    // say vector.step : vector<48xindex> is to be distributed.
    // --------------------------------------------------------
    // The the distribution should be as follows:
    // wave0:
    // t0: 0 1 24 25
    // t1: 2 3 26 27
    // t2: 4 5 28 29
    // t4: 6 7 30 31
    //
    // wave1:
    // t0: 8 9 32 33
    // t1: 10 11 34 35
    // t2: 12 13 36 37
    // t4: 14 15 38 39
    // ... etc
    //
    // So wave0 & t0 value this constant offset that we generate
    // below initially. Then followed by thread and subgroup weighted
    // addition that is weighted by their stride.
    for (size_t i = 0; i < distributedLen; i++) {
      int64_t offset = 0;
      for (const auto &[dimInfo, packedStride] :
           zip(remainingDims, remainingPackedStrides)) {
        offset += ((i / packedStride) % dimInfo.dimLen) * dimInfo.dimStride;
      }
      offsets.push_back(APInt(/*width=*/64, offset));
    }
    VectorType offsetType =
        VectorType::get({distributedLen}, builder.getIndexType());
    auto constOffset = builder.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(offsetType, offsets));
    Value finalOffset = constOffset;
    for (const DimInfo &dimInfo : distributedDims) {
      assert(dimInfo.dimIdx.has_value());
      if (dimInfo.dimStride != 0) {
        auto strideVal =
            builder.create<arith::ConstantIndexOp>(loc, dimInfo.dimStride);
        auto dimIdxOffsetPerElem = builder.create<arith::MulIOp>(
            loc, strideVal, dimInfo.dimIdx.value());
        auto dimIdxOffset = builder.create<vector::BroadcastOp>(
            loc, offsetType, dimIdxOffsetPerElem);
        finalOffset =
            builder.create<arith::AddIOp>(loc, finalOffset, dimIdxOffset);
      }
    }
    return cast<VectorValue>(finalOffset);
  }

  DistributeStep(MLIRContext *context, Value threadId, int64_t subgroupSize)
      : OpDistributionPattern(context), threadId(threadId),
        subgroupSize(subgroupSize) {}
  LogicalResult matchAndRewrite(vector::StepOp stepOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Location loc = stepOp.getLoc();
    VectorValue result = stepOp.getResult();
    NestedLayoutAttr resultLayout =
        dyn_cast<NestedLayoutAttr>(signature[result]);
    if (!resultLayout) {
      return rewriter.notifyMatchFailure(
          stepOp, "missing nested layout for step op result");
    }
    SmallVector<Value> subgroupIndices, threadIndices;
    if (failed(populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
                                            resultLayout, subgroupIndices,
                                            threadIndices))) {
      return rewriter.notifyMatchFailure(
          stepOp, "warp or thread tiles have overlapping strides");
    }

    SmallVector<int64_t> undistributedShape =
        resultLayout.getUndistributedPackedShape();
    SmallVector<int64_t> undistributedStrides = getStrides(undistributedShape);
    constexpr int64_t subgroupIdx = 0;
    constexpr int64_t threadIdx = 3;

    ArrayRef<int64_t> subgroupLengths = resultLayout.getSubgroupTile();
    ArrayRef<int64_t> threadLengths = resultLayout.getThreadTile();
    // Step op by definition should be single dimensional.
    SmallVector<int64_t> distributedShape =
        signature[result].getDistributedShape();

    int64_t distributedElements = ShapedType::getNumElements(distributedShape);
    int64_t originalElements = result.getType().getNumElements();
    SmallVector<DimInfo, 2> distributedDims{
        {subgroupIndices[0], subgroupLengths[0],
         undistributedStrides[subgroupIdx]},
        {threadIndices[0], threadLengths[0], undistributedStrides[threadIdx]}};
    llvm::sort(distributedDims, [](const DimInfo &lhs, const DimInfo &rhs) {
      return lhs.dimStride > rhs.dimStride;
    });
    VectorValue slicedStepOp = generateSlicedStep(
        rewriter, loc, distributedDims, distributedElements, originalElements);
    VectorType finalSlicedStepOpType =
        VectorType::get({distributedShape}, result.getType().getElementType());
    auto finalSlicedStepOp = rewriter.create<vector::ShapeCastOp>(
        loc, finalSlicedStepOpType, slicedStepOp);
    replaceOpWithDistributedValues(rewriter, stepOp, {finalSlicedStepOp});
    return success();
  }

  Value threadId;
  int64_t subgroupSize;
};

} // namespace

void populateGPUDistributeNestedLayoutAttrPatterns(RewritePatternSet &patterns,
                                                   Value threadId,
                                                   int64_t subgroupSize,
                                                   int64_t maxBitsPerShuffle) {
  patterns.add<DistributeTransferRead, DistributeTransferWrite>(
      patterns.getContext(), threadId, subgroupSize);
  patterns.add<DistributeBroadcast, DistributeTranspose>(patterns.getContext());
  patterns.add<DistributeMultiReduction>(patterns.getContext(), subgroupSize,
                                         maxBitsPerShuffle);
  patterns.add<DistributeContract>(patterns.getContext());
  patterns.add<DistributeBatchOuterToLayoutConversions>(patterns.getContext());
  patterns.add<DistributeStep>(patterns.getContext(), threadId, subgroupSize);
}

}; // namespace mlir::iree_compiler
