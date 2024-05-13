// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <numeric>
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
static void populateWarpAndThreadIndices(RewriterBase &rewriter, Value threadId,
                                         NestedLayoutAttr vectorLayout,
                                         SmallVector<Value> &warpIndices,
                                         SmallVector<Value> &threadIndices) {
  int64_t subgroupRank = vectorLayout.getSubgroupBasis().size();
  // The delinearized thread IDs are returned from outer most to inner most,
  // i.e. before applying the layout described dimensions ordering.
  SmallVector<Value> threadIds =
      vectorLayout.computeThreadIds(threadId, rewriter);

  SmallVector<Value> filteredSubgroupIds;
  for (auto [id, active] :
       llvm::zip(threadIds, vectorLayout.getSubgroupActiveIds())) {
    if (active)
      filteredSubgroupIds.push_back(id);
  }
  SmallVector<Value> filteredThreadIds;
  for (auto [id, active] : llvm::zip(llvm::drop_begin(threadIds, subgroupRank),
                                     vectorLayout.getThreadActiveIds())) {
    if (active)
      filteredThreadIds.push_back(id);
  }

  // Subgroup and thread (lane) indices normalized to the order in which
  // they are used by each dimension.
  warpIndices = llvm::to_vector(
      llvm::map_range(invertPermutationVector(vectorLayout.getSubgroupOrder()),
                      [&](int64_t i) { return filteredSubgroupIds[i]; }));
  threadIndices = llvm::to_vector(
      llvm::map_range(invertPermutationVector(vectorLayout.getThreadOrder()),
                      [&](int64_t i) { return filteredThreadIds[i]; }));
}

namespace {

/// Pattern to distribute `vector.transfer_read` ops with nested layouts.
struct DistributeTransferRead final
    : OpDistributionPattern<vector::TransferReadOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferRead(MLIRContext *context, Value threadId)
      : OpDistributionPattern(context), threadId(threadId) {}

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
         StaticTileOffsetRange(distShape, tileShape)) {
      SmallVector<Value> slicedIndices = getTransferIndicesFromNestedLayout(
          rewriter, indices, offsets, vectorLayout, readOp.getPermutationMap(),
          warpIndices, threadIndices);

      Value slicedRead = rewriter.create<vector::TransferReadOp>(
          readOp.getLoc(), innerVectorType, readOp.getSource(), slicedIndices,
          readOp.getPermutationMapAttr(), readOp.getPadding(), readOp.getMask(),
          readOp.getInBoundsAttr());

      acc = rewriter.create<vector::InsertStridedSliceOp>(
          readOp.getLoc(), slicedRead, acc, offsets, strides);
    }

    replaceOpWithDistributedValues(rewriter, readOp, acc);
    return success();
  }

  Value threadId;
};

/// Pattern to distribute `vector.transfer_write` ops with nested layouts.
struct DistributeTransferWrite final
    : OpDistributionPattern<vector::TransferWriteOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferWrite(MLIRContext *context, Value threadId)
      : OpDistributionPattern(context), threadId(threadId) {}

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
    populateWarpAndThreadIndices(rewriter, threadId, vectorLayout, warpIndices,
                                 threadIndices);

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
        VectorType::get(vectorLayout.getElementsPerThread(), elementType);

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
  // Get strides for dimensions based on layouts.
  SmallVector<int64_t> threadBasis(layout.getThreadBasis());
  SmallVector<int64_t> basisStrides(threadBasis.size());
  // Take prefix sum to get strides.
  std::exclusive_scan(threadBasis.rbegin(), threadBasis.rend(),
                      basisStrides.rbegin(), 1, std::multiplies<>{});
  // Remove non-active thread ids.
  SmallVector<int64_t> activeThreadStrides;
  for (auto [i, stride] : llvm::enumerate(basisStrides)) {
    if (layout.getThreadActiveIds()[i]) {
      activeThreadStrides.push_back(stride);
    }
  }
  // TODO: Do we need to do inversion or not?
  return activeThreadStrides[layout.getThreadOrder()[dim]];
}

static int64_t getShuffleWidth(NestedLayoutAttr layout, int64_t dim) {
  return layout.getThreadsPerOuter()[dim];
}

/// The lowering for multi_reduction is done in two steps:
///   1. Local Reduce: Each thread reduces all elements carried by it along
///      the reduction dimensions. This is the batch, outer and element dims.
///   2. Thread Reduce: Each thread reduces result of step 1 across threads
///      by doing a butterfly shuffle.
///
/// Currently, reduction across warps is not supported, but it would just add
/// another step, Warp Reduce, where threads do an atomic addition on a buffer.
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
    auto accVector = dyn_cast<VectorValue>(multiReduceOp.getAcc());
    if (!accVector) {
      return rewriter.notifyMatchFailure(
          multiReduceOp, "unimplemented: scalar accumulator distribution");
    }
    auto resVector = dyn_cast<VectorValue>(multiReduceOp.getResult());
    if (!resVector) {
      return rewriter.notifyMatchFailure(
          multiReduceOp, "unimplemented: scalar result distribution");
    }

    auto srcLayout = dyn_cast_or_null<NestedLayoutAttr>(signature[srcVector]);
    if (!srcLayout) {
      return rewriter.notifyMatchFailure(multiReduceOp,
                                         "expected nested layout attr");
    }

    Type elemTy = srcVector.getType().getElementType();
    unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
    if (elemBitwidth != maxBitsPerShuffle) {
      return rewriter.notifyMatchFailure(
          multiReduceOp, llvm::formatv("unimplemented: packed shuffle",
                                       elemBitwidth, maxBitsPerShuffle));
    }

    VectorValue disSrc =
        getDistributed(rewriter, srcVector, signature[srcVector]);
    VectorValue disAcc =
        getDistributed(rewriter, accVector, signature[accVector]);

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

    auto localReduction = rewriter.create<vector::MultiDimReductionOp>(
        loc, disSrc, disAcc, distributedReductionMask, multiReduceOp.getKind());
    auto locallyReduced = dyn_cast<VectorValue>(localReduction.getResult());

    assert(locallyReduced && "result should have been a vector");

    // Flatten the locally reduced value.
    VectorType shaped = locallyReduced.getType();
    int64_t numElements = shaped.getNumElements();
    SmallVector<int64_t> flatShape(1, numElements);
    VectorType flatVecType = VectorType::get(flatShape, elemTy);
    VectorValue flat =
        rewriter.create<vector::ShapeCastOp>(loc, flatVecType, locallyReduced);

    FailureOr<VectorValue> threadReduced = doThreadReduction(
        rewriter, srcLayout, flat, multiReduceOp.getKind(), reducedDims);
    if (failed(threadReduced)) {
      return failure();
    }

    VectorValue unflattened = rewriter.create<vector::ShapeCastOp>(
        loc, shaped, threadReduced.value());
    replaceOpWithDistributedValues(rewriter, multiReduceOp, unflattened);

    return failure();
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
      for (unsigned i = 0; i < reductionMask.size(); ++i) {
        if (reductionMask[i]) {
          extracted = doPackedThreadReductionOnDim(rewriter, layout, extracted,
                                                   kind, i);
        }
      }

      res = rewriter.create<vector::InsertOp>(loc, extracted, res, i);
    }

    return res;
  }

  Value doPackedThreadReductionOnDim(RewriterBase &rewriter,
                                     NestedLayoutAttr layout, Value val,
                                     vector::CombiningKind kind,
                                     int64_t dim) const {
    Location loc = val.getLoc();
    int64_t offset = getShuffleOffset(layout, dim);
    int64_t width = getShuffleWidth(layout, dim);

    for (int i = offset; i < offset * width; i <<= 1) {
      auto shuffleOp = rewriter.create<gpu::ShuffleOp>(
          loc, val, i, subgroupSize, gpu::ShuffleMode::XOR);
      val =
          makeArithReduction(rewriter, loc, kind, shuffleOp.getShuffleResult(),
                             val, nullptr, nullptr);
    }

    return val;
  }

  int64_t subgroupSize;
  int64_t maxBitsPerShuffle;
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

} // namespace

void populateGPUDistributeNestedLayoutAttrPatterns(RewritePatternSet &patterns,
                                                   Value threadId,
                                                   int64_t subgroupSize,
                                                   int64_t maxBitsPerShuffle) {
  patterns.add<DistributeTransferRead, DistributeTransferWrite>(
      patterns.getContext(), threadId);
  patterns.add<DistributeBroadcast, DistributeTranspose>(patterns.getContext());
  patterns.add<DistributeMultiReduction>(patterns.getContext(), subgroupSize,
                                         maxBitsPerShuffle);
}

}; // namespace mlir::iree_compiler
