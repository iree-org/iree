// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

    // The delinearized thread IDs are returned from outer most to inner most,
    // i.e. before applying the layout described dimensions ordering.
    ValueRange threadIds = vectorLayout.computeThreadIds(threadId, rewriter);

    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    SmallVector<int64_t> tileShape = vectorLayout.getDistributedShape();
    int64_t rank = vectorLayout.getBatchOrder().size();

    // Let the unroll order first unroll the batch dimensions, then the
    // outer vector dimensions. We unroll in the order specified by the
    // layout.
    SmallVector<int64_t> loopOrder;
    int64_t base = 0;
    for (int64_t b : vectorLayout.getBatchOrder()) {
      loopOrder.push_back(base + b);
      tileShape[base + b] = 1;
    }
    base += rank;
    // We must unroll along the outer dimensions as well to match the rank
    // requirements of vector transfer ops (<= memref rank up to broadcasts).
    for (int64_t o : vectorLayout.getOuterOrder()) {
      loopOrder.push_back(base + o);
      tileShape[base + o] = 1;
    }
    base += rank;
    for (int i = 0, e = rank; i < e; ++i) {
      loopOrder.push_back(base + i);
    }

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

    // Subgroup and thread (lane) indices normalized to the order in which
    // they are used by each dimension.
    SmallVector<Value> warpIndices = llvm::to_vector(
        llvm::map_range(vectorLayout.getSubgroupOrder(),
                        [&](int64_t i) { return threadIds[i]; }));
    SmallVector<Value> threadIndices = llvm::to_vector(
        llvm::map_range(vectorLayout.getThreadOrder(),
                        [&](int64_t i) { return threadIds[i + rank]; }));

    auto isBroadcast = [](AffineExpr expr) {
      if (auto constExpr = dyn_cast<AffineConstantExpr>(expr))
        return constExpr.getValue() == 0;
      return false;
    };

    ValueRange indices = readOp.getIndices();
    SmallVector<int64_t> strides(rank, 1);
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distShape, tileShape, loopOrder)) {
      // Permute the batch and outer vector offsets to match the order of
      // the vector dimensions using the inverse of the batch/offset order.
      SmallVector<int64_t> batchOffsets = applyPermutation(
          ArrayRef<int64_t>(offsets.begin(), rank),
          invertPermutationVector(vectorLayout.getBatchOrder()));
      SmallVector<int64_t> outerVectorOffsets = applyPermutation(
          ArrayRef<int64_t>(offsets.begin() + rank, rank),
          invertPermutationVector(vectorLayout.getOuterOrder()));

      SmallVector<Value> slicedIndices(indices.begin(), indices.end());
      for (const auto &[i, dim] :
           llvm::enumerate(readOp.getPermutationMap().getResults())) {
        if (isBroadcast(dim))
          continue;
        unsigned pos = cast<AffineDimExpr>(dim).getPosition();
        SmallVector<OpFoldResult> ids = {
            warpIndices[i], rewriter.getIndexAttr(batchOffsets[i]),
            rewriter.getIndexAttr(outerVectorOffsets[i]), threadIndices[i]};
        // The order in which a vector dimension is "tiled" is
        // subgroups -> batches -> outer vectors -> threads -> elements
        SmallVector<int64_t> sizes = {
            vectorLayout.getSubgroupsPerWorkgroup()[i],
            vectorLayout.getBatchesPerSubgroup()[i],
            vectorLayout.getOutersPerBatch()[i],
            vectorLayout.getThreadsPerOuter()[i]};
        slicedIndices[pos] =
            linearizeIndex(rewriter, indices[pos], ids, sizes,
                           vectorLayout.getElementsPerThread()[i]);
      }
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

} // namespace

void populateGPUDistributeNestedLayoutAttrPatterns(
    Value threadId, RewritePatternSet &patterns) {
  patterns.add<DistributeTransferReadNestedLayoutAttr>(patterns.getContext(),
                                                       threadId);
}

}; // namespace mlir::iree_compiler
