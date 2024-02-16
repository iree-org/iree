// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
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

#define DEBUG_TYPE "iree-gpu-nested-layout-distribution"

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
  ValueRange threadIds = vectorLayout.computeThreadIds(threadId, rewriter);

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

enum class ContractKind { MK_KN_MN, UNKNOWN };

ContractKind inferContractKind(MLIRContext *ctx, SmallVector<AffineMap> maps) {
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [&](MapList m) { return AffineMap::inferFromExprList(m, ctx); };
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  if (maps == infer({{m, k}, {k, n}, {m, n}}))
    return ContractKind::MK_KN_MN;
  return ContractKind::UNKNOWN;
}

/// Pattern to distribute `vector.transfer_write` ops with nested layouts.
struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeContract(MLIRContext *context, Value threadId)
      : OpDistributionPattern(context), threadId(threadId) {}

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<VectorType>(contractOp.getResultType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          contractOp, "unhandled contraction to scalar value");
    }

    auto resultValue = cast<VectorValue>(contractOp.getResult());
    NestedLayoutAttr resultLayout =
        dyn_cast<NestedLayoutAttr>(signature[resultValue]);
    if (!resultLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing nested layout for contraction result");
    }
    int64_t rank = resultLayout.getBatchOrder().size();

    NestedLayoutAttr lhsLayout =
        dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
    if (!lhsLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing nested layout for contraction lhs");
    }
    NestedLayoutAttr rhsLayout =
        dyn_cast<NestedLayoutAttr>(signature[contractOp.getRhs()]);
    if (!rhsLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing nested layout for contraction rhs");
    }

    auto mfmaAttr =
        contractOp->getAttrOfType<IREE::GPU::MFMAAttr>("iree.amdgpu.mfma");
    if (!mfmaAttr) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing iree.amdgpu.mfma intrinsic attribute");
    }

    ContractKind contractKind =
        inferContractKind(getContext(), contractOp.getIndexingMapsArray());
    if (contractKind == ContractKind::UNKNOWN) {
      return rewriter.notifyMatchFailure(contractOp, "unknown contract kind");
    }

    SmallVector<int64_t> distShape = resultLayout.getDistributedShape();
    LLVM_DEBUG({
      llvm::dbgs() << "distributed shape: [";
      llvm::interleaveComma(distShape, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    // Create a zero vector with the full distributed vector shape for
    // accumulating unrolled contraction results.
    auto tileType = VectorType::get(distShape, resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(
        contractOp.getLoc(), tileType, rewriter.getZeroAttr(tileType));
    VectorValue finalTile = cast<VectorValue>(zero);
    LLVM_DEBUG(llvm::dbgs() << "init tile: " << finalTile << "\n");

    SmallVector<int64_t, 2> lhsBatchOffsets(rank, 0);
    SmallVector<int64_t, 2> rhsBatchOffsets(rank, 0);

    ArrayRef<int64_t> resultBatches = resultLayout.getBatchesPerSubgroup();
    SmallVector<int64_t, 2> resultBatchTileSizes(rank, 1);
    LLVM_DEBUG({
      llvm::dbgs() << "result batches: [";
      llvm::interleaveComma(resultBatches, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    // Iterate over all result batches and unroll computation to direct MFMA
    // intrinsic ops.
    Location loc = contractOp.getLoc();
    auto resultTiles = StaticTileOffsetRange(
        resultBatches, resultBatchTileSizes, resultLayout.getBatchOrder());
    for (SmallVector<int64_t, 2> resultBatchOffsets : resultTiles) {
      LLVM_DEBUG({
        llvm::dbgs() << "current result batch offsets: [";
        llvm::interleaveComma(resultBatchOffsets, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

      // Get the slice of the accumulator in this batch.
      Value acc = getDistributed(
          rewriter, cast<VectorValue>(contractOp.getAcc()), resultLayout);
      Value accSlice =
          rewriter.create<vector::ExtractOp>(loc, acc, resultBatchOffsets);

      // Get the k batch size for lhs and rhs vector.
      std::optional<int64_t> kBatch =
          getKBatchSize(contractKind, lhsLayout, rhsLayout);
      LLVM_DEBUG(llvm::dbgs() << "k batch size = " << kBatch << "\n");
      if (!kBatch) {
        return rewriter.notifyMatchFailure(contractOp,
                                           "A/B vector k batch mismatch");
      }

      // Perform contraction. Do separate outer product with mfma operation and
      // accumulate to the same vector.
      for (int k = 0; k < kBatch; ++k) {
        if (!getOperandBatchOffsets(contractKind, k, resultBatchOffsets,
                                    resultLayout, lhsBatchOffsets,
                                    rhsBatchOffsets, lhsLayout, rhsLayout)) {
          return rewriter.notifyMatchFailure(
              contractOp, "cannot deduce lhs/rhs batch offsets");
        }
        LLVM_DEBUG({
          llvm::dbgs() << "current lhs batch offsets: [";
          llvm::interleaveComma(lhsBatchOffsets, llvm::dbgs());
          llvm::dbgs() << "]\n";
          llvm::dbgs() << "current rhs batch offsets: [";
          llvm::interleaveComma(rhsBatchOffsets, llvm::dbgs());
          llvm::dbgs() << "]\n";
        });
        Value lhsSlice = rewriter.create<vector::ExtractOp>(
            loc, getDistributed(rewriter, contractOp.getLhs(), lhsLayout),
            lhsBatchOffsets);
        Value rhsSlice = rewriter.create<vector::ExtractOp>(
            loc, getDistributed(rewriter, contractOp.getRhs(), rhsLayout),
            rhsBatchOffsets);
        accSlice = computeMMA(rewriter, loc, lhsSlice, rhsSlice, accSlice,
                              mfmaAttr.getIntrinsic().getValue());
      }
      finalTile = rewriter.create<vector::InsertOp>(loc, accSlice, finalTile,
                                                    resultBatchOffsets);
    }

    replaceOpWithDistributedValues(rewriter, contractOp, finalTile);
    return success();
  }

  std::optional<int64_t> getKBatchSize(ContractKind kind,
                                       NestedLayoutAttr lhsLayout,
                                       NestedLayoutAttr rhsLayout) const {
    int64_t lhsKBatch = 0, rhsKBatch = 0;
    if (kind == ContractKind::MK_KN_MN) {
      lhsKBatch = lhsLayout.getBatchesPerSubgroup()[1];
      rhsKBatch = rhsLayout.getBatchesPerSubgroup()[0];
    } else {
      return std::nullopt;
    }

    if (lhsKBatch != rhsKBatch)
      return std::nullopt;
    return lhsKBatch;
  }

  bool getOperandBatchOffsets(
      ContractKind kind, int64_t kOffset,
      // Copy intentionally given we need to mutate in the function body
      SmallVector<int64_t, 2> resultOffsets, NestedLayoutAttr resultLayout,
      SmallVector<int64_t, 2> &lhsOffsets, SmallVector<int64_t, 2> &rhsOffsets,
      NestedLayoutAttr lhsLayout, NestedLayoutAttr rhsLayout) const {
    // The result offsets are permutated and we need to revert the permutation.
    // The following works based on the fact that for 2-D cases, a permuation
    // vector's reverse is just itself. So this only works for 2-D cases.
    applyPermutationToVector(resultOffsets, resultLayout.getBatchOrder());

    // resultOffsets contains batch indices into the C/D vector. It is a 2-D
    // index for both M and N. We need to split out for M and N, and add index
    // for K.
    if (kind == ContractKind::MK_KN_MN) {
      lhsOffsets[0] = resultOffsets[0];
      lhsOffsets[1] = kOffset;
      rhsOffsets[0] = kOffset;
      rhsOffsets[1] = resultOffsets[1];
    } else {
      return false;
    }

    // Now apply permutation on lhs/rhs according to their batch order.
    applyPermutationToVector(lhsOffsets, lhsLayout.getBatchOrder());
    applyPermutationToVector(rhsOffsets, rhsLayout.getBatchOrder());
    return true;
  }

  VectorType getMFMAVectorType(Value value) const {
    auto type = cast<VectorType>(value.getType());
    SmallVector<int64_t> shape;
    for (int64_t dim : type.getShape()) {
      if (dim != 1)
        shape.push_back(dim);
    }
    return VectorType::get(shape, type.getElementType());
  }

  Value computeMMA(OpBuilder &builder, Location loc, Value a, Value b, Value c,
                   IREE::GPU::MFMAIntrinsic intrinsic) const {
    // TODO: query these types from the intrinsic attribute outside of this
    // function.
    VectorType aType = getMFMAVectorType(a);
    VectorType bType = getMFMAVectorType(b);
    VectorType cType = getMFMAVectorType(c);
    Value aCast = builder.create<vector::ShapeCastOp>(a.getLoc(), aType, a);
    Value bCast = builder.create<vector::ShapeCastOp>(b.getLoc(), bType, b);
    Value cCast = builder.create<vector::ShapeCastOp>(c.getLoc(), cType, c);

    uint32_t m, n, k, blocks;
    switch (intrinsic) {
    case IREE::GPU::MFMAIntrinsic::F16_16x16x16_F32:
      m = n = k = 16;
      blocks = 1;
      break;
    case IREE::GPU::MFMAIntrinsic::F16_32x32x8_F32:
      m = n = 32;
      k = 8;
      blocks = 1;
      break;
    }
    Value mfmaOp = builder.create<amdgpu::MFMAOp>(loc, cType, m, n, k, blocks,
                                                  aCast, bCast, cCast);
    return builder.create<vector::ShapeCastOp>(c.getLoc(), c.getType(), mfmaOp);
  }

  Value threadId;
};

} // namespace

void populateGPUDistributeNestedLayoutAttrPatterns(
    Value threadId, RewritePatternSet &patterns) {
  patterns.add<DistributeContract, DistributeTransferReadNestedLayoutAttr,
               DistributeTransferWriteNestedLayoutAttr>(patterns.getContext(),
                                                        threadId);
}

}; // namespace mlir::iree_compiler
