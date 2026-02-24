// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Utils/Indexing.h"
#include "iree/compiler/Utils/Permutation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/PatternApplicator.h"

namespace mlir::iree_compiler {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

static bool isBroadcast(AffineExpr expr) {
  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
    return constExpr.getValue() == 0;
  }
  return false;
}

/// Given a set of base transfer |indices|, |offsets| for the batch/outer
/// dimensions, and distributed warp and thread indices, computes the indices
/// of the distributed transfer operation based on the |vectorLayout|.
static SmallVector<Value> getTransferIndicesFromNestedLayout(
    OpBuilder &b, ValueRange indices, ArrayRef<int64_t> offsets,
    NestedLayoutAttr vectorLayout, AffineMap permutationMap,
    ArrayRef<Value> warpIndices, ArrayRef<Value> threadIndices) {

  int64_t rank = vectorLayout.getRank();
  // Permute the batch and outer vector offsets to match the order of
  // the vector dimensions using the inverse of the batch/offset order.
  ArrayRef<int64_t> batchOffsets(offsets.begin(), rank);
  ArrayRef<int64_t> outerVectorOffsets(offsets.begin() + rank, rank);

  SmallVector<Value> slicedIndices(indices);
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
        warpIndices[i], arith::ConstantIndexOp::create(b, loc, batchOffsets[i]),
        arith::ConstantIndexOp::create(b, loc, outerVectorOffsets[i]),
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
    if (std::optional<int64_t> offsetConst = getConstantIntValue(offset)) {
      disjoint = *offsetConst < elementCount;
    }
    slicedIndices[pos] =
        affine::AffineLinearizeIndexOp::create(b, loc, ids, sizes, disjoint);
  }
  return slicedIndices;
}

static SmallVector<int64_t>
getDistributedTransferOffsetsFromNestedLayout(ArrayRef<int64_t> offsets,
                                              NestedLayoutAttr vectorLayout) {

  int64_t rank = vectorLayout.getRank();
  ArrayRef<int64_t> batchOffsets(offsets.begin(), rank);
  ArrayRef<int64_t> outerOffsets(offsets.begin() + rank, rank);
  ArrayRef<int64_t> outerSizes = vectorLayout.getOuterTile();
  ArrayRef<int64_t> elementSizes = vectorLayout.getElementTile();

  SmallVector<int64_t> slicedOffsets;
  slicedOffsets.reserve(rank);
  for (auto [batchOffset, outerOffset, outerSize, elementSize] :
       llvm::zip(batchOffsets, outerOffsets, outerSizes, elementSizes)) {
    slicedOffsets.push_back(batchOffset * outerSize * elementSize +
                            outerOffset * elementSize);
  }
  return slicedOffsets;
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

/// Given a distributed vector that has B1XB2xO1XO2xE1XE2,
/// convert that to B1XO1xE1xB2xO2xE2 form.
static VectorValue getDeinterleavedPackedForm(PatternRewriter &rewriter,
                                              VectorValue val,
                                              NestedLayoutAttr layout) {
  Location loc = val.getDefiningOp()->getLoc();
  SmallVector<int64_t> interleavedPackedShape(layout.getRank() * 3, 0);
  for (int64_t undistributedDim : llvm::seq<int64_t>(layout.getRank())) {
    SmallVector<int64_t> packedShapePerDim =
        layout.getPackedShapeForUndistributedDim(undistributedDim);
    interleavedPackedShape[layout.getRank() * 0 + undistributedDim] =
        packedShapePerDim[1];
    interleavedPackedShape[layout.getRank() * 1 + undistributedDim] =
        packedShapePerDim[2];
    interleavedPackedShape[layout.getRank() * 2 + undistributedDim] =
        packedShapePerDim[4];
  }
  VectorType interleavedPackedType =
      VectorType::get(interleavedPackedShape, val.getType().getElementType());
  VectorValue interleavedPackedShaped =
      vector::ShapeCastOp::create(rewriter, loc, interleavedPackedType, val);

  // 0 1 2 3 4 5 ---> 0 2 4 1 3 5
  SmallVector<int64_t> perm;
  perm.reserve(layout.getRank() * 3);
  for (int64_t undistributedDim : llvm::seq<int64_t>(layout.getRank())) {
    for (int64_t tileGroupIdx : llvm::seq<int64_t>(3)) {
      perm.push_back(tileGroupIdx * layout.getRank() + undistributedDim);
    }
  }
  return vector::TransposeOp::create(rewriter, loc, interleavedPackedShaped,
                                     perm);
}

/// Given a distributed vector that has B1XB2xO1XO2xE1XE2,
/// convert that to [B1XO1xE1]x[B2xO2xE2] form.
static VectorValue getDeinterleavedUnpackedForm(PatternRewriter &rewriter,
                                                VectorValue val,
                                                NestedLayoutAttr layout) {
  Location loc = val.getDefiningOp()->getLoc();
  VectorValue deinterleavedPacked =
      getDeinterleavedPackedForm(rewriter, val, layout);
  ArrayRef<int64_t> deinterleavedPackedShape =
      deinterleavedPacked.getType().getShape();
  SmallVector<int64_t> unpackedShape;
  unpackedShape.reserve(layout.getRank() * 3);
  for (int64_t unDistrDim : llvm::seq<int64_t>(layout.getRank())) {
    int64_t collapsedDimLen = deinterleavedPackedShape[unDistrDim * 3 + 0] *
                              deinterleavedPackedShape[unDistrDim * 3 + 1] *
                              deinterleavedPackedShape[unDistrDim * 3 + 2];
    unpackedShape.push_back(collapsedDimLen);
  }
  VectorType unpackedType = VectorType::get(
      unpackedShape, deinterleavedPacked.getType().getElementType());
  return vector::ShapeCastOp::create(rewriter, loc, unpackedType,
                                     deinterleavedPacked);
}

/// Given a distributed vector that has [B1xO1xE1]x[B2xO2xE2],
/// convert that to B1 x B2 x O1 X O2 x E1 x E2 form.
static VectorValue getInterleavedPackedForm(PatternRewriter &rewriter,
                                            VectorValue val,
                                            NestedLayoutAttr layout) {
  Location loc = val.getDefiningOp()->getLoc();
  SmallVector<int64_t> nonInterleavedPackedShape;
  nonInterleavedPackedShape.reserve(layout.getRank() * 3);
  for (int64_t undistributedDim : llvm::seq<int64_t>(layout.getRank())) {
    SmallVector<int64_t> packedShapePerDim =
        layout.getPackedShapeForUndistributedDim(undistributedDim);
    nonInterleavedPackedShape.push_back(packedShapePerDim[1]);
    nonInterleavedPackedShape.push_back(packedShapePerDim[2]);
    nonInterleavedPackedShape.push_back(packedShapePerDim[4]);
  }
  VectorType nonInterleavedPackedType = VectorType::get(
      nonInterleavedPackedShape, val.getType().getElementType());
  VectorValue nonInterleavedPackedShaped =
      vector::ShapeCastOp::create(rewriter, loc, nonInterleavedPackedType, val);
  // 0 1 2 3 4 5 ---> 0 3 1 4 2 5
  SmallVector<int64_t> perm;
  perm.reserve(layout.getRank() * 3);
  for (int64_t tileGroupIdx : llvm::seq<int64_t>(3)) {
    for (int64_t undistributedDim : llvm::seq<int64_t>(layout.getRank())) {
      perm.push_back(tileGroupIdx + 3 * undistributedDim);
    }
  }
  return vector::TransposeOp::create(rewriter, loc, nonInterleavedPackedShaped,
                                     perm);
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
  if (threadIds.empty() && rank != 0) {
    return failure();
  }
  warpIndices = SmallVector<Value>(threadIds.begin(), threadIds.begin() + rank);
  threadIndices = SmallVector<Value>(threadIds.begin() + rank,
                                     threadIds.begin() + 2 * rank);
  return success();
}

static VectorValue getSlicedPermutedValue(PatternRewriter &rewriter,
                                          Location loc,
                                          ArrayRef<int64_t> offsets,
                                          NestedLayoutAttr vectorLayout,
                                          VectorValue mask) {
  SmallVector<int64_t> sliceMaskOffsets =
      getDistributedTransferOffsetsFromNestedLayout(offsets, vectorLayout);
  SmallVector<int64_t> strides(vectorLayout.getElementTile().size(), 1);
  VectorValue slicedMask = vector::ExtractStridedSliceOp::create(
      rewriter, loc, mask, sliceMaskOffsets, vectorLayout.getElementTile(),
      strides);
  return slicedMask;
}

/// Project a vector based on a provided projection map.
/// Firstly, this will transpose the vector in a way sliced out
/// dims become outermost. Then it performs a vector.extract
/// remove the dims that are not present in the results of the map.
/// Note that the implementation is similiar to vector.extract_stride_slice
/// but with projecting out the indexed/sliced dimensions from the result.
static VectorValue projectVector(RewriterBase &rewriter, Location loc,
                                 VectorValue val, AffineMap projectionMap) {
  SmallVector<int64_t> remaningDims;
  auto allDims =
      llvm::to_vector(llvm::seq<int64_t>(projectionMap.getNumDims()));
  llvm::SmallDenseSet<int64_t> slicedDims(allDims.begin(), allDims.end());
  for (int64_t resultIdx : llvm::seq<int64_t>(projectionMap.getNumResults())) {
    int64_t iterSpacePos = projectionMap.getDimPosition(resultIdx);
    remaningDims.push_back(iterSpacePos);
    slicedDims.erase(iterSpacePos);
  }

  auto transposePerm = llvm::to_vector_of<int64_t>(slicedDims);
  transposePerm.append(remaningDims);
  auto transposed =
      vector::TransposeOp::create(rewriter, loc, val, transposePerm);

  SmallVector<int64_t> extractedPos(slicedDims.size(), 0);
  auto sliced =
      vector::ExtractOp::create(rewriter, loc, transposed, extractedPos);
  return cast<VectorValue>(sliced.getResult());
}

static VectorValue extractSliceAsVector(RewriterBase &rewriter, Location loc,
                                        Value src, ArrayRef<int64_t> offsets) {
  Value slice = vector::ExtractOp::create(rewriter, loc, src, offsets);
  // Promote the slicedVector to 0-d vector if it is a scalar.
  if (!isa<VectorType>(slice.getType())) {
    auto promotedType = VectorType::get({}, getElementTypeOrSelf(slice));
    slice = vector::BroadcastOp::create(rewriter, loc, promotedType, slice);
  }
  return cast<VectorValue>(slice);
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

    NestedLayoutAttr vectorLayout =
        dyn_cast<NestedLayoutAttr>(signature[readOp.getResult()]);
    if (!vectorLayout) {
      return rewriter.notifyMatchFailure(readOp,
                                         "non-nested transfer_read layout");
    }

    VectorValue mask = readOp.getMask();
    NestedLayoutAttr maskLayout;
    if (mask) {
      maskLayout = dyn_cast<NestedLayoutAttr>(signature[mask]);
      if (!maskLayout) {
        return rewriter.notifyMatchFailure(readOp,
                                           "non-nested mask vector layout");
      }
      mask = getDistributed(rewriter, mask, maskLayout);
      mask = getDeinterleavedUnpackedForm(rewriter, mask, maskLayout);
    }

    // Guard on memrefs for distribution. In isolation this pattern is agnostic
    // to tensors or memrefs.
    if (!isa<MemRefType>(readOp.getBase().getType())) {
      return rewriter.notifyMatchFailure(readOp,
                                         "distribution expects memrefs");
    }

    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    SmallVector<int64_t> tileShape = getElementVectorTileShape(vectorLayout);
    int64_t rank = vectorLayout.getRank();

    Type elementType = readOp.getBase().getType().getElementType();
    auto vectorType = VectorType::get(distShape, elementType);
    // The shape of the vector we read is pre-permutation. The permutation is
    // a transpose on the resulting read vector.
    auto innerVectorType =
        VectorType::get(vectorLayout.getElementTile(), elementType);

    // Initialize the full distributed vector for unrolling the batch/outer
    // vector dimensions.
    Value zero =
        arith::ConstantOp::create(rewriter, readOp.getLoc(), vectorType,
                                  rewriter.getZeroAttr(vectorType));
    VectorValue acc = cast<VectorValue>(zero);

    SmallVector<Value> warpIndices, threadIndices;
    if (failed(populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
                                            vectorLayout, warpIndices,
                                            threadIndices))) {
      return rewriter.notifyMatchFailure(
          readOp, "warp or thread tiles have overlapping strides");
    }

    ValueRange indices = readOp.getIndices();
    AffineMap permMap = readOp.getPermutationMap();
    SmallVector<int64_t> strides(rank, 1);

    SmallVector<SmallVector<int64_t>> allMaskOffsets;
    if (mask) {
      SmallVector<int64_t> maskDistShape = maskLayout.getDistributedShape();
      SmallVector<int64_t> maskTileShape =
          getElementVectorTileShape(maskLayout);
      allMaskOffsets =
          llvm::to_vector(StaticTileOffsetRange(maskDistShape, maskTileShape));
    }

    for (auto [idx, offsets] :
         llvm::enumerate(StaticTileOffsetRange(distShape, tileShape))) {
      SmallVector<Value> slicedIndices = getTransferIndicesFromNestedLayout(
          rewriter, indices, offsets, vectorLayout, permMap, warpIndices,
          threadIndices);

      VectorValue slicedMask = nullptr;
      if (mask) {
        SmallVector<int64_t> maskDistShape = maskLayout.getDistributedShape();
        SmallVector<int64_t> maskTileShape =
            getElementVectorTileShape(maskLayout);
        SmallVector<int64_t> maskOffsets = allMaskOffsets[idx];
        slicedMask = getSlicedPermutedValue(rewriter, readOp.getLoc(),
                                            maskOffsets, maskLayout, mask);
      }

      VectorValue slicedRead = vector::TransferReadOp::create(
          rewriter, readOp.getLoc(), innerVectorType, readOp.getBase(),
          slicedIndices, readOp.getPermutationMapAttr(), readOp.getPadding(),
          slicedMask, readOp.getInBoundsAttr());

      if (acc.getType().getRank() == 0) {
        // TODO: This should really be a folding pattern in
        // insert_strided_slice, but instead insert_strided_slice just doesn't
        // support 0-d vectors...
        acc = slicedRead;
      } else {
        acc = vector::InsertStridedSliceOp::create(
            rewriter, readOp.getLoc(), slicedRead, acc, offsets, strides);
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
                          int64_t subgroupSize, ArrayRef<int64_t> workgroupSize)
      : OpDistributionPattern(context), threadId(threadId),
        subgroupSize(subgroupSize) {

    // The number of threads in the workgroup is the product of the dimensions
    // of workgroupSize, unless workgroupSize is empty.
    if (!workgroupSize.empty()) {
      numThreadsInWorkgroup = llvm::product_of(workgroupSize);
    }
  }

  /// Compute a boolean in SIMT semantics that is true for the first virtual
  /// lane(thread) id (vtid) and virtual subgroup id (vsid) carrying broadcasted
  /// data.
  ///
  /// We do this by computing a basis for vtid and vsid computation, and adding
  /// a check for basis elements that are not used (i.e. they are duplicated)
  /// to be zero.
  FailureOr<Value> getNoOverlapCondition(OpBuilder &b, Location loc,
                                         NestedLayoutAttr layout) const {
    ArrayRef<int64_t> threadTile = layout.getThreadTile();
    ArrayRef<int64_t> threadStrides = layout.getThreadStrides();
    ArrayRef<int64_t> subgroupTile = layout.getSubgroupTile();
    // Multiply the subgroup strides by subgroup_size to reflect thread id
    // relative strides.
    auto subgroupStrides =
        llvm::map_to_vector(layout.getSubgroupStrides(),
                            [&](int64_t x) { return x * subgroupSize; });
    auto concatTiles =
        llvm::to_vector(llvm::concat<const int64_t>(subgroupTile, threadTile));
    auto concatStrides = llvm::to_vector(
        llvm::concat<const int64_t>(subgroupStrides, threadStrides));
    SmallVector<int64_t> basis;
    SmallVector<size_t> dimToResult;
    if (failed(basisFromSizesStrides(concatTiles, concatStrides, basis,
                                     dimToResult))) {
      return failure();
    }
    // Make the outer bound numThreadsInWorkgroup / prod(basis) to remove
    // redundant checks.
    if (numThreadsInWorkgroup.has_value()) {
      int64_t outerBound =
          numThreadsInWorkgroup.value() / llvm::product_of(basis);
      basis.insert(basis.begin(), outerBound);
    }
    // Create a delinearize operation and check that all results not present in
    // dimToResult are 0.
    SmallVector<Value> delinearized;
    b.createOrFold<affine::AffineDelinearizeIndexOp>(
        delinearized, loc, threadId, basis,
        /*hasOuterbound=*/numThreadsInWorkgroup.has_value());
    // Get all results which are not in dimToResult and check they are 0.
    Value condition = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
    for (auto [idx, result] : llvm::enumerate(delinearized)) {
      if (llvm::is_contained(dimToResult, idx)) {
        continue;
      }
      Value isZero = b.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, result,
          arith::ConstantIndexOp::create(b, loc, 0));
      condition = b.createOrFold<arith::AndIOp>(loc, condition, isZero);
    }
    return condition;
  }

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    NestedLayoutAttr vectorLayout =
        dyn_cast<NestedLayoutAttr>(signature[writeOp.getValueToStore()]);
    if (!vectorLayout) {
      return rewriter.notifyMatchFailure(writeOp,
                                         "non-nested transfer_write layout");
    }

    if (!isa<MemRefType>(writeOp.getBase().getType())) {
      return rewriter.notifyMatchFailure(writeOp,
                                         "distribution expects memrefs");
    }

    VectorValue mask = writeOp.getMask();
    NestedLayoutAttr maskLayout;
    if (mask) {
      maskLayout = dyn_cast<NestedLayoutAttr>(signature[mask]);
      if (!maskLayout) {
        return rewriter.notifyMatchFailure(writeOp,
                                           "non-nested mask vector layout");
      }
      mask = getDistributed(rewriter, mask, maskLayout);
      mask = getDeinterleavedUnpackedForm(rewriter, mask, maskLayout);
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

    // If the distribution results in threads writing to the same address, guard
    // with an scf.if to ensure only one thread writes per duplication group.
    Location loc = writeOp.getLoc();
    FailureOr<Value> doWrite =
        getNoOverlapCondition(rewriter, loc, vectorLayout);
    if (failed(doWrite)) {
      return rewriter.notifyMatchFailure(
          writeOp, "failed to compute no-overlap condition");
    }
    auto ifOp = scf::IfOp::create(rewriter, loc, doWrite.value());
    rewriter.setInsertionPoint(ifOp.thenYield());

    Value distributedVector =
        getDistributed(rewriter, writeOp.getValueToStore(), vectorLayout);

    ValueRange indices = writeOp.getIndices();
    AffineMap permMap = writeOp.getPermutationMap();

    SmallVector<SmallVector<int64_t>> allMaskOffsets;
    if (mask) {
      SmallVector<int64_t> maskDistShape = maskLayout.getDistributedShape();
      SmallVector<int64_t> maskTileShape =
          getElementVectorTileShape(maskLayout);
      allMaskOffsets =
          llvm::to_vector(StaticTileOffsetRange(maskDistShape, maskTileShape));
    }

    for (auto [idx, offsets] :
         llvm::enumerate(StaticTileOffsetRange(distShape, tileShape))) {
      SmallVector<Value> slicedIndices = getTransferIndicesFromNestedLayout(
          rewriter, indices, offsets, vectorLayout, permMap, warpIndices,
          threadIndices);
      // Extract the "element vector" from the inner most dimensions. All outer
      // dimensions are either unrolled or distributed such that this is a
      // contiguous slice.
      ArrayRef<int64_t> offsetArray(offsets);
      VectorValue slicedVector =
          extractSliceAsVector(rewriter, writeOp.getLoc(), distributedVector,
                               offsetArray.take_front(rank * 2));

      VectorValue slicedMask = nullptr;
      if (mask) {
        SmallVector<int64_t> maskDistShape = maskLayout.getDistributedShape();
        SmallVector<int64_t> maskTileShape =
            getElementVectorTileShape(maskLayout);
        SmallVector<int64_t> maskOffsets = allMaskOffsets[idx];
        slicedMask = getSlicedPermutedValue(rewriter, writeOp.getLoc(),
                                            maskOffsets, maskLayout, mask);
      }

      vector::TransferWriteOp::create(rewriter, writeOp.getLoc(), slicedVector,
                                      writeOp.getBase(), slicedIndices,
                                      writeOp.getPermutationMapAttr(),
                                      slicedMask, writeOp.getInBoundsAttr());
    }

    rewriter.eraseOp(writeOp);
    return success();
  }

  Value threadId;
  int64_t subgroupSize;
  std::optional<int64_t> numThreadsInWorkgroup = std::nullopt;
};

/// Pattern to distribute `vector.transfer_gather` ops with nested layouts.
struct DistributeTransferGather final
    : OpDistributionPattern<IREE::VectorExt::TransferGatherOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferGather(MLIRContext *context, Value threadId,
                           int64_t subgroupSize)
      : OpDistributionPattern(context), threadId(threadId),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(IREE::VectorExt::TransferGatherOp gatherOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {

    NestedLayoutAttr vectorLayout =
        dyn_cast<NestedLayoutAttr>(signature[gatherOp.getResult()]);
    if (!vectorLayout) {
      return rewriter.notifyMatchFailure(gatherOp,
                                         "non-nested transfer_gather layout");
    }

    SmallVector<NestedLayoutAttr> indexVecLayouts;
    SmallVector<VectorValue> disIndexVecs;
    for (Value indexVec : gatherOp.getIndexVecs()) {
      auto vec = cast<VectorValue>(indexVec);
      NestedLayoutAttr layout = dyn_cast<NestedLayoutAttr>(signature[vec]);
      if (!layout) {
        return rewriter.notifyMatchFailure(gatherOp,
                                           "non-nested index vec layout");
      }
      indexVecLayouts.push_back(layout);
      vec = getDistributed(rewriter, vec, layout);
      vec = getDeinterleavedUnpackedForm(rewriter, vec, layout);
      disIndexVecs.push_back(vec);
    }

    VectorValue mask = gatherOp.getMask();
    NestedLayoutAttr maskLayout;
    if (mask) {
      maskLayout = dyn_cast<NestedLayoutAttr>(signature[mask]);
      if (!maskLayout) {
        return rewriter.notifyMatchFailure(gatherOp,
                                           "non-nested mask vector layout");
      }
      mask = getDistributed(rewriter, mask, maskLayout);
      mask = getDeinterleavedUnpackedForm(rewriter, mask, maskLayout);
    }

    // Guard on memrefs for distribution. In isolation this pattern is agnostic
    // to tensors or memrefs.
    if (!isa<MemRefType>(gatherOp.getBase().getType())) {
      return rewriter.notifyMatchFailure(gatherOp,
                                         "distribution expects memrefs");
    }

    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    SmallVector<int64_t> tileShape = getElementVectorTileShape(vectorLayout);
    int64_t rank = vectorLayout.getRank();

    Type elementType = gatherOp.getBase().getType().getElementType();
    auto vectorType = VectorType::get(distShape, elementType);
    // The shape of the vector we read is pre-permutation. The permutation is
    // a transpose on the resulting read vector.
    auto innerVectorType =
        VectorType::get(vectorLayout.getElementTile(), elementType);

    // Initialize the full distributed vector for unrolling the batch/outer
    // vector dimensions.
    Value zero =
        arith::ConstantOp::create(rewriter, gatherOp.getLoc(), vectorType,
                                  rewriter.getZeroAttr(vectorType));
    VectorValue acc = cast<VectorValue>(zero);

    SmallVector<Value> warpIndices, threadIndices;
    if (failed(populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
                                            vectorLayout, warpIndices,
                                            threadIndices))) {
      return rewriter.notifyMatchFailure(
          gatherOp, "warp or thread tiles have overlapping strides");
    }

    ValueRange indices = gatherOp.getOffsets();
    SmallVector<int64_t> strides(rank, 1);

    // getPermutationMap inverts the source map, mapping gathered (symbol) and
    // broadcast (constant) dims to constant 0. This is correct here because
    // getTransferIndicesFromNestedLayout treats constant-0 dims as broadcast,
    // leaving the original base offset unchanged for gathered dimensions.
    AffineMap permMap = gatherOp.getPermutationMap();

    SmallVector<SmallVector<int64_t>> allMaskOffsets;
    std::vector<StaticTileOffsetRange::IteratorTy> allIndexVecOffsets;
    if (mask) {
      SmallVector<int64_t> maskDistShape = maskLayout.getDistributedShape();
      SmallVector<int64_t> maskTileShape =
          getElementVectorTileShape(maskLayout);
      allMaskOffsets =
          llvm::to_vector(StaticTileOffsetRange(maskDistShape, maskTileShape));
    }
    for (NestedLayoutAttr layout : indexVecLayouts) {
      SmallVector<int64_t> vecDistShape = layout.getDistributedShape();
      SmallVector<int64_t> vecTileShape = getElementVectorTileShape(layout);
      allIndexVecOffsets.push_back(
          StaticTileOffsetRange(vecDistShape, vecTileShape).begin());
    }

    for (auto [idx, offsets] :
         llvm::enumerate(StaticTileOffsetRange(distShape, tileShape))) {
      SmallVector<Value> slicedIndices = getTransferIndicesFromNestedLayout(
          rewriter, indices, offsets, vectorLayout, permMap, warpIndices,
          threadIndices);

      // Extract offset from index_vecs.
      SmallVector<Value> slicedIndexVecs;
      for (auto [indexVecIdx, disIndexVec, layout] :
           llvm::enumerate(disIndexVecs, indexVecLayouts)) {
        SmallVector<int64_t> offsets =
            llvm::to_vector(*(allIndexVecOffsets[indexVecIdx]));
        ++allIndexVecOffsets[indexVecIdx];
        VectorValue slicedIndexVec = getSlicedPermutedValue(
            rewriter, gatherOp.getLoc(), offsets, layout, disIndexVec);
        slicedIndexVecs.push_back(slicedIndexVec);
      }

      VectorValue slicedMask = nullptr;
      if (mask) {
        SmallVector<int64_t> maskDistShape = maskLayout.getDistributedShape();
        SmallVector<int64_t> maskTileShape =
            getElementVectorTileShape(maskLayout);
        SmallVector<int64_t> maskOffsets = allMaskOffsets[idx];
        slicedMask = getSlicedPermutedValue(rewriter, gatherOp.getLoc(),
                                            maskOffsets, maskLayout, mask);
      }

      VectorValue slicedGather = IREE::VectorExt::TransferGatherOp::create(
          rewriter, gatherOp.getLoc(), innerVectorType, gatherOp.getBase(),
          slicedIndices, slicedIndexVecs, gatherOp.getIndexingMapsAttr(),
          gatherOp.getPadding(), slicedMask);

      if (acc.getType().getRank() == 0) {
        // TODO: This should really be a folding pattern in
        // insert_strided_slice, but instead insert_strided_slice just doesn't
        // support 0-d vectors...
        acc = slicedGather;
      } else {
        acc = vector::InsertStridedSliceOp::create(
            rewriter, gatherOp.getLoc(), slicedGather, acc, offsets, strides);
      }
    }

    replaceOpWithDistributedValues(rewriter, gatherOp, acc);
    return success();
  }

  Value threadId;
  int64_t subgroupSize;
};

/// Pattern to distribute `iree_linalg_ext.map_scatter` ops with nested layouts.
/// Only the input is distributed, since the output is never a vector. The
/// distribution of the input is similar to that of a vector.transfer_write.
struct DistributeMapScatter final
    : OpDistributionPattern<IREE::LinalgExt::MapScatterOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeMapScatter(MLIRContext *context, Value threadId,
                       int64_t subgroupSize)
      : OpDistributionPattern(context), threadId(threadId),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(IREE::LinalgExt::MapScatterOp mapScatterOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto input = dyn_cast<VectorValue>(mapScatterOp.getInput());
    if (!input) {
      return rewriter.notifyMatchFailure(mapScatterOp, "input is not a vector");
    }
    NestedLayoutAttr vectorLayout =
        dyn_cast<NestedLayoutAttr>(signature[input]);
    if (!vectorLayout) {
      return rewriter.notifyMatchFailure(mapScatterOp,
                                         "non-nested map_scatter layout");
    }
    if (!isa<MemRefType>(mapScatterOp.getOutput().getType())) {
      return rewriter.notifyMatchFailure(mapScatterOp,
                                         "distribution expects memrefs");
    }
    SmallVector<Value> warpIndices, threadIndices;
    if (failed(populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
                                            vectorLayout, warpIndices,
                                            threadIndices))) {
      return rewriter.notifyMatchFailure(
          mapScatterOp, "warp or thread tiles have overlapping strides");
    }

    Value distributedVector = getDistributed(rewriter, input, vectorLayout);

    Location loc = mapScatterOp.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    SmallVector<int64_t> tileShape = getElementVectorTileShape(vectorLayout);
    for (auto [idx, offsets] :
         llvm::enumerate(StaticTileOffsetRange(distShape, tileShape))) {
      // Extract the "element vector" from the inner most dimensions. All outer
      // dimensions are either unrolled or distributed such that this is a
      // contiguous slice.
      ArrayRef<int64_t> offsetArray(offsets);
      VectorValue distributedInput = extractSliceAsVector(
          rewriter, loc, distributedVector,
          offsetArray.take_front(vectorLayout.getRank() * 2));

      // Clone the map_scatter op with the "element vector" as the input, and
      // adjust the transformation region to account for the distributed
      // offsets.
      AffineMap permutationMap =
          rewriter.getMultiDimIdentityMap(input.getType().getRank());
      SmallVector<Value> indices(input.getType().getRank(), zero);
      SmallVector<Value> distributedOffsets =
          getTransferIndicesFromNestedLayout(rewriter, indices, offsets,
                                             vectorLayout, permutationMap,
                                             warpIndices, threadIndices);
      IREE::LinalgExt::MapScatterOp distributedMapScatter =
          clone(rewriter, mapScatterOp, mapScatterOp.getResultTypes(),
                {distributedInput, mapScatterOp.getOutput()});
      int64_t sliceRank = distributedInput.getType().getRank();
      int64_t rankDiff = input.getType().getRank() - sliceRank;
      // Add the distributed offsets in the map_scatter transformation body.
      auto transformationBuilder = [&](ArrayRef<BlockArgument> newIndices) {
        SmallVector<Value> replacementIndices(distributedOffsets);
        for (auto [i, replacementIdx] : llvm::enumerate(replacementIndices)) {
          // Rank-reduced dimensions can be directly replaced by the distributed
          // index, since their size is 1 in the new map_scatter input.
          if (i < rankDiff) {
            continue;
          }
          // Otherwise, the dimension is a contiguous element dimension, so
          // the mapping is achieved by adding the corresponding block argument
          // to the sliced index.
          BlockArgument newTransformationIdx = newIndices[i - rankDiff];
          replacementIdx = arith::AddIOp::create(
              rewriter, loc, newTransformationIdx, replacementIdx);
        }
        return replacementIndices;
      };
      distributedMapScatter.insertTransformationAtStart(
          rewriter, transformationBuilder, sliceRank);
    }

    rewriter.eraseOp(mapScatterOp);
    return success();
  }

  Value threadId;
  int64_t subgroupSize;
};

static VectorValue broadcastToShape(RewriterBase &rewriter, Value source,
                                    ArrayRef<int64_t> shape,
                                    ArrayRef<bool> broadcastedDims) {
  // Since vector dialect does not have a broadcastToShape operation, we first
  // broadcast and then transpose the vector to get the desired broadcast.
  assert(shape.size() == broadcastedDims.size());
  // Move all broadcastedDims as leading dimensions and perform the broadcast.
  SmallVector<int64_t> broadcastedIndices;
  SmallVector<int64_t> unbroadcastedIndices;
  for (auto [i, isBroadcasted] : llvm::enumerate(broadcastedDims)) {
    if (isBroadcasted) {
      broadcastedIndices.push_back(i);
    } else {
      unbroadcastedIndices.push_back(i);
    }
  }

  SmallVector<int64_t> perm = llvm::to_vector(
      llvm::concat<int64_t>(broadcastedIndices, unbroadcastedIndices));
  SmallVector<int64_t> leadingBroadcastShape = applyPermutation(shape, perm);

  VectorType broadcastedVecType =
      VectorType::get(leadingBroadcastShape, getElementTypeOrSelf(source));
  VectorValue broadcasted = vector::BroadcastOp::create(
      rewriter, source.getLoc(), broadcastedVecType, source);

  // Transpose the broadcasted dims to the right place.
  SmallVector<int64_t> inversePerm = invertPermutationVector(perm);
  if (isIdentityPermutation(inversePerm)) {
    return broadcasted;
  }
  return vector::TransposeOp::create(rewriter, source.getLoc(), broadcasted,
                                     inversePerm);
}

struct DistributeBroadcast final : OpDistributionPattern<vector::BroadcastOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcastOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue dstVector = broadcastOp.getVector();
    auto vectorLayout = dyn_cast<NestedLayoutAttr>(signature[dstVector]);
    if (!vectorLayout) {
      return rewriter.notifyMatchFailure(broadcastOp,
                                         "non-nested result vector layout");
    }
    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();

    // The way nested layout distribution works, there is no partial
    // thread distribution, so a broadcast is always thread local. So we need to
    // broadcast the shape:
    //
    // [o_batch, o_outer, o_element] to
    // [b_batch, o_batch, b_outer, o_outer, b_element, o_element]
    //
    // where b_... is broadcasted dimensions and o_... is old dimensions.
    SmallVector<bool> broadcastedDims(distShape.size(), false);

    // Get a layout for the broadcasted dimensions.
    VectorValue srcVector = dyn_cast<VectorValue>(broadcastOp.getSource());
    int64_t broadcastRank = vectorLayout.getRank();
    if (srcVector) {
      broadcastRank -= srcVector.getType().getRank();
    }

    // Mark the first `broadcastRank` dims in each tile to be broadcasted.
    int64_t rank = vectorLayout.getRank();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < broadcastRank; ++j) {
        broadcastedDims[j + (i * rank)] = true;
      }
    }

    Value distSource = broadcastOp.getSource();
    if (srcVector) {
      auto sourceLayout = dyn_cast<NestedLayoutAttr>(signature[srcVector]);
      if (!sourceLayout) {
        return rewriter.notifyMatchFailure(broadcastOp,
                                           "non-nested source vector layout");
      }
      distSource = getDistributed(rewriter, srcVector, sourceLayout);
    }
    VectorValue broadcasted =
        broadcastToShape(rewriter, distSource, distShape, broadcastedDims);
    replaceOpWithDistributedValues(rewriter, broadcastOp, broadcasted);
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
    : MaskedOpDistributionPattern<vector::MultiDimReductionOp> {
  using MaskedOpDistributionPattern::MaskedOpDistributionPattern;

  DistributeMultiReduction(MLIRContext *context, int64_t subgroupSize,
                           int64_t maxBitsPerShuffle, int64_t benefit = 1)
      : MaskedOpDistributionPattern(context, benefit),
        subgroupSize(subgroupSize), maxBitsPerShuffle(maxBitsPerShuffle) {}

  LogicalResult
  matchAndRewrite(vector::MultiDimReductionOp multiReduceOp,
                  DistributionSignature &signature, vector::MaskOp maskOp,
                  std::optional<DistributionSignature> &maskSignature,
                  PatternRewriter &rewriter) const {
    Location loc = multiReduceOp.getLoc();
    VectorValue srcVector = multiReduceOp.getSource();
    Value acc = multiReduceOp.getAcc();
    Value res = multiReduceOp.getResult();
    auto accVector = dyn_cast<VectorValue>(acc);
    auto resVector = dyn_cast<VectorValue>(res);

    auto srcLayout =
        dyn_cast_if_present<NestedLayoutAttr>(signature[srcVector]);
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

    VectorValue mask = nullptr;
    if (maskOp) {
      auto maskLayout = dyn_cast_if_present<NestedLayoutAttr>(
          maskSignature.value()[maskOp.getMask()]);
      if (!maskLayout) {
        return rewriter.notifyMatchFailure(maskOp,
                                           "expected nested layout attr");
      }
      mask = getDistributed(rewriter, maskOp.getMask(), maskLayout);
      Value passThruSrc = getCombiningIdentityValue(
          loc, rewriter, multiReduceOp.getKind(), disSrc.getType());

      disSrc = cast<VectorValue>(
          arith::SelectOp::create(rewriter, loc, mask, disSrc, passThruSrc)
              .getResult());
    }

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
    Value localReduction = vector::MultiDimReductionOp::create(
        rewriter, loc, disSrc, localInit, distributedReductionMask,
        multiReduceOp.getKind());

    // TODO: As per current upstream lowering implementations, there is no point
    // in doing this because it does a select much later in a finer granularity
    // rather than supporting predication. Moreover, since we are doing a select
    // to cater reductions accross the distribution, we can choose not to mask
    // the op post-distribution.

    VectorValue locallyReduced;
    if (accVector) {
      locallyReduced = dyn_cast<VectorValue>(localReduction);
    } else {
      // Broadcast scalar accumulator to vector.
      VectorType vecType = VectorType::get(ArrayRef{int64_t(1)}, elemTy);
      locallyReduced =
          vector::BroadcastOp::create(rewriter, loc, vecType, localReduction);
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
      VectorValue flat = vector::ShapeCastOp::create(rewriter, loc, flatVecType,
                                                     locallyReduced);

      // Do inter-thread/warp reduce.
      FailureOr<VectorValue> threadReducedFlat = doThreadReduction(
          rewriter, srcLayout, flat, multiReduceOp.getKind(), reducedDims);
      if (failed(threadReducedFlat)) {
        return failure();
      }

      // Do reduction against accumulator, which needs to be done after thread
      // reduction.
      threadReduced = vector::ShapeCastOp::create(rewriter, loc, shaped,
                                                  threadReducedFlat.value());
    }

    if (!accVector) {
      // Broadcast the scalar (e.g., f32) to a vector type (e.g., vector<f32>)
      // because the following implementation requires the operand to be a
      // vector.
      disAcc = vector::BroadcastOp::create(rewriter, loc, shaped, disAcc);
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
        Value accReducedVal = vector::ExtractOp::create(
            rewriter, loc, accReduction, ArrayRef{int64_t(0)});
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

    auto constOp = arith::ConstantOp::create(rewriter, loc,
                                             rewriter.getZeroAttr(flatVecType));
    auto res = cast<VectorValue>(constOp.getResult());

    for (unsigned i = 0; i < numElements; ++i) {
      Value extracted = vector::ExtractOp::create(rewriter, loc, flat, i);
      // Reduce across all reduction dimensions 1-by-1.
      for (unsigned i = 0, e = reductionMask.size(); i != e; ++i) {
        if (reductionMask[i]) {
          int64_t offset = getShuffleOffset(layout, i);
          int64_t width = getShuffleWidth(layout, i);
          assert(offset <= std::numeric_limits<uint32_t>::max() &&
                 width <= std::numeric_limits<uint32_t>::max());

          extracted = gpu::SubgroupReduceOp::create(
              rewriter, loc, extracted, combiningKindToAllReduce(kind),
              /*uniform=*/false, /*cluster_size=*/width,
              /*cluster_stride=*/offset);
        }
      }

      res = vector::InsertOp::create(rewriter, loc, extracted, res, i);
    }
    return res;
  }

  Value getBufferForSubgroupReduction(RewriterBase &rewriter, MemRefType memTy,
                                      Value val) const {
    auto alloc = memref::AllocOp::create(rewriter, val.getLoc(), memTy);
    // Insert gpu.barrier to make sure previous iteration of batch loop has
    // fully read the subgroup partial reductions.
    // TODO: We should be only creating a barrier if this buffer is going to be
    // reused.
    gpu::BarrierOp::create(rewriter, val.getLoc(), alloc);
    return alloc;
  }

  NestedLayoutAttr
  getLayoutForReductionFromBuffer(NestedLayoutAttr srcLayout,
                                  ArrayRef<int64_t> reductionDims) const {
    // Create new layout where the elements of a subgroup are
    // distributed to every threads.
    IREE::VectorExt::NestedLayoutAttr bufferReduceLayout;
    auto subgroupTileLens =
        llvm::to_vector_of<int64_t>(srcLayout.getSubgroupTile());
    auto batchTileLens = llvm::to_vector_of<int64_t>(srcLayout.getBatchTile());
    auto outerTileLens = llvm::to_vector_of<int64_t>(srcLayout.getOuterTile());
    auto threadTileLens =
        llvm::to_vector_of<int64_t>(srcLayout.getThreadTile());
    auto elementTileLens =
        llvm::to_vector_of<int64_t>(srcLayout.getElementTile());
    auto subgroupStrides =
        llvm::to_vector_of<int64_t>(srcLayout.getSubgroupStrides());
    auto threadStrides =
        llvm::to_vector_of<int64_t>(srcLayout.getThreadStrides());

    // Check if we had enough threads on one of the reduction dimensions
    // to use for a subgroup reduction. If not, do a serialized reduction.
    // This usually works, because we would be distributing the reduction
    // dimension on atleast more threads than number of subgroups, and if we
    // aren't, it's probably best to do a serialized reduction anyway.
    int64_t threadsRequired = 1;
    for (int64_t rDim : reductionDims) {
      // The size or #lanes needs to be a power of 2.
      threadsRequired *= llvm::PowerOf2Ceil(subgroupTileLens[rDim]);
    }
    std::optional<int64_t> availableThreads;
    int64_t threadStride = 0;
    for (int64_t rDim : reductionDims) {
      // TODO: We could merge two different dimension threads into one, but they
      // can be disjoint.
      if (threadTileLens[rDim] >= threadsRequired) {
        availableThreads = threadTileLens[rDim];
        threadStride = threadStrides[rDim];
        break;
      }
    }

    for (int64_t rDim : reductionDims) {
      batchTileLens[rDim] = 1;
      outerTileLens[rDim] = 1;
      elementTileLens[rDim] = 1;
      if (availableThreads.has_value()) {
        int64_t used = llvm::PowerOf2Ceil(subgroupTileLens[rDim]);
        threadStrides[rDim] = threadStride;
        threadTileLens[rDim] = used;
        availableThreads.value() /= used;
        threadStride *= used;
      } else {
        threadStrides[rDim] = 0;
        threadTileLens[rDim] = 1;
      }
      subgroupTileLens[rDim] = 1;
      subgroupStrides[rDim] = 0;
    }
    bufferReduceLayout = IREE::VectorExt::NestedLayoutAttr::get(
        srcLayout.getContext(), subgroupTileLens, batchTileLens, outerTileLens,
        threadTileLens, elementTileLens, subgroupStrides, threadStrides);
    return bufferReduceLayout;
  }

  void writePartialResultToBuffer(RewriterBase &rewriter, Location loc,
                                  VectorValue valueToWrite, Value buffer,
                                  NestedLayoutAttr srcLayout,
                                  ArrayRef<int64_t> reductionDims) const {
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    VectorType unDistributedType = valueToWrite.getType();
    SmallVector<Value> indices(unDistributedType.getRank(), c0);
    SmallVector<bool> inBounds(unDistributedType.getRank(), true);
    auto write = vector::TransferWriteOp::create(rewriter, loc, valueToWrite,
                                                 buffer, indices, inBounds);
    // Set layouts signature for write.
    // We need to set the layout on the srcVector/first operand.
    auto subgroupTileLens =
        llvm::to_vector_of<int64_t>(srcLayout.getSubgroupTile());
    auto batchTileLens = llvm::to_vector_of<int64_t>(srcLayout.getBatchTile());
    auto outerTileLens = llvm::to_vector_of<int64_t>(srcLayout.getOuterTile());
    auto threadTileLens =
        llvm::to_vector_of<int64_t>(srcLayout.getThreadTile());
    auto elementTileLens =
        llvm::to_vector_of<int64_t>(srcLayout.getElementTile());
    auto subgroupStrides =
        llvm::to_vector_of<int64_t>(srcLayout.getSubgroupStrides());
    auto threadStrides =
        llvm::to_vector_of<int64_t>(srcLayout.getThreadStrides());
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
    setSignatureForRedistribution(rewriter, write, {interSubGroupLayout}, {});
  }

  Value doSubgroupReductionFromBuffer(RewriterBase &rewriter, Location loc,
                                      Value buffer, NestedLayoutAttr srcLayout,
                                      VectorLayoutInterface resLayout,
                                      ArrayRef<int64_t> reductionDims,
                                      vector::CombiningKind kind,
                                      Value acc) const {
    NestedLayoutAttr readLayout =
        getLayoutForReductionFromBuffer(srcLayout, reductionDims);
    Value padValue = getCombiningIdentityValue(loc, rewriter, kind,
                                               getElementTypeOrSelf(buffer));
    auto readTy = VectorType::get(readLayout.getUndistributedShape(),
                                  getElementTypeOrSelf(buffer));
    auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto inBounds = rewriter.getBoolArrayAttr(
        SmallVector<bool>(readLayout.getRank(), true));
    auto mask = vector::CreateMaskOp::create(
        rewriter, loc, readTy.clone(rewriter.getI1Type()),
        memref::getMixedSizes(rewriter, loc, buffer));
    auto read = vector::TransferReadOp::create(
        rewriter, loc,
        /*vectorType=*/readTy,
        /*source=*/buffer,
        /*indices=*/SmallVector<Value>(readLayout.getRank(), zero),
        /*permMap=*/rewriter.getMultiDimIdentityMap(readLayout.getRank()),
        /*padding=*/padValue,
        /*mask=*/mask,
        /*inBounds=*/inBounds);
    setSignatureForRedistribution(rewriter, mask, {}, {readLayout});
    setSignatureForRedistribution(rewriter, read, {readLayout}, {readLayout});
    // A newly created reduction to complete the reduction
    // that reduces the data that was otherwise was on
    // different subgroups.
    // Since the data was distributed to every thread, it will
    // form a gpu.subgroup_reduce operation later.
    auto secondReduction = vector::MultiDimReductionOp::create(
        rewriter, loc, kind, read, acc, reductionDims);
    if (resLayout) {
      setSignatureForRedistribution(rewriter, secondReduction,
                                    {readLayout, resLayout}, {resLayout});
    } else {
      setSignatureForRedistribution(rewriter, secondReduction, {readLayout},
                                    {});
    }
    return secondReduction.getResult();
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
    Value isoRankThreadReduced = vector::ShapeCastOp::create(
        rewriter, loc, partialReducedDistributedType, threadReduced);

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
    auto unDistributedType = VectorType::get(
        partialReductionShape, srcVector.getType().getElementType());
    VectorValue valueToWrite = IREE::VectorExt::ToSIMDOp::create(
        rewriter, loc, unDistributedType, isoRankThreadReduced);

    auto workgroupMemoryAddressSpace = Attribute(gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::AddressSpace::Workgroup));
    MemRefType allocType = MemRefType::get(
        partialReductionShape, srcVector.getType().getElementType(),
        AffineMap(), workgroupMemoryAddressSpace);
    auto alloc =
        getBufferForSubgroupReduction(rewriter, allocType, valueToWrite);
    writePartialResultToBuffer(rewriter, loc, valueToWrite, alloc, srcLayout,
                               reductionDims);
    // Wait for writes to buffer to finish.
    gpu::BarrierOp::create(rewriter, loc, alloc);
    return doSubgroupReductionFromBuffer(rewriter, loc, alloc, srcLayout,
                                         resLayout, reductionDims, kind, acc);
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
struct DistributeContract final
    : MaskedOpDistributionPattern<vector::ContractionOp> {
  using MaskedOpDistributionPattern::MaskedOpDistributionPattern;

  DistributeContract(MLIRContext *context, int64_t benefit = 1)
      : MaskedOpDistributionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(vector::ContractionOp contractOp,
                  DistributionSignature &signature, vector::MaskOp maskOp,
                  std::optional<DistributionSignature> &maskSignature,
                  PatternRewriter &rewriter) const override {
    Location loc = contractOp.getLoc();
    FailureOr<VectorContractOpInfo> maybeOpInfo =
        VectorContractOpInfo::inferFromIndexingMaps(
            contractOp.getIndexingMapsArray());
    if (failed(maybeOpInfo)) {
      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
    }
    // If mmaAttr exists, defer the lowering to use MMA.
    // Notify failure if the "iree.gpu.mma" intrinsic attribute is present.
    auto mmaAttr =
        contractOp->getAttrOfType<IREE::GPU::MmaInterfaceAttr>("iree.gpu.mma");
    if (mmaAttr) {
      return rewriter.notifyMatchFailure(
          contractOp, "iree.gpu.mma intrinsic attribute exists");
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

    VectorValue mask = nullptr;
    if (maskOp) {
      auto maskLayout = dyn_cast_if_present<NestedLayoutAttr>(
          maskSignature.value()[maskOp.getMask()]);
      if (!maskLayout) {
        return rewriter.notifyMatchFailure(maskOp,
                                           "expected nested layout attr");
      }
      mask = getDistributed(rewriter, maskOp.getMask(), maskLayout);
      Value passThruLhs = getCombiningIdentityValue(
          loc, rewriter, contractOp.getKind(), disLhs.getType());
      Value passThruRhs = getCombiningIdentityValue(
          loc, rewriter, contractOp.getKind(), disRhs.getType());

      VectorValue deInterleavedMask =
          getDeinterleavedUnpackedForm(rewriter, mask, maskLayout);
      VectorValue maskLhs = projectVector(rewriter, loc, deInterleavedMask,
                                          contractOp.getIndexingMapsArray()[0]);
      VectorValue interleavedMaskLhs =
          getInterleavedPackedForm(rewriter, maskLhs, lhsLayout);

      VectorValue maskRhs = projectVector(rewriter, loc, deInterleavedMask,
                                          contractOp.getIndexingMapsArray()[1]);
      VectorValue interleavedMaskRhs =
          getInterleavedPackedForm(rewriter, maskRhs, rhsLayout);

      disLhs = cast<VectorValue>(arith::SelectOp::create(rewriter, loc,
                                                         interleavedMaskLhs,
                                                         disLhs, passThruLhs)
                                     .getResult());
      disRhs = cast<VectorValue>(arith::SelectOp::create(rewriter, loc,
                                                         interleavedMaskRhs,
                                                         disRhs, passThruRhs)
                                     .getResult());
    }

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

    // Step 1: local contraction
    Value localInit = getCombiningIdentityValue(
        loc, rewriter, contractOp.getKind(), disAcc.getType());
    Value localContract = doDistributedContraction(
        rewriter, loc, ctx, contractOp, disLhs, disRhs, localInit);

    // TODO: As per current upstream lowering implementations, there is no point
    // in doing this because it does a select much later in a finer granularity
    // rather than supporting predication. Moreover, since we are doing a select
    // to cater reductions accross the distribution, we can choose not to mask
    // the op post-distribution.

    VectorValue localContractValue;
    if (accVector) {
      localContractValue = dyn_cast<VectorValue>(localContract);
    } else {
      VectorType vecType = VectorType::get(ArrayRef{int64_t(1)}, accElemTy);
      localContractValue =
          vector::BroadcastOp::create(rewriter, loc, vecType, localContract);
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
    Value shapeCasted = vector::ShapeCastOp::create(
        rewriter, loc, partialReducedDistributedType, localContractValue);
    VectorType unDistributedType =
        VectorType::get(reductionLayout.getUndistributedShape(),
                        localContractValue.getType().getElementType());
    Value undistrLocalReduced = IREE::VectorExt::ToSIMDOp::create(
        rewriter, loc, unDistributedType, shapeCasted);

    // Create the partial reduction
    auto partialReduction = vector::MultiDimReductionOp::create(
        rewriter, loc, contractOp.getKind(), undistrLocalReduced, acc,
        partialReductionDims);
    if (resVector) {
      setSignatureForRedistribution(rewriter, partialReduction,
                                    {reductionLayout, signature[resVector]},
                                    {signature[resVector]});
    } else {
      setSignatureForRedistribution(rewriter, partialReduction,
                                    {reductionLayout}, {});
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

    auto localContractOp = vector::ContractionOp::create(
        rewriter, loc, lhs, rhs, localInit,
        rewriter.getAffineMapArrayAttr(newMaps),
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
    VectorValue transposed = vector::TransposeOp::create(
        rewriter, transposeOp.getLoc(), input, permutation);
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

    auto interleaved = vector::TransposeOp::create(
        rewriter, loc, getDistributed(rewriter, input, layoutA),
        interleavePermutation);

    // Shape cast to match the new layout.

    SmallVector<int64_t> transposedShapeB(shapeB);
    applyPermutationToVector(transposedShapeB, interleavePermutation);
    Type reshapedType = VectorType::get(
        transposedShapeB, interleaved.getResultVectorType().getElementType());

    auto reshaped =
        vector::ShapeCastOp::create(rewriter, loc, reshapedType, interleaved);

    // Inverse transpose to preserve original order.
    SmallVector<int64_t> invertedPermutation =
        invertPermutationVector(interleavePermutation);

    auto layouted = vector::TransposeOp::create(rewriter, loc, reshaped,
                                                invertedPermutation);

    replaceOpWithDistributedValues(rewriter, toLayoutOp, layouted.getResult());
    return success();
  }
};

/// Distributes vector.step to:
///
/// %elements = vector.step
/// %batch_outer_strides = arith.constant
/// %base = %outer_elements + %element_step
/// %thread_contrib = %thread_id * %thread_stride
/// %subgroup_contrib = %subgroup_id * %subgroup_stride
/// %index = %base + %thread_contrib + %subgroup_contrib
struct DistributeStep final : OpDistributionPattern<vector::StepOp> {
  using OpDistributionPattern::OpDistributionPattern;

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

    // Packed shape is [S, B, O, T, E] for this rank-1 vector.
    SmallVector<int64_t> packed = resultLayout.getUndistributedPackedShape();
    int64_t B = packed[1], O = packed[2];
    int64_t T = packed[3], E = packed[4];

    // Row-major strides of [S, B, O, T, E].
    int64_t sgStride = B * O * T * E;
    int64_t batchStride = O * T * E;
    int64_t outerStride = T * E;
    int64_t threadStride = E;

    int64_t BO = B * O;
    IndexType indexType = rewriter.getIndexType();
    VectorType workType = VectorType::get({BO, E}, indexType);

    // Inner element step: preserves contiguity information.
    auto step =
        vector::StepOp::create(rewriter, loc, VectorType::get({E}, indexType));
    auto stepBcast = vector::BroadcastOp::create(rewriter, loc, workType, step);

    // Outer (b, o) offsets constant.
    SmallVector<APInt> outerOffsets;
    outerOffsets.reserve(BO);
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t o = 0; o < O; ++o) {
        outerOffsets.push_back(APInt(64, b * batchStride + o * outerStride));
      }
    }
    auto outerConst = arith::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(VectorType::get({BO}, indexType), outerOffsets));
    auto outer2d = vector::ShapeCastOp::create(
        rewriter, loc, VectorType::get({BO, 1}, indexType), outerConst);
    auto outerBcast =
        vector::BroadcastOp::create(rewriter, loc, workType, outer2d);

    // Combine: base = outer_offsets + element_step.
    Value val = arith::AddIOp::create(rewriter, loc, outerBcast, stepBcast);

    // Add runtime subgroup and thread contributions.
    for (auto [id, stride] : {std::pair{subgroupIndices[0], sgStride},
                              std::pair{threadIndices[0], threadStride}}) {
      if (stride == 0) {
        continue;
      }
      auto strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
      auto scaled = arith::MulIOp::create(rewriter, loc, strideVal, id);
      auto bcast = vector::BroadcastOp::create(rewriter, loc, workType, scaled);
      val = arith::AddIOp::create(rewriter, loc, val, bcast);
    }

    // Shape-cast from [B*O, E] to distributed shape [B, O, E].
    SmallVector<int64_t> distributedShape =
        signature[result].getDistributedShape();
    VectorType distType =
        VectorType::get(distributedShape, result.getType().getElementType());
    auto shaped = vector::ShapeCastOp::create(rewriter, loc, distType, val);
    replaceOpWithDistributedValues(rewriter, stepOp, {shaped});
    return success();
  }

  Value threadId;
  int64_t subgroupSize;
};

SmallVector<Value> createDistributedMaskBounds(PatternRewriter &rewriter,
                                               Location loc,
                                               ValueRange upperBounds,
                                               NestedLayoutAttr layout,
                                               ArrayRef<Value> subgroupIndices,
                                               ArrayRef<Value> threadIndices) {
  constexpr int64_t subgroupIdx = 0;
  constexpr int64_t batchIdx = 1;
  constexpr int64_t outerIdx = 2;
  constexpr int64_t threadIdx = 3;
  constexpr int64_t elementIdx = 4;
  SmallVector<Value> bounds;
  auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);

  for (auto [unDistributedDim, upperBound] : llvm::enumerate(upperBounds)) {
    SmallVector<int64_t> undistributedShape =
        layout.getPackedShapeForUndistributedDim(unDistributedDim);
    std::array<int64_t, 3> distrShape{undistributedShape[batchIdx],
                                      undistributedShape[outerIdx],
                                      undistributedShape[elementIdx]};
    int64_t elementPerThread = ShapedType::getNumElements(distrShape);
    auto allValid =
        arith::ConstantIndexOp::create(rewriter, loc, elementPerThread);
    int64_t elementTileSize = distrShape.back();
    auto elementTileLastIdx =
        arith::ConstantIndexOp::create(rewriter, loc, elementTileSize - 1);

    // A special condition if the pre-distribution bounds match
    // the mask dimension length, then the distributed bounds
    // should exhibit the same property.

    APInt constUpperBound;
    if (matchPattern(upperBound.getDefiningOp(),
                     m_ConstantInt(&constUpperBound))) {
      int64_t undistributedDimLen =
          ShapedType::getNumElements(undistributedShape);
      if (constUpperBound.getZExtValue() == undistributedDimLen) {
        bounds.push_back(allValid);
        continue;
      }
    }
    auto lastValidIdx = arith::SubIOp::create(rewriter, loc, upperBound, one);
    auto delineraizedLastValidIdx = affine::AffineDelinearizeIndexOp::create(
        rewriter, loc, lastValidIdx, undistributedShape);
    SmallVector<Value> packedLastValidIdx =
        delineraizedLastValidIdx.getResults();

    // When subgroup id is equal to the subgroup that encounters the bound,
    // Every [vtid] less than [vtid that encounters last valid element] should
    // have a all valid element tile
    auto linearizedLastValidIdxPreThreads =
        affine::AffineLinearizeIndexOp::create(
            rewriter, loc,
            ValueRange{packedLastValidIdx[batchIdx],
                       packedLastValidIdx[outerIdx], elementTileLastIdx},
            distrShape);
    // Bound is defined as lastIdx + 1;
    auto distrUpperBoundPreThreads = arith::AddIOp::create(
        rewriter, loc, linearizedLastValidIdxPreThreads, one);

    auto linearizedLastValidIdx = affine::AffineLinearizeIndexOp::create(
        rewriter, loc,
        ValueRange{packedLastValidIdx[batchIdx], packedLastValidIdx[outerIdx],
                   packedLastValidIdx[elementIdx]},
        distrShape);
    auto distrUpperBound =
        arith::AddIOp::create(rewriter, loc, linearizedLastValidIdx, one);

    // The following code constructs a selection tree
    // that in effect follows the code:
    // * upperbound --> delinearize --> u0, u1, u2, u3, u4
    //
    // if sg < u0,
    //   all valid.
    // elif sg > u0,
    //   all invalid.
    // elif sg == u0,
    //   if tid < u3:
    //     [u1][u2][max]
    //   if tid > u3:
    //     all invalid.
    //   if tid == u3:
    //     [u1][u2][u4]

    // tid == u3
    auto cmpBoundTidEq = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq,
        threadIndices[unDistributedDim], packedLastValidIdx[threadIdx]);
    // tid < u3
    auto cmpBoundTidSlt = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt,
        threadIndices[unDistributedDim], packedLastValidIdx[threadIdx]);
    // sg == u0
    auto cmpBoundSgEq = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq,
        subgroupIndices[unDistributedDim], packedLastValidIdx[subgroupIdx]);
    // sg < u0
    auto cmpBoundSgSlt = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt,
        subgroupIndices[unDistributedDim], packedLastValidIdx[subgroupIdx]);

    // selectTid0 = tid < u3 ? [u1][u2][max] : all invalid
    auto selectTid0 = arith::SelectOp::create(rewriter, loc, cmpBoundTidSlt,
                                              distrUpperBoundPreThreads, zero);
    // selectTid1 = tid == u3 : [u1][u2][u4] : selectTid0
    auto selectTid1 = arith::SelectOp::create(rewriter, loc, cmpBoundTidEq,
                                              distrUpperBound, selectTid0);
    // selectSg0 = sg < u0 ? all valid : all invalid
    auto selectSg0 =
        arith::SelectOp::create(rewriter, loc, cmpBoundSgSlt, allValid, zero);
    // selectSg1 = sg == u0 ? selectTid1 : selectSg0
    auto selectSg1 = arith::SelectOp::create(rewriter, loc, cmpBoundSgEq,
                                             selectTid1, selectSg0);
    bounds.push_back(selectSg1);
  }
  return bounds;
}

struct DistributeCreateMask final
    : OpDistributionPattern<vector::CreateMaskOp> {
  using OpDistributionPattern::OpDistributionPattern;
  DistributeCreateMask(MLIRContext *context, Value threadId,
                       int64_t subgroupSize)
      : OpDistributionPattern(context), threadId(threadId),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(vector::CreateMaskOp maskOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Location loc = maskOp.getLoc();
    VectorValue result = maskOp.getResult();
    NestedLayoutAttr resultLayout =
        dyn_cast<NestedLayoutAttr>(signature[result]);
    if (!resultLayout) {
      return rewriter.notifyMatchFailure(
          maskOp, "missing nested layout for step op result");
    }
    SmallVector<Value> subgroupIndices, threadIndices;
    if (failed(populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
                                            resultLayout, subgroupIndices,
                                            threadIndices))) {
      return rewriter.notifyMatchFailure(
          maskOp, "warp or thread tiles have overlapping strides");
    }

    SmallVector<Value> distributedBounds = createDistributedMaskBounds(
        rewriter, loc, maskOp.getOperands(), resultLayout, subgroupIndices,
        threadIndices);

    Type elemType = maskOp.getType().getElementType();
    auto distrUnpackedType =
        VectorType::get(resultLayout.getDistributedUnpackedShape(), elemType);
    auto distrMask = vector::CreateMaskOp::create(
        rewriter, loc, distrUnpackedType, distributedBounds);
    VectorValue interleavedDistrMask =
        getInterleavedPackedForm(rewriter, distrMask, resultLayout);
    replaceOpWithDistributedValues(rewriter, maskOp, {interleavedDistrMask});
    return success();
  }
  Value threadId;
  int64_t subgroupSize;
};

struct DistributeConstantMask final
    : OpDistributionPattern<vector::ConstantMaskOp> {
  using OpDistributionPattern::OpDistributionPattern;
  DistributeConstantMask(MLIRContext *context, Value threadId,
                         int64_t subgroupSize)
      : OpDistributionPattern(context), threadId(threadId),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(vector::ConstantMaskOp maskOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Location loc = maskOp.getLoc();
    VectorValue result = maskOp.getResult();
    NestedLayoutAttr resultLayout =
        dyn_cast<NestedLayoutAttr>(signature[result]);
    if (!resultLayout) {
      return rewriter.notifyMatchFailure(
          maskOp, "missing nested layout for step op result");
    }
    SmallVector<Value> subgroupIndices, threadIndices;
    if (failed(populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
                                            resultLayout, subgroupIndices,
                                            threadIndices))) {
      return rewriter.notifyMatchFailure(
          maskOp, "warp or thread tiles have overlapping strides");
    }

    SmallVector<Value> constOperands;
    for (int64_t size : maskOp.getMaskDimSizes()) {
      Value index = arith::ConstantIndexOp::create(rewriter, loc, size);
      constOperands.push_back(index);
    }

    SmallVector<Value> distributedBounds =
        createDistributedMaskBounds(rewriter, loc, constOperands, resultLayout,
                                    subgroupIndices, threadIndices);

    Type elemType = maskOp.getType().getElementType();
    auto distrUnpackedType =
        VectorType::get(resultLayout.getDistributedUnpackedShape(), elemType);
    auto distrMask = vector::CreateMaskOp::create(
        rewriter, loc, distrUnpackedType, distributedBounds);
    VectorValue interleavedDistrMask =
        getInterleavedPackedForm(rewriter, distrMask, resultLayout);
    replaceOpWithDistributedValues(rewriter, maskOp, {interleavedDistrMask});
    return success();
  }
  Value threadId;
  int64_t subgroupSize;
};

struct DistributeShapeCast final : OpDistributionPattern<vector::ShapeCastOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp shapeCast,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue src = shapeCast.getSource();
    VectorValue dst = shapeCast.getResult();
    NestedLayoutAttr srcLayout =
        dyn_cast_if_present<NestedLayoutAttr>(signature[src]);
    NestedLayoutAttr dstLayout =
        dyn_cast_if_present<NestedLayoutAttr>(signature[dst]);
    if (!srcLayout || !dstLayout) {
      return rewriter.notifyMatchFailure(shapeCast,
                                         "expected nested layout attr");
    }

    // unpack -> reshape -> pack
    VectorValue unpacked = getDeinterleavedUnpackedForm(
        rewriter, getDistributed(rewriter, shapeCast.getSource(), srcLayout),
        srcLayout);
    VectorValue reshaped = vector::ShapeCastOp::create(
        rewriter, shapeCast.getLoc(),
        VectorType::get(dstLayout.getUndistributedShape(),
                        unpacked.getType().getElementType()),
        unpacked);
    VectorValue packed =
        getInterleavedPackedForm(rewriter, reshaped, dstLayout);
    replaceOpWithDistributedValues(rewriter, shapeCast, packed);
    return success();
  }
};

/// Distributes an iree_codegen.inner_tiled operation to a distributed
/// iree_codegen.inner_tiled operation with distributed semantics.
///
/// The pattern assumes that the inputs have the correct layouts and distributes
/// the operations according to the operation intrinsic and semantics.
struct DistributeInnerTiled final
    : OpDistributionPattern<IREE::Codegen::InnerTiledOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(IREE::Codegen::InnerTiledOp tiledOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    if (tiledOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(tiledOp, "requires vector semantics");
    }
    auto semantics =
        dyn_cast<IREE::GPU::InnerTiledSemanticsAttr>(tiledOp.getSemantics());
    if (!semantics) {
      return rewriter.notifyMatchFailure(tiledOp,
                                         "expected GPU inner tiled semantics");
    }
    if (semantics.getDistributed()) {
      return rewriter.notifyMatchFailure(tiledOp,
                                         "already distributed inner tiled op");
    }

    Location loc = tiledOp.getLoc();
    MLIRContext *ctx = tiledOp.getContext();

    // Get layouts, distributed forms, and unpacked forms for all inputs.
    SmallVector<Value> unpackedInputs;
    for (Value input : tiledOp.getInputs()) {
      auto vec = dyn_cast<VectorValue>(input);
      if (!vec) {
        return rewriter.notifyMatchFailure(tiledOp,
                                           "expected vector-typed input");
      }
      auto layout = dyn_cast_if_present<NestedLayoutAttr>(signature[vec]);
      if (!layout) {
        return rewriter.notifyMatchFailure(tiledOp,
                                           "missing nested layout for input");
      }
      VectorValue dist = getDistributed(rewriter, vec, layout);
      unpackedInputs.push_back(
          getDeinterleavedUnpackedForm(rewriter, dist, layout));
    }

    // Get layouts, distributed forms, and unpacked forms for all outputs.
    SmallVector<Value> unpackedOutputs;
    SmallVector<NestedLayoutAttr> resultLayouts;
    for (Value output : tiledOp.getOutputs()) {
      auto vec = dyn_cast<VectorValue>(output);
      if (!vec) {
        return rewriter.notifyMatchFailure(tiledOp,
                                           "expected vector-typed output");
      }
      auto layout = dyn_cast_if_present<NestedLayoutAttr>(signature[vec]);
      if (!layout) {
        return rewriter.notifyMatchFailure(tiledOp,
                                           "missing nested layout for output");
      }
      VectorValue dist = getDistributed(rewriter, vec, layout);
      unpackedOutputs.push_back(
          getDeinterleavedUnpackedForm(rewriter, dist, layout));
    }

    // Collect result layouts.
    for (Value result : tiledOp->getResults()) {
      auto vec = dyn_cast<VectorValue>(result);
      if (!vec) {
        return rewriter.notifyMatchFailure(tiledOp,
                                           "expected vector-typed result");
      }
      auto layout = dyn_cast_if_present<NestedLayoutAttr>(signature[vec]);
      if (!layout) {
        return rewriter.notifyMatchFailure(tiledOp,
                                           "missing nested layout for result");
      }
      resultLayouts.push_back(layout);
    }

    // Create distributed inner_tiled op.
    auto newSemantics = IREE::GPU::InnerTiledSemanticsAttr::get(
        ctx, /*distributed=*/true, semantics.getOpaque());

    auto newOp = IREE::Codegen::InnerTiledOp::create(
        rewriter, loc, unpackedInputs, unpackedOutputs,
        tiledOp.getIndexingMaps(), tiledOp.getIteratorTypes(),
        tiledOp.getKind(), newSemantics, tiledOp.getPermutations());

    // Convert results back to interleaved distributed form.
    SmallVector<Value> distributedResults;
    for (auto [result, layout] :
         llvm::zip_equal(newOp->getResults(), resultLayouts)) {
      distributedResults.push_back(getInterleavedPackedForm(
          rewriter, cast<VectorValue>(result), layout));
    }

    replaceOpWithDistributedValues(rewriter, tiledOp, distributedResults);
    return success();
  }
};

} // namespace

void populateGPUDistributeNestedLayoutAttrPatterns(
    RewritePatternSet &patterns, Value threadId, int64_t subgroupSize,
    ArrayRef<int64_t> workgroupSize, int64_t maxBitsPerShuffle) {
  patterns.add<DistributeTransferRead, DistributeTransferGather,
               DistributeMapScatter>(patterns.getContext(), threadId,
                                     subgroupSize);
  patterns.add<DistributeTransferWrite>(patterns.getContext(), threadId,
                                        subgroupSize, workgroupSize);
  patterns.add<DistributeBroadcast, DistributeTranspose, DistributeShapeCast>(
      patterns.getContext());
  patterns.add<DistributeMultiReduction>(patterns.getContext(), subgroupSize,
                                         maxBitsPerShuffle);
  patterns.add<DistributeContract>(patterns.getContext());
  patterns.add<DistributeBatchOuterToLayoutConversions>(patterns.getContext());
  patterns.add<DistributeInnerTiled>(patterns.getContext());
  patterns.add<DistributeStep>(patterns.getContext(), threadId, subgroupSize);
  patterns.add<DistributeCreateMask, DistributeConstantMask>(
      patterns.getContext(), threadId, subgroupSize);
}

}; // namespace mlir::iree_compiler
