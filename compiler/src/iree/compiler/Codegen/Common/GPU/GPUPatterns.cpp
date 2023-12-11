// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"

#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

namespace {
/// Applies tranformation to drop unit dims in destination vector.transfer_read
/// destination so that the resulting vector is 2D.
//
/// Example:
/// ```
/// %cst = arith.constant 0.000000e+00 : f32
/// %c2 = arith.constant 2 : index
/// %c3 = arith.constant 3 : index
/// %c4 = arith.constant 4 : index
/// %0 = vector.transfer_read %a[%c2, %c3, %c4], %cst
///        {in_bounds = [true, true, true]}
///      : memref<128x16x256xf32>, vector<16x1x8xf32>
/// ```
/// Will be converted into:
/// ```
/// #map = affine_map<(d0, d1) -> (d0 * 4096 + d1 + 8964)>
/// %c0 = arith.constant 0 : index
/// %cst = arith.constant 0.000000e+00 : f32
/// %0 = memref.subview %arg0[2, 3, 4] [16, 1, 8] [1, 1, 1]
///      : memref<128x16x256xf32> to memref<16x8xf32, #map>
/// %1 = vector.transfer_read %0[%c0, %c0], %cst {in_bounds = [true, true]}
///      : memref<16x8xf32, #map>, vector<16x8xf32>
/// %2 = vector.broadcast %1 : vector<16x8xf32> to  vector<1x16x8xf32>
/// %3 = vector.transpose %2, [1, 0, 2]
///      : vector<1x16x8xf32> to vector<16x1x8xf32>
/// ```
struct FlattenTransferReadOp : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferReadOp.getLoc();
    Value vector = transferReadOp.getVector();
    VectorType vectorType = llvm::cast<VectorType>(vector.getType());
    Value source = transferReadOp.getSource();
    MemRefType sourceType = llvm::dyn_cast<MemRefType>(source.getType());
    // Contiguity check is valid on tensors only.
    if (!sourceType)
      return failure();
    // Already 2D or lower nothing to do.
    if (vectorType.getRank() < 3)
      return failure();
    // The innermost dim is always considered non-unit as it wont be dropped
    // Therefore, we initialize `numberOfNonUnitDims` to 1 and not 0
    int numberOfNonUnitDims = 1;
    // Track of the location of the outer non-unit dim in the source
    // vector e.g if vector<1x16x1x32> -> vector<16x32> here the outer non-unit
    // dim is the one with size 16 at index 1 in the source vector. We
    // initialize as: `indexOfOuterNonUnitDim` = vectorType.getRank() - 2 = 2,
    // which is the highest index it can have for any 4D shape, we then traverse
    // the source vector shape to update this value to `indexOfOuterNonUnitDim`
    // = 1. This works out nicely for a case like vector<1x1x1x32> ->
    // vector<1x32> where `numberOfNonUnitDims` is desired to be 2, as the unit
    // dim adjacent to the innermost dim is considered the outermost non-unit
    // dim for the rest of the pattern if an actual outer non-unit dim does not
    // exist
    int indexOfOuterNonUnitDim = vectorType.getRank() - 2;
    for (int i = 0; i < vectorType.getRank() - 1; i++) {
      if (vectorType.getShape()[i] != 1) {
        numberOfNonUnitDims++;
        indexOfOuterNonUnitDim = i;
      }
    }
    // Bail out if 2D vector cannot be formed
    if (numberOfNonUnitDims > 2) {
      return failure();
    }
    int rankOfCollapsedVector = 2;
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim())
      return failure();
    if (!transferReadOp.getPermutationMap().isMinorIdentity())
      return failure();
    if (transferReadOp.getMask())
      return failure();
    ArrayAttr newInBoundsAttr = rewriter.getBoolArrayAttr(
        SmallVector<bool>(rankOfCollapsedVector, true));
    auto newidentityMap =
        rewriter.getMultiDimIdentityMap(rankOfCollapsedVector);

    SmallVector<int64_t> vectorShapeCollapse = {
        vectorType.getShape()[indexOfOuterNonUnitDim],
        vectorType.getShape()[vectorType.getRank() - 1]};
    SmallVector<int64_t> vectorShapeBroadcast = vectorShapeCollapse;
    for (int i = 0; i < vectorType.getRank() - rankOfCollapsedVector; i++) {
      vectorShapeBroadcast.insert(vectorShapeBroadcast.begin(), 1);
    }

    VectorType vectorTypeCollapse =
        VectorType::get(vectorShapeCollapse, vectorType.getElementType());
    VectorType vectorTypeBroadcast =
        VectorType::get(vectorShapeBroadcast, vectorType.getElementType());

    SmallVector<OpFoldResult> subViewOffsets, subViewSizes, subViewStrides;
    subViewSizes.append(sourceType.getRank() - vectorType.getRank(),
                        rewriter.getIndexAttr(1));
    for (int64_t dim : vectorType.getShape())
      subViewSizes.push_back(rewriter.getIndexAttr(dim));
    for (int i = 0; i < sourceType.getRank(); i++) {
      subViewOffsets.push_back(transferReadOp.getIndices()[i]);
      subViewStrides.push_back(rewriter.getIndexAttr(1));
    }
    MemRefType resultType =
        llvm::cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
            vectorShapeCollapse, sourceType, subViewOffsets, subViewSizes,
            subViewStrides));
    Value subView = rewriter.create<memref::SubViewOp>(
        loc, resultType, source, subViewOffsets, subViewSizes, subViewStrides);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value readCollapse = rewriter.create<vector::TransferReadOp>(
        loc, vectorTypeCollapse, subView, ValueRange{c0, c0}, newidentityMap,
        transferReadOp.getPadding(), transferReadOp.getMask(), newInBoundsAttr);

    Value readBroadcast = rewriter.create<vector::BroadcastOp>(
        loc, vectorTypeBroadcast, readCollapse);
    SmallVector<int64_t> tranposePermutation;
    for (int i = 0; i < vectorType.getRank(); i++) {
      if (i == vectorType.getRank() - 2)
        continue;
      tranposePermutation.push_back(i);
    }
    tranposePermutation.insert(tranposePermutation.begin() +
                                   indexOfOuterNonUnitDim,
                               vectorType.getRank() - 2);
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(
        transferReadOp, readBroadcast, tranposePermutation);
    return success();
  }
};

// Merges transpose op into the transfer read op. Transpose are not supported on
// MMA types but MMA load can transpose the matrix when loading.
struct CombineTransferReadOpBroadcast final
    : public OpRewritePattern<vector::BroadcastOp> {
  using OpRewritePattern<vector::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto transferReadOp =
        op.getSource().getDefiningOp<vector::TransferReadOp>();
    if (!transferReadOp || transferReadOp.getMask() ||
        transferReadOp.hasOutOfBoundsDim()) {
      return failure();
    }
    int64_t rankDiff = op.getResultVectorType().getRank() -
                       transferReadOp.getVectorType().getRank();
    SmallVector<AffineExpr> exprs(rankDiff, rewriter.getAffineConstantExpr(0));
    ArrayRef<AffineExpr> originalExpr =
        transferReadOp.getPermutationMap().getResults();
    exprs.append(originalExpr.begin(), originalExpr.end());
    AffineMap newMap =
        AffineMap::get(transferReadOp.getPermutationMap().getNumDims(),
                       transferReadOp.getPermutationMap().getNumSymbols(),
                       exprs, op.getContext());
    ArrayAttr inBounds = rewriter.getBoolArrayAttr(
        SmallVector<bool>(op.getResultVectorType().getRank(), true));
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        op, op.getType(), transferReadOp.getSource(),
        transferReadOp.getIndices(), newMap, transferReadOp.getPadding(),
        transferReadOp.getMask(), inBounds);
    return success();
  }
};

template <typename T>
using LinalgPromotionPattern =
    mlir::iree_compiler::IREE::LinalgExt::LinalgPromotionPattern<T>;

/// Returns true if op is appropriate contract for promotion.
static LogicalResult contractOpFilter(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return failure();
  // Limit promotion to matmul and batch matmul, there may be generic
  // ops with more batch dimensions we didn't distribute and therefore
  // cannot find a higher bound.
  return success(
      linalg::isaContractionOpInterface(cast<linalg::LinalgOp>(op)) &&
      linalgOp.getNumParallelLoops() >= 2 &&
      linalgOp.getNumParallelLoops() <= 3);
}

// A `dealloc` is converted into a call to `free` on the underlying data buffer.
// The memref descriptor being an SSA value, there is no need to clean it up
// in any way.
struct DropSharedMemoryDeallocOp : public OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasSharedMemoryAddressSpace(
            llvm::cast<MemRefType>(op.getMemref().getType())))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateVectorTransferToGPUMMAPreparationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FlattenTransferReadOp>(patterns.getContext());
}

void populateCombineVectorTransferReadBroadcastPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CombineTransferReadOpBroadcast>(patterns.getContext());
}

void populateContractPromotionPatterns(RewritePatternSet &patterns,
                                       ArrayRef<int64_t> operandsToPromote) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<LinalgPromotionPattern<linalg::MatmulOp>,
                  LinalgPromotionPattern<linalg::BatchMatmulOp>,
                  LinalgPromotionPattern<linalg::GenericOp>>(
      context,
      linalg::LinalgPromotionOptions()
          .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                        deallocateWorkgroupMemory)
          .setCopyInOutFns(copyToWorkgroupMemory, copyToWorkgroupMemory)
          .setOperandsToPromote(operandsToPromote)
          .setUseFullTileBuffers({false, false}),
      IREE::LinalgExt::LinalgTransformationFilter(
          {StringAttr::get(context, getWorkgroupKTiledMarker())},
          StringAttr::get(context, getWorkgroupMemoryMarker()))
          .setMatchByDefault()
          .addFilter(contractOpFilter));
}

void populateDropSharedMemoryDeallocOpPatterns(RewritePatternSet &patterns) {
  patterns.add<DropSharedMemoryDeallocOp>(patterns.getContext());
}

} // namespace mlir::iree_compiler
