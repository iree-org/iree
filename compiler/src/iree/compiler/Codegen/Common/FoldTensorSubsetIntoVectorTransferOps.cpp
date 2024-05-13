// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

/// Returns true if all rank reduced in the given `extractOp` happen in leading
/// dimensions earlier than last `trailingRank` dimensions.
static bool areAllRankReducedLeadingDim(tensor::ExtractSliceOp extractOp,
                                        unsigned trailingRank) {
  // If no ranks are reduced at all, it's a degenerated case; always true.
  if (extractOp.getSourceType().getRank() == extractOp.getType().getRank())
    return true;

  RankedTensorType inferredType = extractOp.inferResultType(
      extractOp.getSourceType(), extractOp.getMixedOffsets(),
      extractOp.getMixedSizes(), extractOp.getMixedStrides());
  return extractOp.getType().getShape().take_back(trailingRank) ==
         inferredType.getShape().take_back(trailingRank);
}

namespace {
/// Fold transfer_reads of a tensor.extract_slice op. E.g.:
///
/// ```
/// %0 = tensor.extract_slice %t[%a, %b] [%c, %d] [1, 1]
///     : tensor<?x?xf32> to tensor<?x?xf32>
/// %1 = vector.transfer_read %0[%e, %f], %cst {in_bounds = [true, true]}
///     : tensor<?x?xf32>, vector<4x5xf32>
/// ```
/// is rewritten to:
/// ```
/// %p0 = arith.addi %a, %e : index
/// %p1 = arith.addi %b, %f : index
/// %1 = vector.transfer_read %t[%p0, %p1], %cst {in_bounds = [true, true]}
///     : tensor<?x?xf32>, vector<4x5xf32>
/// ```
// TODO: this is brittle and should be deprecated in favor of a more general
// pattern that applies on-demand.
class FoldExtractSliceIntoTransferRead final
    : public OpRewritePattern<vector::TransferReadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp xferOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (xferOp.getTransferRank() == 0)
      return failure();
    if (xferOp.hasOutOfBoundsDim())
      return failure();
    if (!xferOp.getPermutationMap().isMinorIdentity())
      return failure();
    if (xferOp.getMask())
      return failure();
    auto extractOp = xferOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp)
      return failure();
    if (!extractOp.hasUnitStride())
      return failure();

    // Bail on illegal rank-reduction: we need to check that the rank-reduced
    // dims are exactly the leading dims. I.e. the following is illegal:
    // ```
    //    %0 = tensor.extract_slice %t[0,0,0][2,1,4][1,1,1] :
    //      tensor<2x1x4xf32> to tensor<2x4xf32>
    //    %1 = vector.transfer_read %0[0,0], %cst :
    //      tensor<2x4xf32>, vector<2x4xf32>
    // ```
    //
    // Cannot fold into:
    // ```
    //    %0 = vector.transfer_read %t[0,0,0], %cst :
    //      tensor<2x1x4xf32>, vector<2x4xf32>
    // ```
    // For this, check the trailing `vectorRank` dims of the extract_slice
    // result tensor match the trailing dims of the inferred result tensor.
    if (!areAllRankReducedLeadingDim(extractOp, extractOp.getType().getRank()))
      return failure();

    int64_t rankReduced =
        extractOp.getSourceType().getRank() - extractOp.getType().getRank();

    SmallVector<Value> newIndices;
    // In case this is a rank-reducing ExtractSliceOp, copy rank-reduced
    // indices first.
    for (int64_t i = 0; i < rankReduced; ++i) {
      OpFoldResult offset = extractOp.getMixedOffsets()[i];
      newIndices.push_back(getValueOrCreateConstantIndexOp(
          rewriter, extractOp.getLoc(), offset));
    }
    for (const auto &it : llvm::enumerate(xferOp.getIndices())) {
      OpFoldResult offset =
          extractOp.getMixedOffsets()[it.index() + rankReduced];
      newIndices.push_back(rewriter.create<arith::AddIOp>(
          xferOp->getLoc(), it.value(),
          getValueOrCreateConstantIndexOp(rewriter, extractOp.getLoc(),
                                          offset)));
    }
    SmallVector<bool> inBounds(xferOp.getTransferRank(), true);
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        xferOp, xferOp.getVectorType(), extractOp.getSource(), newIndices,
        xferOp.getPadding(), ArrayRef<bool>{inBounds});

    return success();
  }
};

/// Fold tensor.insert_slice into vector.transfer_write if the transfer_write
/// could directly write to the insert_slice's destination. E.g.:
///
/// ```
/// %0 = vector.transfer_write %v, %t1[%c0, %c0] {in_bounds = [true, true]}
///     : vector<4x5xf32>, tensor<4x5xf32>
/// %1 = tensor.insert_slice %0 into %t2[%a, %b] [4, 5] [1, 1]
///     : tensor<4x5xf32> into tensor<?x?xf32>
/// ```
/// is rewritten to:
/// ```
/// %1 = vector.transfer_write %v, %t2[%a, %b] {in_bounds = [true, true]}
///     : vector<4x5xf32>, tensor<?x?xf32>
/// ```
// TODO: this is brittle and should be deprecated in favor of a more general
// pattern that applies on-demand.
class FoldInsertSliceIntoTransferWrite final
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (!insertOp.hasUnitStride())
      return failure();

    auto xferOp = insertOp.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!xferOp)
      return failure();

    // TODO: support 0-d corner case.
    if (xferOp.getTransferRank() == 0)
      return failure();

    if (xferOp.hasOutOfBoundsDim())
      return failure();
    if (xferOp.getVectorType().getRank() != xferOp.getShapedType().getRank())
      return failure();
    if (xferOp.getMask())
      return failure();
    // Fold only if the TransferWriteOp completely overwrites the `source` with
    // a vector. I.e., the result of the TransferWriteOp is a new tensor whose
    // content is the data of the vector.
    if (!llvm::equal(xferOp.getVectorType().getShape(),
                     xferOp.getShapedType().getShape()))
      return failure();
    if (!xferOp.getPermutationMap().isIdentity())
      return failure();

    // Bail on illegal rank-reduction: we need to check that the rank-reduced
    // dims are exactly the leading dims. I.e. the following is illegal:
    // ```
    //    %0 = vector.transfer_write %v, %t[0,0], %cst :
    //      vector<2x4xf32>, tensor<2x4xf32>
    //    %1 = tensor.insert_slice %0 into %tt[0,0,0][2,1,4][1,1,1] :
    //      tensor<2x4xf32> into tensor<2x1x4xf32>
    // ```
    //
    // Cannot fold into:
    // ```
    //    %0 = vector.transfer_write %v, %t[0,0,0], %cst :
    //      vector<2x4xf32>, tensor<2x1x4xf32>
    // ```
    // For this, check the trailing `vectorRank` dims of the insert_slice result
    // tensor match the trailing dims of the inferred result tensor.
    int64_t rankReduced =
        insertOp.getType().getRank() - insertOp.getSourceType().getRank();
    int64_t vectorRank = xferOp.getVectorType().getRank();
    RankedTensorType inferredSourceTensorType =
        tensor::ExtractSliceOp::inferResultType(
            insertOp.getType(), insertOp.getMixedOffsets(),
            insertOp.getMixedSizes(), insertOp.getMixedStrides());
    auto actualSourceTensorShape = insertOp.getSourceType().getShape();
    if (rankReduced > 0 &&
        actualSourceTensorShape.take_back(vectorRank) !=
            inferredSourceTensorType.getShape().take_back(vectorRank))
      return failure();

    SmallVector<Value> indices = getValueOrCreateConstantIndexOp(
        rewriter, insertOp.getLoc(), insertOp.getMixedOffsets());
    SmallVector<bool> inBounds(xferOp.getTransferRank(), true);
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        insertOp, xferOp.getVector(), insertOp.getDest(), indices,
        ArrayRef<bool>{inBounds});
    return success();
  }
};

/// Fold tensor.extract_slice into vector.transfer_write if
///   1. The vector.transfer_write op has only one use.
///   2. All the offests of the tensor.extract_slice op are zeros.
///   3. The vector.transfer_write op does not have masks.
///   4. The vector.transfer_write op writes to a tensor.empty op.
///
/// E.g.:
///
/// ```
/// %0 = vector.transfer_write %v, %t[%a, %b, %c]
///   {in_bounds = [true, true, true]}
///   : vector<1x64x128xf16>, tensor<1x64x128xf16>
/// %extracted_slice = tensor.extract_slice %0[0, 0, 0] [1, %3, 128] [1, 1, 1]
///   : tensor<1x64x128xf16> to tensor<1x?x128xf16>
/// ```
/// is rewritten to:
/// ```
/// %1 = vector.transfer_write %v, %t2[%a, %b, %c]
///   {in_bounds = [true, false, true]}
///   : vector<1x64x128xf16>, tensor<1x?x128xf16>
/// ```
class FoldExtractSliceIntoTransferWrite final
    : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    if (extractSliceOp.getDroppedDims().any()) {
      return rewriter.notifyMatchFailure(
          extractSliceOp,
          "expect it is not a rank-reduced tensor.extract_slice op");
    }
    if (!llvm::all_of(extractSliceOp.getMixedOffsets(), isZeroIndex)) {
      return rewriter.notifyMatchFailure(extractSliceOp,
                                         "expect all the offsets are zeros");
    }

    auto xferOp =
        extractSliceOp.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!xferOp) {
      return rewriter.notifyMatchFailure(
          extractSliceOp, "expect the source is from transfer.vector_write op");
    }
    if (!xferOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(
          extractSliceOp,
          "expect the transfer.vector_write op has only one use");
    }
    if (!xferOp.getSource().getDefiningOp<tensor::EmptyOp>()) {
      return rewriter.notifyMatchFailure(
          extractSliceOp, "expect the transfer.vector_write op to write into a "
                          "tensor.empty op");
    }
    if (xferOp.getMask()) {
      return failure();
    }

    Location loc = extractSliceOp.getLoc();
    SmallVector<OpFoldResult> mixedSizes = extractSliceOp.getMixedSizes();
    auto init = rewriter.create<tensor::EmptyOp>(
        loc, mixedSizes, extractSliceOp.getType().getElementType());

    SmallVector<bool> inBounds;
    inBounds.resize(mixedSizes.size());
    for (auto [idx, vecSize, destSize] :
         llvm::zip_equal(llvm::seq<int64_t>(0, inBounds.size()),
                         xferOp.getVectorType().getShape(), mixedSizes)) {
      auto maybeCst = getConstantIntValue(destSize);
      if (!maybeCst) {
        inBounds[idx] = false;
        continue;
      }
      if (*maybeCst >= vecSize) {
        inBounds[idx] = false;
      } else {
        inBounds[idx] = true;
      }
    }

    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        extractSliceOp, xferOp.getVector(), init, xferOp.getIndices(),
        xferOp.getPermutationMap(), inBounds);

    return success();
  }
};

} // namespace

void mlir::iree_compiler::populateVectorTransferTensorSliceTransforms(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns
      .add<FoldExtractSliceIntoTransferRead, FoldInsertSliceIntoTransferWrite,
           FoldExtractSliceIntoTransferWrite>(patterns.getContext(), benefit);
}
