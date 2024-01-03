// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
    // When there's more than 1 use of the result of tensor.extract_slice,
    // folding extractOp + xferOp won't eliminate the need for the former. So
    // this transformation wouldn't be beneficial. It would, however, generate
    // another "user" of the original tensor, which then would prevent the
    // hoisting logic to hoist the following pair in the example below:
    //  * "tensor.extract_slice" + "tensor.insert_slice"
    // ```mlir
    // %res = scf.for {{.*}} iter_args(%loop_arg = %extracted_slice) -> {{.*}} {
    //     %extracted_slice = tensor.extract_slice %loop_arg
    //     %read = vector.transfer_read %extracted_slice
    //     // Some opeeration using %read that generates %res
    //     %write = vector.transfer_write %res, %extracted_slice
    //     %inserted_slice = tensor.insert_slice %write into %loop_arg
    //     scf.yield %inserted_slice
    //  }
    //```
    for (auto user : extractOp.getResult().getUsers()) {
      if (user != xferOp)
        return failure();
    }

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

} // namespace

void mlir::iree_compiler::populateVectorTransferTensorSliceTransforms(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns
      .add<FoldExtractSliceIntoTransferRead, FoldInsertSliceIntoTransferWrite>(
          patterns.getContext(), benefit);
}
