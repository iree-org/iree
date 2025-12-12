// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FUSETENSORTOBUFFERCONVERTERSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Helper function to get all PCF::WriteSliceOps from a PCF::GenericOp or
// PCF::LoopOp that write to a specific result.
template <typename PCFOpTy>
static FailureOr<SmallVector<IREE::PCF::WriteSliceOp>>
getProducerSlices(PCFOpTy pcfOp, OpResult result) {
  static_assert(std::is_same_v<PCFOpTy, IREE::PCF::GenericOp> ||
                    std::is_same_v<PCFOpTy, IREE::PCF::LoopOp>,
                "PCFOpTy must be PCF::GenericOp or PCF::LoopOp");
  BlockArgument tiedArg = pcfOp.getRegionRefArgs()[result.getResultNumber()];

  // The fusion is only valid if the sref type is parent only sync scope.
  auto srefType = dyn_cast<IREE::PCF::ShapedRefType>(tiedArg.getType());
  if (!srefType || !srefType.isReturnOnlySync()) {
    return failure();
  }

  // Collect all WriteSliceOps that use this argument. Skip ReadSliceOps but
  // fail if there are any other users.
  SmallVector<IREE::PCF::WriteSliceOp> writeSlices;
  for (Operation *user : tiedArg.getUsers()) {
    if (isa<IREE::PCF::ReadSliceOp>(user)) {
      continue;
    }
    auto writeSlice = dyn_cast<IREE::PCF::WriteSliceOp>(user);
    if (!writeSlice) {
      return failure();
    }
    writeSlices.push_back(writeSlice);
  }

  return writeSlices;
}

struct FuseStoreToBuffer
    : public OpRewritePattern<IREE::Codegen::StoreToBufferOp> {
  using OpRewritePattern<IREE::Codegen::StoreToBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Codegen::StoreToBufferOp storeOp,
                                PatternRewriter &rewriter) const override {
    Value tensor = storeOp.getTensor();

    Operation *definingOp = tensor.getDefiningOp();
    auto producerLoop = dyn_cast_if_present<IREE::PCF::LoopOp>(definingOp);
    auto producerGeneric =
        dyn_cast_if_present<IREE::PCF::GenericOp>(definingOp);
    if (!producerLoop && !producerGeneric) {
      return failure();
    }

    // Make sure that the buffer operand of the store dominates the producer
    // loop.
    DominanceInfo domInfo(definingOp);
    Value buffer = storeOp.getBuffer();
    if (!domInfo.dominates(buffer, definingOp)) {
      return failure();
    }

    // Get the write_slice ops that produce the result of written by the store.
    auto result = cast<OpResult>(tensor);
    FailureOr<SmallVector<IREE::PCF::WriteSliceOp>> maybeSlices = failure();
    if (producerLoop) {
      maybeSlices = getProducerSlices(producerLoop, result);
    } else {
      assert(producerGeneric && "unexpected undefined generic");
      maybeSlices = getProducerSlices(producerGeneric, result);
    }
    if (failed(maybeSlices)) {
      return failure();
    }

    SmallVector<IREE::PCF::WriteSliceOp> writeSlices = *maybeSlices;

    // For each WriteSliceOp, write its source to the store op's buffer.
    for (IREE::PCF::WriteSliceOp writeSlice : writeSlices) {
      rewriter.setInsertionPoint(writeSlice);

      // Create subview for the destination.
      Value destSlice = memref::SubViewOp::create(
          rewriter, writeSlice.getLoc(), buffer, writeSlice.getMixedOffsets(),
          writeSlice.getMixedSizes(), writeSlice.getMixedStrides());

      // Handle different source types. Create but don't replace the write_slice
      // ops. We rely on unused result cleanup patterns to drop them when
      // possible.
      Type sourceType = writeSlice.getSourceType();
      if (auto tensorType = dyn_cast<RankedTensorType>(sourceType)) {
        IREE::Codegen::StoreToBufferOp::create(
            rewriter, storeOp.getLoc(), writeSlice.getSource(), destSlice);
      } else if (auto memrefType = dyn_cast<MemRefType>(sourceType)) {
        memref::CopyOp::create(rewriter, storeOp.getLoc(),
                               writeSlice.getSource(), destSlice);
      } else if (auto vectorType = dyn_cast<VectorType>(sourceType)) {
        SmallVector<bool> inBounds(vectorType.getRank(), true);
        for (auto [inBound, vecSize, storeSize] :
             llvm::zip_equal(inBounds, vectorType.getShape(),
                             writeSlice.getStaticSizes())) {
          inBound = vecSize == storeSize;
        }
        SmallVector<Value> offsets(
            vectorType.getRank(),
            arith::ConstantIndexOp::create(rewriter, writeSlice.getLoc(), 0));
        vector::TransferWriteOp::create(rewriter, storeOp.getLoc(),
                                        writeSlice.getSource(), destSlice,
                                        offsets, inBounds);
      } else {
        assert(false && "Invalid write_slice operand type");
      }
    }

    rewriter.eraseOp(storeOp);
    return success();
  }
};

struct FuseDispatchTensorStore
    : public OpRewritePattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using OpRewritePattern<
      IREE::TensorExt::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Unimplemented: non-unit stride.
    if (!storeOp.hasUnitStride()) {
      return failure();
    }
    Value value = storeOp.getValue();
    Value target = storeOp.getTarget();

    Operation *definingOp = value.getDefiningOp();
    auto producerLoop = dyn_cast_if_present<IREE::PCF::LoopOp>(definingOp);
    auto producerGeneric =
        dyn_cast_if_present<IREE::PCF::GenericOp>(definingOp);
    if (!producerLoop && !producerGeneric) {
      return failure();
    }

    DominanceInfo domInfo(definingOp);
    if (!domInfo.dominates(target, definingOp)) {
      return failure();
    }

    // Get the write_slice ops that produce the result of written by the store.
    auto result = cast<OpResult>(value);
    FailureOr<SmallVector<IREE::PCF::WriteSliceOp>> maybeSlices = failure();
    if (producerLoop) {
      maybeSlices = getProducerSlices(producerLoop, result);
    } else {
      assert(producerGeneric && "unexpected undefined generic");
      maybeSlices = getProducerSlices(producerGeneric, result);
    }
    if (failed(maybeSlices)) {
      return failure();
    }

    SmallVector<IREE::PCF::WriteSliceOp> writeSlices = *maybeSlices;

    // Check that all source operands are tensors as that's the only type
    // that can be written to the special dispatch type. Also non-unit stride
    // is currently unsupported.
    if (!llvm::all_of(writeSlices, [](IREE::PCF::WriteSliceOp writeSlice) {
          return isa<RankedTensorType>(writeSlice.getSourceType()) &&
                 writeSlice.hasUnitStride();
        })) {
      return failure();
    }

    // For each WriteSliceOp, create a new DispatchTensorStoreOp of just the
    // written slice.
    AffineExpr d0, d1;
    bindSymbols(rewriter.getContext(), d0, d1);
    AffineExpr add = d0 + d1;
    for (IREE::PCF::WriteSliceOp writeSlice : writeSlices) {
      rewriter.setInsertionPoint(writeSlice);

      // Add the offsets of the WriteSliceOp to the offsets of the store.
      SmallVector<OpFoldResult> newOffsets;
      SmallVector<OpFoldResult> writeOffsets = writeSlice.getMixedOffsets();
      SmallVector<OpFoldResult> storeOffsets = storeOp.getMixedOffsets();

      for (auto [writeOffset, storeOffset] :
           llvm::zip_equal(writeOffsets, storeOffsets)) {
        newOffsets.push_back(affine::makeComposedFoldedAffineApply(
            rewriter, writeSlice.getLoc(), add, {writeOffset, storeOffset}));
      }

      // Get the source to store. If the write_slice source is a rank-reducing
      // insert_slice into tensor.empty, use the insert_slice source directly.
      Value sourceToStore = writeSlice.getSource();
      SmallVector<OpFoldResult> sizesToStore = writeSlice.getMixedSizes();
      if (auto insertOp =
              sourceToStore.getDefiningOp<tensor::InsertSliceOp>()) {
        if (insertOp.getDest().getDefiningOp<tensor::EmptyOp>()) {
          auto insertSourceType = insertOp.getSourceType();
          auto insertResultType = insertOp.getResultType();
          if (isRankReducedType(insertResultType, insertSourceType) ==
              SliceVerificationResult::Success) {
            sourceToStore = insertOp.getSource();
            sizesToStore = insertOp.getMixedSizes();
          }
        }
      }

      // Use the sizes of the source tensor.
      IREE::TensorExt::DispatchTensorStoreOp::create(
          rewriter, storeOp.getLoc(), sourceToStore, target,
          storeOp.getTargetDims(), newOffsets, sizesToStore,
          storeOp.getMixedStrides());
    }

    rewriter.eraseOp(storeOp);
    return success();
  }
};

/// Pattern to fuse tensor.expand_shape ops that only introduce unit dimensions
/// into iree_codegen.store_to_buffer by introducing a memref.collapse_shape.
struct FuseRankIncreasingExpandIntoBufferStore
    : public OpRewritePattern<IREE::Codegen::StoreToBufferOp> {
  using OpRewritePattern<IREE::Codegen::StoreToBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Codegen::StoreToBufferOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Check if the tensor is produced by an expand_shape.
    auto expandOp = storeOp.getTensor().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp) {
      return failure();
    }

    // Check that the expand_shape only introduces unit dimensions.
    if (isRankReducedType(expandOp.getResultType(), expandOp.getSrcType()) !=
        SliceVerificationResult::Success) {
      return failure();
    }

    SmallVector<ReassociationIndices> reassociation =
        expandOp.getReassociationIndices();

    // Create a collapse_shape on the buffer to drop the unit dimensions.
    Value buffer = storeOp.getBuffer();
    auto bufferType = cast<MemRefType>(buffer.getType());

    // Compute the collapsed type accounting for memref layout.
    MemRefType collapsedType = memref::CollapseShapeOp::computeCollapsedType(
        bufferType, reassociation);

    Value collapsedBuffer = memref::CollapseShapeOp::create(
        rewriter, storeOp.getLoc(), collapsedType, buffer, reassociation);

    // Create a new store_to_buffer with the expand_shape source and collapsed
    // buffer.
    IREE::Codegen::StoreToBufferOp::create(rewriter, storeOp.getLoc(),
                                           expandOp.getSrc(), collapsedBuffer);

    rewriter.eraseOp(storeOp);
    return success();
  }
};

/// Pattern to fuse tensor.expand_shape ops that only introduce unit dimensions
/// into iree_tensor_ext.dispatch.tensor.store.
struct FuseRankIncreasingExpandIntoDispatchTensorStore
    : public OpRewritePattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using OpRewritePattern<
      IREE::TensorExt::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Check if the value is produced by an expand_shape.
    auto expandOp = storeOp.getValue().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp) {
      return failure();
    }

    // Check that the expand_shape only introduces unit dimensions.
    if (isRankReducedType(expandOp.getResultType(), expandOp.getSrcType()) !=
        SliceVerificationResult::Success) {
      return failure();
    }

    // Replace the store with one that uses the expand_shape source directly.
    // dispatch.tensor.store supports rank-reducing semantics.
    IREE::TensorExt::DispatchTensorStoreOp::create(
        rewriter, storeOp.getLoc(), expandOp.getSrc(), storeOp.getTarget(),
        storeOp.getTargetDims(), storeOp.getMixedOffsets(),
        storeOp.getMixedSizes(), storeOp.getMixedStrides());

    rewriter.eraseOp(storeOp);
    return success();
  }
};

/// Pattern to fuse rank-reducing tensor.insert_slice ops into
/// iree_tensor_ext.dispatch.tensor.store.
/// This handles IR like:
///   %empty = tensor.empty() : tensor<1x64x1x68xf16>
///   %inserted = tensor.insert_slice %src into %empty[0, 0, 0, 0] [1, 64, 1,
///   68] [1, 1, 1, 1] dispatch.tensor.store %inserted, ...
/// by replacing the store source with the source tensor directly when the
/// insert_slice only introduces unit dimensions.
struct FuseRankReducingInsertSliceIntoDispatchTensorStore
    : public OpRewritePattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using OpRewritePattern<
      IREE::TensorExt::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Check if the value is produced by an insert_slice.
    auto insertOp = storeOp.getValue().getDefiningOp<tensor::InsertSliceOp>();
    if (!insertOp) {
      return failure();
    }

    // Check that the destination is produced by tensor.empty.
    if (!insertOp.getDest().getDefiningOp<tensor::EmptyOp>()) {
      return failure();
    }

    // Check that the insert_slice is rank-reducing (only introduces unit
    // dimensions).
    auto sourceType = insertOp.getSourceType();
    auto resultType = insertOp.getResultType();
    if (isRankReducedType(resultType, sourceType) !=
        SliceVerificationResult::Success) {
      return failure();
    }

    // Replace the store with one that uses the insert_slice source directly.
    // dispatch.tensor.store supports rank-reducing semantics.
    IREE::TensorExt::DispatchTensorStoreOp::create(
        rewriter, storeOp.getLoc(), insertOp.getSource(), storeOp.getTarget(),
        storeOp.getTargetDims(), storeOp.getMixedOffsets(),
        storeOp.getMixedSizes(), storeOp.getMixedStrides());

    rewriter.eraseOp(storeOp);
    return success();
  }
};

/// Helper to check if an operation is inside a PCF loop or generic.
static bool isInsidePCFContext(Operation *op) {
  return op->getParentOfType<IREE::PCF::LoopOp>() ||
         op->getParentOfType<IREE::PCF::GenericOp>();
}

/// Pattern to compose nested tensor.extract_slice ops when inside a PCF
/// context. This avoids materialization of intermediate tensors that cause
/// spurious workgroup allocations during bufferization.
struct ComposeExtractSliceInsidePCFLoop
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp consumerOp,
                                PatternRewriter &rewriter) const override {
    // Check if the source is another extract_slice.
    auto producerOp =
        consumerOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!producerOp) {
      return failure();
    }

    // Only compose if the consumer is inside a PCF context.
    if (!isInsidePCFContext(consumerOp)) {
      return failure();
    }

    // Compute the composed offsets, sizes, and strides.
    SmallVector<OpFoldResult> combinedOffsets, combinedSizes, combinedStrides;
    if (failed(affine::mergeOffsetsSizesAndStrides(
            rewriter, consumerOp.getLoc(), producerOp.getMixedOffsets(),
            producerOp.getMixedSizes(), producerOp.getMixedStrides(),
            producerOp.getDroppedDims(), consumerOp.getMixedOffsets(),
            consumerOp.getMixedSizes(), consumerOp.getMixedStrides(),
            combinedOffsets, combinedSizes, combinedStrides))) {
      return failure();
    }

    // Create the composed extract_slice.
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        consumerOp, consumerOp.getType(), producerOp.getSource(),
        combinedOffsets, combinedSizes, combinedStrides);
    return success();
  }
};

/// Pattern to compose tensor.extract_slice with dispatch.tensor.load when
/// inside a PCF context. This avoids materialization of intermediate tensors.
struct ComposeExtractSliceOfDispatchLoadInsidePCFLoop
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp consumerOp,
                                PatternRewriter &rewriter) const override {
    // Check if the source is a dispatch.tensor.load.
    auto loadOp = consumerOp.getSource()
                      .getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    if (!loadOp) {
      return failure();
    }

    // Only compose if the consumer is inside a PCF context.
    if (!isInsidePCFContext(consumerOp)) {
      return failure();
    }

    // Compute the composed offsets, sizes, and strides.
    SmallVector<OpFoldResult> combinedOffsets, combinedSizes, combinedStrides;
    if (failed(affine::mergeOffsetsSizesAndStrides(
            rewriter, consumerOp.getLoc(), loadOp.getMixedOffsets(),
            loadOp.getMixedSizes(), loadOp.getMixedStrides(),
            loadOp.getDroppedDims(), consumerOp.getMixedOffsets(),
            consumerOp.getMixedSizes(), consumerOp.getMixedStrides(),
            combinedOffsets, combinedSizes, combinedStrides))) {
      return failure();
    }

    // Create the composed dispatch.tensor.load.
    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorLoadOp>(
        consumerOp, consumerOp.getType(), loadOp.getSource(),
        loadOp.getSourceDims(), combinedOffsets, combinedSizes,
        combinedStrides);
    return success();
  }
};

/// Pattern to compose tensor.extract_slice with iree_codegen.load_from_buffer
/// when inside a PCF context. This creates a memref.subview and loads from it.
struct ComposeExtractSliceOfLoadFromBufferInsidePCF
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp consumerOp,
                                PatternRewriter &rewriter) const override {
    // Check if the source is a load_from_buffer.
    auto loadOp =
        consumerOp.getSource().getDefiningOp<IREE::Codegen::LoadFromBufferOp>();
    if (!loadOp) {
      return failure();
    }

    // Only compose if the consumer is inside a PCF context.
    if (!isInsidePCFContext(consumerOp)) {
      return failure();
    }

    // Don't handle rank-reducing extract_slices. The subview would have a
    // different rank than the result tensor type.
    if (consumerOp.getSourceType().getRank() !=
        consumerOp.getType().getRank()) {
      return failure();
    }

    // Create a subview of the source buffer with the extract_slice's
    // offsets/sizes/strides.
    Value sourceBuffer = loadOp.getBuffer();
    Value subview = memref::SubViewOp::create(
        rewriter, consumerOp.getLoc(), sourceBuffer,
        consumerOp.getMixedOffsets(), consumerOp.getMixedSizes(),
        consumerOp.getMixedStrides());

    // Create a new load_from_buffer from the subview.
    rewriter.replaceOpWithNewOp<IREE::Codegen::LoadFromBufferOp>(
        consumerOp, consumerOp.getType(), subview);
    return success();
  }
};

/// Pattern to convert vector.transfer_read of tensor.empty to ub.poison.
/// Reading from an uninitialized tensor produces undefined values, so we can
/// replace it with poison to avoid allocating memory for the tensor.
struct FoldTransferReadOfTensorEmpty
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    // Check if the source is a tensor.empty.
    if (!readOp.getBase().getDefiningOp<tensor::EmptyOp>()) {
      return failure();
    }

    // Replace with poison of the result vector type.
    rewriter.replaceOpWithNewOp<ub::PoisonOp>(readOp, readOp.getVectorType());
    return success();
  }
};

/// Pattern to fold rank-increasing tensor.insert_slice ops into
/// vector.transfer_write if the insert_slice only introduces unit dimensions.
/// This handles IR like:
///   %0 = vector.transfer_write %v, %t[%c0, %c0] : vector<64x64xi8>,
///   tensor<...> %1 = tensor.insert_slice %0 into %dest[0, 0, 0, 0] [1, 64, 1,
///   64] [1, 1, 1]
///       : tensor<64x64xi8> into tensor<1x64x1x64xi8>
/// by writing directly to %dest with an updated permutation map.
struct FoldRankIncreasingInsertSliceIntoTransferWrite
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    // Check for unit stride.
    if (!insertOp.hasUnitStride()) {
      return failure();
    }

    // Check if the source is a transfer_write.
    auto writeOp =
        insertOp.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!writeOp) {
      return failure();
    }

    // Check for 0-d corner case.
    if (writeOp.getTransferRank() == 0) {
      return failure();
    }

    // Check that the transfer_write has no mask.
    if (writeOp.getMask()) {
      return failure();
    }

    // Check that the insert_slice is rank-increasing (source has fewer dims
    // than result).
    RankedTensorType sourceType = insertOp.getSourceType();
    RankedTensorType resultType = insertOp.getType();
    if (sourceType.getRank() >= resultType.getRank()) {
      return failure();
    }

    // Check that it only introduces unit dimensions (rank-increased type check)
    // by verifying the result is a rank-extended version of the source.
    if (isRankReducedType(resultType, sourceType) !=
        SliceVerificationResult::Success) {
      return failure();
    }

    // Check that the transfer_write fully overwrites its destination.
    // The vector shape must match the source tensor shape.
    VectorType vecType = writeOp.getVectorType();
    if (vecType.getRank() != sourceType.getRank()) {
      return failure();
    }
    for (auto [vecDim, tensorDim] :
         llvm::zip_equal(vecType.getShape(), sourceType.getShape())) {
      if (vecDim != tensorDim) {
        return failure();
      }
    }

    // Check all indices of transfer_write are zero. Otherwise, the vector
    // wouldn't fully overwrite the tensor.
    for (Value idx : writeOp.getIndices()) {
      if (!isZeroInteger(idx)) {
        return failure();
      }
    }

    // Check that the permutation map is identity. We could support more
    // permutations, but identity covers the common case.
    if (!writeOp.getPermutationMap().isIdentity()) {
      return failure();
    }

    // Get the dropped dims (unit dims in result that don't exist in source).
    llvm::SmallBitVector droppedDims = insertOp.getDroppedDims();

    // Build the new permutation map. The permutation map for transfer_write
    // maps from tensor dimensions (numDims = resultRank) to vector dimensions
    // (numResults = vectorRank). For non-unit dims, we map to the corresponding
    // vector dimension. For unit (dropped) dims, we don't include them in the
    // result since we're only writing to those dimensions at specific offsets.
    int64_t resultRank = resultType.getRank();
    int64_t vectorRank = vecType.getRank();
    SmallVector<AffineExpr> exprs;
    [[maybe_unused]] int64_t vectorIdx = 0;
    for (int64_t i = 0; i < resultRank; ++i) {
      if (!droppedDims.test(i)) {
        // This non-unit dimension maps to a vector dimension.
        exprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
        ++vectorIdx;
      }
    }
    assert(vectorIdx == vectorRank && "dimension count mismatch");

    AffineMap newPermutationMap =
        AffineMap::get(resultRank, 0, exprs, rewriter.getContext());

    // Compute new indices: use the insert_slice offsets.
    SmallVector<Value> newIndices = getValueOrCreateConstantIndexOp(
        rewriter, insertOp.getLoc(), insertOp.getMixedOffsets());

    // Compute in_bounds: we're writing the full vector at the given offsets.
    // The vector fully covers the non-unit dimensions.
    SmallVector<bool> inBounds(vectorRank, true);

    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        insertOp, writeOp.getVector(), insertOp.getDest(), newIndices,
        newPermutationMap, inBounds);
    return success();
  }
};

/// Pattern to fuse iree_linalg_ext.map_scatter into
/// iree_codegen.store_to_buffer by replacing the map_scatter's tensor
/// destination with the store's memref. This avoids allocating an intermediate
/// tensor buffer during bufferization.
///
/// Converts:
///   %dest = tensor.empty() or iree_codegen.load_from_buffer %buf
///   %result = iree_linalg_ext.map_scatter %input into %dest {...}
///   iree_codegen.store_to_buffer %result, %buf
/// To:
///   %input_buf = memref.alloca (if input is tensor)
///   iree_codegen.store_to_buffer %input, %input_buf (if input is tensor)
///   iree_linalg_ext.map_scatter %input_buf into %buf {...}
///
/// The destination must come from tensor.empty() or load_from_buffer (with the
/// same buffer as the store) to ensure correctness.
struct FuseMapScatterIntoBufferStore
    : public OpRewritePattern<IREE::Codegen::StoreToBufferOp> {
  using OpRewritePattern<IREE::Codegen::StoreToBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Codegen::StoreToBufferOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Check if the tensor is produced by a map_scatter.
    auto mapScatterOp =
        storeOp.getTensor().getDefiningOp<IREE::LinalgExt::MapScatterOp>();
    if (!mapScatterOp) {
      return failure();
    }

    // Only fuse if the map_scatter has a single use (the store).
    if (!mapScatterOp->hasOneUse()) {
      return failure();
    }

    Value buffer = storeOp.getBuffer();

    // Verify that the output (destination) comes from tensor.empty() or
    // load_from_buffer with the same underlying buffer we're storing to.
    Value dest = mapScatterOp.getOutput();
    if (auto loadOp = dest.getDefiningOp<IREE::Codegen::LoadFromBufferOp>()) {
      // If destination comes from load_from_buffer, verify it's loading from
      // the same buffer we're storing to (otherwise semantics would differ).
      if (loadOp.getBuffer() != buffer) {
        return failure();
      }
    } else if (!dest.getDefiningOp<tensor::EmptyOp>()) {
      // Destination must be from either load_from_buffer or tensor.empty().
      return failure();
    }

    // Check if the input comes from load_from_buffer. If so, use the underlying
    // memref directly for pure buffer semantics. Otherwise, keep it as a tensor
    // (MapScatterOp supports mixed semantics: tensor input with memref output).
    Value input = mapScatterOp.getInput();
    if (auto inputLoadOp =
            input.getDefiningOp<IREE::Codegen::LoadFromBufferOp>()) {
      input = inputLoadOp.getBuffer();
    }
    Location loc = mapScatterOp.getLoc();

    // Create a new map_scatter that writes directly to the memref.
    // When the output is a memref, the op has no tensor result.
    rewriter.setInsertionPoint(storeOp);
    auto newMapScatterOp = IREE::LinalgExt::MapScatterOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{}, input, buffer);

    // Move the transformation region from the old op to the new one.
    rewriter.inlineRegionBefore(
        mapScatterOp.getTransformationRegion(),
        newMapScatterOp.getTransformationRegion(),
        newMapScatterOp.getTransformationRegion().end());

    // Erase both the store and the old map_scatter.
    rewriter.eraseOp(storeOp);
    rewriter.eraseOp(mapScatterOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct FuseTensorToBufferConvertersPass final
    : impl::FuseTensorToBufferConvertersPassBase<
          FuseTensorToBufferConvertersPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<FuseStoreToBuffer, FuseDispatchTensorStore,
                 FuseRankIncreasingExpandIntoBufferStore,
                 FuseRankIncreasingExpandIntoDispatchTensorStore,
                 FuseRankReducingInsertSliceIntoDispatchTensorStore,
                 FoldRankIncreasingInsertSliceIntoTransferWrite,
                 ComposeExtractSliceInsidePCFLoop,
                 ComposeExtractSliceOfDispatchLoadInsidePCFLoop,
                 ComposeExtractSliceOfLoadFromBufferInsidePCF,
                 FoldTransferReadOfTensorEmpty, FuseMapScatterIntoBufferStore>(
        context);
    IREE::PCF::populatePCFDropUnusedResultPatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    // Add reshape fusion patterns that become applicable after fusing stores
    // into pcf.loop. These help eliminate extra allocations from reshape ops.
    populateReshapeToInterfaceTensorPatterns(patterns);
    populateFoldTensorReshapeIntoBufferPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
