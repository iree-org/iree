// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FUSEPCFSTORESPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Helper function to get all PCF::WriteSliceOps from a PCF::GenericOp or
/// PCF::LoopOp that write to a specific result.
template <typename PCFOpTy>
static FailureOr<SmallVector<IREE::PCF::WriteSliceOp>>
getProducerSlices(PCFOpTy pcfOp, OpResult result) {
  static_assert(std::is_same_v<PCFOpTy, IREE::PCF::GenericOp> ||
                    std::is_same_v<PCFOpTy, IREE::PCF::LoopOp>,
                "PCFOpTy must be PCF::GenericOp or PCF::LoopOp");
  BlockArgument tiedArg = pcfOp.getRegionRefArgs()[result.getResultNumber()];

  // The fusion is only valid if the sref type is return only sync scope.
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

    // Get the write_slice ops that produce the result written by the store.
    OpResult result = cast<OpResult>(tensor);
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
      if (isa<RankedTensorType>(sourceType)) {
        IREE::Codegen::StoreToBufferOp::create(
            rewriter, storeOp.getLoc(), writeSlice.getSource(), destSlice);
      } else if (isa<MemRefType>(sourceType)) {
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
        llvm_unreachable("Invalid write_slice operand type");
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

    // Get the write_slice ops that produce the result written by the store.
    OpResult result = cast<OpResult>(value);
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
          RankedTensorType insertSourceType = insertOp.getSourceType();
          RankedTensorType insertResultType = insertOp.getResultType();
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

struct FusePCFStoresPass final
    : impl::FusePCFStoresPassBase<FusePCFStoresPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<FuseStoreToBuffer, FuseDispatchTensorStore>(context);
    IREE::PCF::populatePCFDropUnusedResultPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
