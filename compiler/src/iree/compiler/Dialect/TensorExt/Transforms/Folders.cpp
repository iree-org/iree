// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir::iree_compiler::IREE::TensorExt {

/// Pattern to fold `iree_tensor_ext.dispatch.tensor.load` ->
/// `tensor.extract_slice`.
struct FoldTensorLoadWithExtractSlice
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    auto dispatchTensorLoadOp =
        extractSliceOp.getSource()
            .getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    if (!dispatchTensorLoadOp) {
      return failure();
    }

    SmallVector<OpFoldResult> offsets, sizes, strides;
    // `tensor.extract_slice` (i.e. the producer) folds **into**
    // `iree_tensor_ext.dispatch.tensor.load1 (i.e. the consumer).
    if (failed(affine::mergeOffsetsSizesAndStrides(
            rewriter, dispatchTensorLoadOp->getLoc(), dispatchTensorLoadOp,
            extractSliceOp, dispatchTensorLoadOp.getDroppedDims(), offsets,
            sizes, strides))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorLoadOp>(
        extractSliceOp, extractSliceOp.getType(),
        dispatchTensorLoadOp.getSource(), dispatchTensorLoadOp.getSourceDims(),
        offsets, sizes, strides);
    return success();
  }
};

/// Pattern to fold `tensor.insert_slice` with
/// `iree_tensor_ext.dispatch.tensor.store` operations.
struct FoldInsertSliceWithTensorStoreOp
    : OpRewritePattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp dispatchTensorStoreOp,
                  PatternRewriter &rewriter) const override {
    auto insertSliceOp =
        dispatchTensorStoreOp.getValue().getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSliceOp) {
      return failure();
    }

    // Fold is safe when dest carries no values that need to pass through:
    // either `tensor.empty` (uninitialized) or a `dispatch.tensor.load` of
    // the same target with matching offsets/sizes/strides (the load/store
    // round-trip leaves outside-insert positions unchanged).
    auto isFoldSafe = [&] {
      Operation *destOp = insertSliceOp.getDest().getDefiningOp();
      if (!destOp) {
        return false;
      }
      if (isa<tensor::EmptyOp>(destOp)) {
        return true;
      }
      auto loadOp = dyn_cast<IREE::TensorExt::DispatchTensorLoadOp>(destOp);
      if (!loadOp || loadOp.getSource() != dispatchTensorStoreOp.getTarget()) {
        return false;
      }
      return mlir::detail::sameOffsetsSizesAndStrides(
          loadOp, dispatchTensorStoreOp, isEqualConstantIntOrValue);
    };
    if (!isFoldSafe()) {
      return failure();
    }

    SmallVector<OpFoldResult> offsets, sizes, strides;
    // `tensor.insert_slice` (i.e. the producer) folds **into**
    // `iree_tensor_ext.dispatch.tensor.store` (i.e. the consumer).
    if (failed(affine::mergeOffsetsSizesAndStrides(
            rewriter, dispatchTensorStoreOp->getLoc(), dispatchTensorStoreOp,
            insertSliceOp, dispatchTensorStoreOp.getDroppedDims(), offsets,
            sizes, strides))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorStoreOp>(
        dispatchTensorStoreOp, insertSliceOp.getSource(),
        dispatchTensorStoreOp.getTarget(),
        dispatchTensorStoreOp.getTargetDims(), offsets, sizes, strides);
    return success();
  }
};

void populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns
      .insert<FoldTensorLoadWithExtractSlice, FoldInsertSliceWithTensorStoreOp>(
          context);
}

} // namespace mlir::iree_compiler::IREE::TensorExt
