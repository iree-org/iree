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

    // Only fold when dest is a load of the same target with matching
    // offsets/sizes/strides; otherwise positions outside the partial insert
    // lose passthrough.
    auto loadOp = insertSliceOp.getDest()
                      .getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    if (!loadOp || loadOp.getSource() != dispatchTensorStoreOp.getTarget() ||
        !mlir::detail::sameOffsetsSizesAndStrides(loadOp, dispatchTensorStoreOp,
                                                  isEqualConstantIntOrValue)) {
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
