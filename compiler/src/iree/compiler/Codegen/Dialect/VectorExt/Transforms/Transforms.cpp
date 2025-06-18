// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Rewrite/PatternApplicator.h"

namespace mlir::iree_compiler::IREE::VectorExt {

namespace {

template <typename LoadOrStoreOpTy>
static Value getMemRefOperand(LoadOrStoreOpTy op) {
  return op.getMemref();
}

static Value getMemRefOperand(vector::LoadOp op) { return op.getBase(); }

static Value getMemRefOperand(vector::StoreOp op) { return op.getBase(); }

static Value getMemRefOperand(vector::MaskedLoadOp op) { return op.getBase(); }

static Value getMemRefOperand(vector::MaskedStoreOp op) { return op.getBase(); }

static LogicalResult
resolveSourceIndicesExpandShape(Location loc, PatternRewriter &rewriter,
                                LayoutedExpandOp expandOp, ValueRange indices,
                                SmallVectorImpl<Value> &sourceIndices) {
  NestedLayoutAttr layout = expandOp.getLayout();
  int64_t rank = layout.getRank();
  for (auto i : llvm::seq<int64_t>(rank)) {
    SmallVector<Value> dimIndices = {
        indices[0 * rank + i], indices[1 * rank + i], indices[2 * rank + i],
        indices[3 * rank + i], indices[4 * rank + i]};
    SmallVector<int64_t> basis = layout.getPackedShapeForUndistributedDim(i);
    // LayoutedExpandOp guarantees that the inner tile indices are in_bounds, so
    // the indices must be disjoint.
    Value linIndex =
        rewriter.create<affine::AffineLinearizeIndexOp>(loc, dimIndices, basis,
                                                        /*disjoint=*/true);
    // Add the offset to the lin index.
    Value offset = expandOp.getOffsets()[i];
    AffineExpr a, b;
    bindDims(rewriter.getContext(), a, b);
    Value offsetedLinIndex = affine::makeComposedAffineApply(
        rewriter, loc, a + b, {linIndex, offset});
    sourceIndices.push_back(offsetedLinIndex);
  }
  return success();
}

template <typename OpTy>
class LoadOpOfLayoutedExpandShapeOpFolder final
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy loadOp,
                                PatternRewriter &rewriter) const override;
};

template <typename OpTy>
class StoreOpOfLayoutedExpandShapeOpFolder final
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy storeOp,
                                PatternRewriter &rewriter) const override;
};

template <typename OpTy>
LogicalResult LoadOpOfLayoutedExpandShapeOpFolder<OpTy>::matchAndRewrite(
    OpTy loadOp, PatternRewriter &rewriter) const {
  auto expandShapeOp =
      getMemRefOperand(loadOp).template getDefiningOp<LayoutedExpandOp>();

  if (!expandShapeOp)
    return failure();

  SmallVector<Value> indices(loadOp.getIndices().begin(),
                             loadOp.getIndices().end());
  SmallVector<Value> sourceIndices;
  if (failed(resolveSourceIndicesExpandShape(
          loadOp.getLoc(), rewriter, expandShapeOp, indices, sourceIndices)))
    return failure();
  llvm::TypeSwitch<Operation *, void>(loadOp)
      .Case([&](memref::LoadOp op) {
        rewriter.replaceOpWithNewOp<memref::LoadOp>(
            loadOp, expandShapeOp.getViewSource(), sourceIndices,
            op.getNontemporal());
      })
      .Case([&](vector::LoadOp op) {
        rewriter.replaceOpWithNewOp<vector::LoadOp>(
            op, op.getType(), expandShapeOp.getViewSource(), sourceIndices,
            op.getNontemporal());
      })
      .Case([&](vector::MaskedLoadOp op) {
        rewriter.replaceOpWithNewOp<vector::MaskedLoadOp>(
            op, op.getType(), expandShapeOp.getViewSource(), sourceIndices,
            op.getMask(), op.getPassThru());
      })
      .Default([](Operation *) { llvm_unreachable("unexpected operation."); });
  return success();
}

template <typename OpTy>
LogicalResult StoreOpOfLayoutedExpandShapeOpFolder<OpTy>::matchAndRewrite(
    OpTy storeOp, PatternRewriter &rewriter) const {
  auto expandShapeOp =
      getMemRefOperand(storeOp).template getDefiningOp<LayoutedExpandOp>();

  if (!expandShapeOp)
    return failure();

  SmallVector<Value> indices(storeOp.getIndices().begin(),
                             storeOp.getIndices().end());
  SmallVector<Value> sourceIndices;
  if (failed(resolveSourceIndicesExpandShape(
          storeOp.getLoc(), rewriter, expandShapeOp, indices, sourceIndices)))
    return failure();
  llvm::TypeSwitch<Operation *, void>(storeOp)
      .Case([&](memref::StoreOp op) {
        rewriter.replaceOpWithNewOp<memref::StoreOp>(
            storeOp, op.getValueToStore(), expandShapeOp.getViewSource(),
            sourceIndices, op.getNontemporal());
      })
      .Case([&](vector::StoreOp op) {
        rewriter.replaceOpWithNewOp<vector::StoreOp>(
            op, op.getValueToStore(), expandShapeOp.getViewSource(),
            sourceIndices, op.getNontemporal());
      })
      .Case([&](vector::MaskedStoreOp op) {
        rewriter.replaceOpWithNewOp<vector::MaskedStoreOp>(
            op, expandShapeOp.getViewSource(), sourceIndices, op.getMask(),
            op.getValueToStore());
      })
      .Default([](Operation *) { llvm_unreachable("unexpected operation."); });
  return success();
}

} // namespace

void populateVectorExtFoldMemRefAliasOpPatterns(RewritePatternSet &patterns) {
  patterns.add<LoadOpOfLayoutedExpandShapeOpFolder<memref::LoadOp>,
               LoadOpOfLayoutedExpandShapeOpFolder<vector::LoadOp>,
               LoadOpOfLayoutedExpandShapeOpFolder<vector::MaskedLoadOp>,
               StoreOpOfLayoutedExpandShapeOpFolder<memref::StoreOp>,
               StoreOpOfLayoutedExpandShapeOpFolder<vector::StoreOp>,
               StoreOpOfLayoutedExpandShapeOpFolder<vector::MaskedStoreOp>>(
      patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::VectorExt
