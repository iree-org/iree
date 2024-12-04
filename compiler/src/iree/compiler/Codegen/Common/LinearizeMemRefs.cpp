// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===- LinearizeMemRefs.cpp - Flatten n-D MemRef subspan ------------------===//
//
// This file implements an interprocedural pass to linearize memrefs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linearize-memrefs"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LINEARIZEMEMREFS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static SmallVector<int64_t> getLinearizedShape(MemRefType type) {
  if (type.getRank() == 0)
    return {};

  int64_t linearizedShape = 1;
  for (auto shape : type.getShape()) {
    if (shape == ShapedType::kDynamic)
      return {ShapedType::kDynamic};
    linearizedShape *= shape;
  }
  return {linearizedShape};
}

static FailureOr<MemRefType> linearizeType(MemRefType memrefType) {
  // Fetch linearized shape.
  SmallVector<int64_t> linearizedShape = getLinearizedShape(memrefType);
  // Fetch offset and strides of the old memref.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memrefType, strides, offset)))
    return failure();
  if (!strides.empty() && strides.back() != 1)
    return failure();
  // Form layout for the linearized memref.
  StridedLayoutAttr layoutAttr;
  // If the offset is 0, we do not need a strided layout as the stride is
  // 1, so we only use the strided layout if the offset is not 0.
  if (offset != 0) {
    layoutAttr = StridedLayoutAttr::get(memrefType.getContext(), offset,
                                        ArrayRef<int64_t>{1});
  }
  Type elementType = memrefType.getElementType();
  return MemRefType::get(linearizedShape, elementType, layoutAttr,
                         memrefType.getMemorySpace());
}

static FailureOr<MemRefType>
getLinearizedTypeFromSourceType(MemRefType currentTypeOfSourceMemref) {
  if (!currentTypeOfSourceMemref)
    return failure();
  // Convert current type later.
  return linearizeType(currentTypeOfSourceMemref);
}

template <typename OpTy>
struct LinearizeMemrefAlloc : public OpRewritePattern<OpTy> {
  LinearizeMemrefAlloc(MLIRContext *context, PatternBenefit benefit = 10)
      : OpRewritePattern<OpTy>(context, benefit) {}

  LogicalResult matchAndRewrite(OpTy allocOp,
                                PatternRewriter &rewriter) const override {
    static_assert(std::is_same<OpTy, memref::AllocOp>() ||
                      std::is_same<OpTy, memref::AllocaOp>(),
                  "expected only memref::AllocOp or memref::AllocaOp");
    Location loc = allocOp->getLoc();
    MemRefType currentTypeOfSourceMemref =
        dyn_cast<MemRefType>(allocOp.getMemref().getType());
    if (currentTypeOfSourceMemref.getRank() < 2)
      return success();
    FailureOr<MemRefType> maybeNewTypeOfSourceMemref =
        getLinearizedTypeFromSourceType(currentTypeOfSourceMemref);
    if (failed(maybeNewTypeOfSourceMemref))
      return failure();
    MemRefType newTypeOfSourceMemref = *maybeNewTypeOfSourceMemref;

    SmallVector<Value> sizes =
        llvm::map_to_vector(allocOp.getMixedSizes(), [&](OpFoldResult size) {
          Value val;
          if (dyn_cast<Attribute>(size)) {
            val = rewriter.create<arith::ConstantIndexOp>(
                loc, *getConstantIntValue(size));
          } else {
            val = cast<Value>(size);
          }
          return val;
        });

    Value linearizedSize = rewriter.create<affine::AffineLinearizeIndexOp>(
        loc, sizes, currentTypeOfSourceMemref.getShape(), true);
    SmallVector<Value> dynamicLinearizedSize;
    if (!newTypeOfSourceMemref.hasStaticShape()) {
      dynamicLinearizedSize.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, linearizedSize));
    }

    rewriter.replaceOpWithNewOp<OpTy>(
        allocOp, newTypeOfSourceMemref, dynamicLinearizedSize,
        allocOp.getSymbolOperands(), allocOp.getAlignmentAttr());
    return success();
  }
};

static Value linearizeOperand(Location loc, PatternRewriter &rewriter,
                              Value operand, MemRefType linearizedType) {
  return rewriter.create<memref::ReinterpretCastOp>(
      loc, linearizedType, operand, 0, linearizedType.getShape(),
      ArrayRef<int64_t>({1}));
}

struct LinearizeMemrefLoad : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp->getLoc();
    MemRefType currentTypeOfSourceMemref = loadOp.getMemRefType();
    if (currentTypeOfSourceMemref.getRank() < 2 &&
        loadOp.getIndices().size() < 2)
      return success();
    FailureOr<MemRefType> maybeNewTypeOfSourceMemref =
        getLinearizedTypeFromSourceType(currentTypeOfSourceMemref);
    if (failed(maybeNewTypeOfSourceMemref))
      return failure();
    MemRefType newTypeOfSourceMemref = *maybeNewTypeOfSourceMemref;

    Value linearizedIndices = rewriter.create<affine::AffineLinearizeIndexOp>(
        loc, loadOp.getIndices(), currentTypeOfSourceMemref.getShape(), true);
    Value linearizedOperand = linearizeOperand(
        loc, rewriter, loadOp.getMemref(), newTypeOfSourceMemref);
    Value linearizedLoad = rewriter.create<memref::LoadOp>(
        loc, linearizedOperand, linearizedIndices);

    rewriter.replaceOp(loadOp, {linearizedLoad});
    return success();
  }
};

struct LinearizeMemrefStore : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = storeOp->getLoc();
    MemRefType currentTypeOfSourceMemref = storeOp.getMemRefType();
    if (currentTypeOfSourceMemref.getRank() < 2 &&
        storeOp.getIndices().size() < 2)
      return success();
    FailureOr<MemRefType> maybeNewTypeOfSourceMemref =
        getLinearizedTypeFromSourceType(currentTypeOfSourceMemref);
    if (failed(maybeNewTypeOfSourceMemref))
      return failure();
    MemRefType newTypeOfSourceMemref = *maybeNewTypeOfSourceMemref;

    auto elementType = storeOp.getMemRefType().getElementType();
    int srcBits = elementType.getIntOrFloatBitWidth();
    Value linearizedIndices = rewriter.create<affine::AffineLinearizeIndexOp>(
        loc, storeOp.getIndices(), currentTypeOfSourceMemref.getShape(), true);
    Value linearizedOperand = linearizeOperand(
        loc, rewriter, storeOp.getMemref(), newTypeOfSourceMemref);
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        storeOp, storeOp.getValueToStore(), linearizedOperand,
        linearizedIndices, srcBits);

    return success();
  }
};

struct LinearizeMemrefDealloc : public OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern<memref::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp deallocOp,
                                PatternRewriter &rewriter) const override {
    Location loc = deallocOp->getLoc();
    MemRefType currentTypeOfSourceMemref =
        dyn_cast<MemRefType>(deallocOp.getMemref().getType());
    if (currentTypeOfSourceMemref.getRank() < 2)
      return success();
    FailureOr<MemRefType> maybeNewTypeOfSourceMemref =
        getLinearizedTypeFromSourceType(currentTypeOfSourceMemref);
    if (failed(maybeNewTypeOfSourceMemref))
      return failure();
    MemRefType newTypeOfSourceMemref = *maybeNewTypeOfSourceMemref;

    Value linearizedOperand = linearizeOperand(
        loc, rewriter, deallocOp.getMemref(), newTypeOfSourceMemref);

    rewriter.replaceOpWithNewOp<memref::DeallocOp>(deallocOp,
                                                   linearizedOperand);
    return success();
  }
};

struct LinearizeMemrefCopy : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Location loc = copyOp->getLoc();
    MemRefType currentTypeOfSourceMemref =
        dyn_cast<MemRefType>(copyOp.getSource().getType());
    MemRefType currentTypeOfTargetMemref =
        dyn_cast<MemRefType>(copyOp.getTarget().getType());
    if (currentTypeOfSourceMemref.getRank() < 2 &&
        currentTypeOfTargetMemref.getRank() < 2)
      return success();
    FailureOr<MemRefType> maybeNewTypeOfSourceMemref =
        getLinearizedTypeFromSourceType(currentTypeOfSourceMemref);
    if (failed(maybeNewTypeOfSourceMemref))
      return failure();
    MemRefType newTypeOfSourceMemref = *maybeNewTypeOfSourceMemref;

    Value linearizedSource = linearizeOperand(loc, rewriter, copyOp.getSource(),
                                              newTypeOfSourceMemref);
    Value linearizedTarget = linearizeOperand(loc, rewriter, copyOp.getTarget(),
                                              newTypeOfSourceMemref);

    rewriter.replaceOpWithNewOp<memref::CopyOp>(copyOp, linearizedSource,
                                                linearizedTarget);
    return success();
  }
};

struct LinearizeVectorLoad : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp->getLoc();
    MemRefType currentTypeOfSourceMemref = loadOp.getMemRefType();
    if (currentTypeOfSourceMemref.getRank() < 2 &&
        loadOp.getIndices().size() < 2)
      return success();
    FailureOr<MemRefType> maybeNewTypeOfSourceMemref =
        getLinearizedTypeFromSourceType(currentTypeOfSourceMemref);
    if (failed(maybeNewTypeOfSourceMemref))
      return failure();
    MemRefType newTypeOfSourceMemref = *maybeNewTypeOfSourceMemref;

    Value linearizedIndices = rewriter.create<affine::AffineLinearizeIndexOp>(
        loc, loadOp.getIndices(), currentTypeOfSourceMemref.getShape(), true);
    Value linearizedOperand = linearizeOperand(loc, rewriter, loadOp.getBase(),
                                               newTypeOfSourceMemref);
    Value linearizedLoad = rewriter.create<vector::LoadOp>(
        loc, loadOp.getType(), linearizedOperand, linearizedIndices);

    rewriter.replaceOp(loadOp, {linearizedLoad});
    return success();
  }
};

struct LinearizeVectorStore : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = storeOp->getLoc();
    MemRefType currentTypeOfSourceMemref = storeOp.getMemRefType();
    if (currentTypeOfSourceMemref.getRank() < 2 &&
        storeOp.getIndices().size() < 2)
      return success();
    FailureOr<MemRefType> maybeNewTypeOfSourceMemref =
        getLinearizedTypeFromSourceType(currentTypeOfSourceMemref);
    if (failed(maybeNewTypeOfSourceMemref))
      return failure();
    MemRefType newTypeOfSourceMemref = *maybeNewTypeOfSourceMemref;

    Value linearizedIndices = rewriter.create<affine::AffineLinearizeIndexOp>(
        loc, storeOp.getIndices(), currentTypeOfSourceMemref.getShape(), true);
    Value linearizedOperand = linearizeOperand(loc, rewriter, storeOp.getBase(),
                                               newTypeOfSourceMemref);
    rewriter.replaceOpWithNewOp<vector::StoreOp>(
        storeOp, storeOp.getValueToStore(), linearizedOperand,
        linearizedIndices);

    return success();
  }
};
//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LinearizeMemRefs final : impl::LinearizeMemRefsBase<LinearizeMemRefs> {
  void runOnOperation() override;
};

void LinearizeMemRefs::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "Linearizing Memrefs...\n");
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  RewritePatternSet patterns(context);
  patterns.add<LinearizeMemrefAlloc<memref::AllocOp>>(context);
  patterns.add<LinearizeMemrefAlloc<memref::AllocaOp>>(context);
  patterns.add<LinearizeMemrefLoad>(context);
  patterns.add<LinearizeMemrefStore>(context);
  patterns.add<LinearizeMemrefDealloc>(context);
  patterns.add<LinearizeMemrefCopy>(context);
  patterns.add<LinearizeVectorLoad>(context);
  patterns.add<LinearizeVectorStore>(context);

  (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));

  return;
}
} // namespace
} // namespace mlir::iree_compiler
