// Copyright 2025 The IREE Authors
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
  for (int64_t shape : type.getShape()) {
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
  if (failed(memrefType.getStridesAndOffset(strides, offset)))
    return failure();
  if (strides.empty())
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

static Value linearizeOperand(Location loc, PatternRewriter &rewriter,
                              Value operand, MemRefType linearizedType,
                              OpFoldResult sizeRes = nullptr) {
  // Fetch offset and strides of the old memref.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(linearizedType.getStridesAndOffset(strides, offset))) {
    // TODO(avarma): Change function signature.
    return nullptr;
  }

  if (linearizedType.hasStaticShape()) {
    return rewriter.create<memref::ReinterpretCastOp>(
        loc, linearizedType, operand, /*offset=*/offset,
        linearizedType.getShape(), /*strides=*/strides);
  } else {
    assert(sizeRes && "expected a linear dynamic size to have been computed");
    SmallVector<OpFoldResult> size = {sizeRes};
    OpFoldResult staticOffset = rewriter.getIndexAttr(offset);
    SmallVector<OpFoldResult> strides;
    strides.push_back(rewriter.getIndexAttr(1));
    return rewriter.create<memref::ReinterpretCastOp>(
        loc, linearizedType, operand, staticOffset, size, strides);
  }
}

/// Utility function used to linearize indices/sizes.
static FailureOr<Value>
createLinearizeIndexOp(PatternRewriter &rewriter, Location loc, Value memref,
                       SmallVector<Value> multiIndices = {}) {
  // Step 1. Form basis.
  FailureOr<SmallVector<OpFoldResult>> basis =
      memref::getMixedSizes(rewriter, loc, memref);
  if (failed(basis))
    return failure();
  // Step 2. Form multiIndices if not already present.
  if (multiIndices.empty()) {
    SmallVector<Value> multiIndicesNew(
        (*basis).size(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
    if (auto attr = llvm::dyn_cast_if_present<Attribute>((*basis)[0])) {
      auto intVal = cast<IntegerAttr>(attr).getInt();
      multiIndicesNew[0] =
          rewriter.create<arith::ConstantIndexOp>(loc, intVal).getResult();
    } else {
      multiIndicesNew[0] = llvm::dyn_cast_if_present<Value>((*basis)[0]);
    }
    multiIndices = multiIndicesNew;
  }
  // Step 3. Form linearizedIndexOp.
  Value linearizedSizes = rewriter.create<affine::AffineLinearizeIndexOp>(
      loc, multiIndices, *basis, true);
  return linearizedSizes;
}

template <typename OpTy>
struct LinearizeMemrefAlloc : public OpRewritePattern<OpTy> {
  LinearizeMemrefAlloc(MLIRContext *context)
      : OpRewritePattern<OpTy>(context) {}

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

    SmallVector<Value> dynamicLinearizedSize;
    if (!newTypeOfSourceMemref.hasStaticShape()) {
      FailureOr<Value> linearizedSizes =
          createLinearizeIndexOp(rewriter, loc, allocOp.getResult());
      if (failed(linearizedSizes))
        return failure();
      dynamicLinearizedSize.push_back(*linearizedSizes);
    }

    Value linearizedOp = rewriter.create<OpTy>(
        loc, newTypeOfSourceMemref, dynamicLinearizedSize,
        allocOp.getSymbolOperands(), allocOp.getAlignmentAttr());
    SmallVector<int64_t, 2> indices(currentTypeOfSourceMemref.getRank(), 0);
    std::iota(indices.begin(), indices.end(), 0);
    Value delinearizedOp = rewriter.create<memref::ExpandShapeOp>(
        loc, currentTypeOfSourceMemref, linearizedOp, ArrayRef({indices}),
        allocOp.getMixedSizes());
    rewriter.replaceOp(allocOp, delinearizedOp);
    return success();
  }
};

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

    FailureOr<Value> linearizedIndices = createLinearizeIndexOp(
        rewriter, loc, loadOp.getMemref(), loadOp.getIndices());
    if (failed(linearizedIndices))
      return failure();
    FailureOr<Value> linearizedSizes =
        createLinearizeIndexOp(rewriter, loc, loadOp.getMemref());
    if (failed(linearizedSizes))
      return failure();
    Value linearizedOperand =
        linearizeOperand(loc, rewriter, loadOp.getMemref(),
                         newTypeOfSourceMemref, *linearizedSizes);
    rewriter.replaceOpWithNewOp<memref::LoadOp>(
        loadOp, linearizedOperand, *linearizedIndices, loadOp.getNontemporal());

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

    FailureOr<Value> linearizedIndices = createLinearizeIndexOp(
        rewriter, loc, storeOp.getMemref(), storeOp.getIndices());
    if (failed(linearizedIndices))
      return failure();
    FailureOr<Value> linearizedSizes =
        createLinearizeIndexOp(rewriter, loc, storeOp.getMemref());
    if (failed(linearizedSizes))
      return failure();
    Value linearizedOperand =
        linearizeOperand(loc, rewriter, storeOp.getMemref(),
                         newTypeOfSourceMemref, *linearizedSizes);

    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        storeOp, storeOp.getValueToStore(), linearizedOperand,
        *linearizedIndices, storeOp.getNontemporal());

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

    FailureOr<Value> linearizedSizes =
        createLinearizeIndexOp(rewriter, loc, deallocOp.getMemref());
    if (failed(linearizedSizes))
      return failure();
    Value linearizedOperand =
        linearizeOperand(loc, rewriter, deallocOp.getMemref(),
                         newTypeOfSourceMemref, *linearizedSizes);

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

    FailureOr<Value> linearizedSizes =
        createLinearizeIndexOp(rewriter, loc, copyOp.getSource());
    if (failed(linearizedSizes))
      return failure();
    Value linearizedSource =
        linearizeOperand(loc, rewriter, copyOp.getSource(),
                         newTypeOfSourceMemref, *linearizedSizes);
    linearizedSizes = createLinearizeIndexOp(rewriter, loc, copyOp.getTarget());
    if (failed(linearizedSizes))
      return failure();
    Value linearizedTarget =
        linearizeOperand(loc, rewriter, copyOp.getTarget(),
                         newTypeOfSourceMemref, *linearizedSizes);

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

    FailureOr<Value> linearizedIndices = createLinearizeIndexOp(
        rewriter, loc, loadOp.getBase(), loadOp.getIndices());
    if (failed(linearizedIndices))
      return failure();
    FailureOr<Value> linearizedSizes =
        createLinearizeIndexOp(rewriter, loc, loadOp.getBase());
    if (failed(linearizedSizes))
      return failure();
    Value linearizedOperand =
        linearizeOperand(loc, rewriter, loadOp.getBase(), newTypeOfSourceMemref,
                         *linearizedSizes);

    rewriter.replaceOpWithNewOp<vector::LoadOp>(
        loadOp, loadOp.getType(), linearizedOperand, *linearizedIndices);

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

    FailureOr<Value> linearizedIndices = createLinearizeIndexOp(
        rewriter, loc, storeOp.getBase(), storeOp.getIndices());
    if (failed(linearizedIndices))
      return failure();
    FailureOr<Value> linearizedSizes =
        createLinearizeIndexOp(rewriter, loc, storeOp.getBase());
    if (failed(linearizedSizes))
      return failure();
    Value linearizedOperand =
        linearizeOperand(loc, rewriter, storeOp.getBase(),
                         newTypeOfSourceMemref, *linearizedSizes);

    rewriter.replaceOpWithNewOp<vector::StoreOp>(
        storeOp, storeOp.getValueToStore(), linearizedOperand,
        *linearizedIndices);

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

  (void)applyPatternsGreedily(moduleOp, std::move(patterns));

  return;
}
} // namespace
} // namespace mlir::iree_compiler
