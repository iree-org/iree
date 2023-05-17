// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- FlattenMemRefSubspanPass.cpp - Flatten n-D MemRef subspan ----------===//
//
// This file implements a pass to flatten n-D MemRef subspan ops to 1-D MemRef
// ones and folds the byte offsets on subspan ops to the consumer load/store
// ops, in preparation for lowering to the final target.
//
// This pass is needed because of how MemRef is used by subspan ops:
//
// 1) Normally MemRef should capture the mapping to the underlying buffer with
// its internal strides and offsets. However, although subspan ops in IREE are
// subview-like constructs, they carry the offset directly on the ops themselves
// and return MemRefs with the identity layout map. This is due to that IREE can
// perform various optimizations over buffer allocation and decide, for example,
// to use the same underlying buffer for two MemRefs, which are converted form
// disjoint tensors initially.
// 2) The byte offset on subspan ops is an offset into the final planned 1-D
// byte buffer, while the MemRef can be n-D without considering a backing
// buffer and its data layout.
//
// So to bridge the gap, we need to linearize the MemRef dimensions to bring it
// onto the same view as IREE: buffers are just a bag of bytes. Then we need to
// fold the byte offset on subspan ops to the consumer load/store ops, so that
// we can rely on transformations in MLIR core, because they assume MemRefs map
// to the underlying buffers with its internal strides and offsets.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flatten-memref-subspan"

namespace mlir {
namespace iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

/// Returns true if the given `type` is a 0-D MemRef.
static bool isRankZeroMemRef(Type type) {
  if (auto memrefType = type.dyn_cast<MemRefType>()) {
    return memrefType.hasRank() && memrefType.getRank() == 0;
  }
  return false;
}

/// Returns true if the given `type` is a 0-D or 1-D MemRef.
static bool isRankZeroOrOneMemRef(Type type) {
  if (auto memrefType = type.dyn_cast<MemRefType>()) {
    return memrefType.hasRank() && memrefType.getRank() <= 1;
  }
  return false;
}

/// Flattens n-D MemRef to 1-D MemRef and allows other types.
struct FlattenMemRefTypeConverter final : public TypeConverter {
  FlattenMemRefTypeConverter() {
    // Allow all other types.
    addConversion([](Type type) -> std::optional<Type> { return type; });

    // Convert n-D MemRef to 1-D MemRef.
    addConversion([](MemRefType type) -> std::optional<Type> {
      int64_t offset;
      SmallVector<int64_t> strides;
      if (failed(getStridesAndOffset(type, strides, offset))) {
        return nullptr;
      }
      // Since the memref gets linearized, use a stride 1, offset 0.
      StridedLayoutAttr layoutAttr;
      if (offset != 0) {
        layoutAttr = StridedLayoutAttr::get(type.getContext(), offset, {1});
      }
      int64_t staticShape =
          type.hasStaticShape() ? type.getNumElements() : ShapedType::kDynamic;
      return MemRefType::get(staticShape, type.getElementType(), layoutAttr,
                             type.getMemorySpace());
    });
  }
};

//===----------------------------------------------------------------------===//
// Flattening Patterns
//===----------------------------------------------------------------------===//

/// Creates a value for the total element count in `shape`, which may have
/// dynamic dimensions in `dynamicDims`.
static Value createTotalElementCountValue(ShapedType type,
                                          ValueRange dynamicDims, Location loc,
                                          OpBuilder &builder) {
  MLIRContext *context = builder.getContext();

  if (type.hasStaticShape()) {
    assert(dynamicDims.empty());
    return builder.create<arith::ConstantIndexOp>(loc, type.getNumElements());
  }

  int dynamicDimIndex = 0;
  SmallVector<Value, 4> dims;
  auto shape = type.getShape();
  AffineExpr sizeExpr = getAffineConstantExpr(1, context);
  for (int i = 0; i < shape.size(); ++i) {
    sizeExpr = sizeExpr * getAffineSymbolExpr(i, context);
    if (ShapedType::isDynamic(shape[i])) {
      dims.push_back(dynamicDims[dynamicDimIndex++]);
    } else {
      dims.push_back(builder.create<arith::ConstantIndexOp>(loc, shape[i]));
    }
  }
  return affine::makeComposedAffineApply(builder, loc, sizeExpr, dims);
}

// Flattens memref allocation ops with more than 1 dimensions to 1 dimension.
template <typename AllocOpTy>
struct FlattenAlloc final : public OpConversionPattern<AllocOpTy> {
  using OpConversionPattern<AllocOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AllocOpTy allocOp, typename AllocOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto oldType = allocOp.getType().template dyn_cast<MemRefType>();
    if (!oldType || !oldType.getLayout().isIdentity()) return failure();

    Value dynamicDim = createTotalElementCountValue(
        oldType, allocOp.getDynamicSizes(), allocOp.getLoc(), rewriter);
    Type newType = this->getTypeConverter()->convertType(oldType);

    rewriter.replaceOpWithNewOp<AllocOpTy>(allocOp, newType.cast<MemRefType>(),
                                           ValueRange{dynamicDim});

    return success();
  }
};

/// Flattens memref global ops with more than 1 dimensions to 1 dimension.
struct FlattenGlobal final : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  static Attribute flattenAttribute(Attribute value, ShapedType newType) {
    if (!value) return value;
    if (auto splatAttr = value.dyn_cast<SplatElementsAttr>()) {
      return splatAttr.reshape(newType);
    } else if (auto denseAttr = value.dyn_cast<DenseElementsAttr>()) {
      return denseAttr.reshape(newType);
    }
    return {};
  }

  LogicalResult matchAndRewrite(
      memref::GlobalOp globalOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto oldType = globalOp.getType().dyn_cast<MemRefType>();
    if (!oldType || !oldType.getLayout().isIdentity()) return failure();

    auto tensorType = RankedTensorType::get({oldType.getNumElements()},
                                            oldType.getElementType());
    auto memRefType =
        MemRefType::get({oldType.getNumElements()}, oldType.getElementType(),
                        AffineMap(), oldType.getMemorySpace());
    auto newInitialValue =
        flattenAttribute(globalOp.getInitialValueAttr(), tensorType);
    rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        globalOp, globalOp.getSymName(), globalOp.getSymVisibilityAttr(),
        memRefType, newInitialValue, globalOp.getConstant(),
        /*alignment=*/IntegerAttr());
    return success();
  }
};

/// Flattens memref global load ops with more than 1 dimensions to 1 dimension.
struct FlattenGetGlobal final
    : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::GetGlobalOp getOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto oldType = getOp.getType().dyn_cast<MemRefType>();
    if (!oldType || !oldType.getLayout().isIdentity()) return failure();

    auto globalOp = dyn_cast_or_null<memref::GlobalOp>(
        SymbolTable::lookupNearestSymbolFrom(getOp, getOp.getNameAttr()));
    if (!globalOp) return failure();

    auto loadedValue = rewriter.createOrFold<memref::GetGlobalOp>(
        getOp.getLoc(), globalOp.getType(), getOp.getNameAttr());

    auto newType = getTypeConverter()->convertType(oldType).cast<ShapedType>();
    rewriter.replaceOpWithNewOp<memref::CastOp>(getOp, newType, loadedValue);
    return success();
  }
};

/// Flattens memref subspan ops with more than 1 dimensions to 1 dimension.
struct FlattenBindingSubspan final
    : public OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceBindingSubspanOp subspanOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto oldType = subspanOp.getType().dyn_cast<MemRefType>();
    // IREE subspan ops only use memref types with the default identity
    // layout maps.
    if (!oldType) return failure();

    OpFoldResult linearShape;
    if (oldType.hasStaticShape()) {
      linearShape = rewriter.getIndexAttr(oldType.getNumElements());
    } else {
      linearShape = createTotalElementCountValue(
          oldType, subspanOp.getDynamicDims(), subspanOp.getLoc(), rewriter);
    }
    OpFoldResult linearShapeWithoutOffset = linearShape;

    // Check if the subspan has offset. Convert the subspan into a new subpan
    // of zero offset with size = linearize(original shape) + byteOffset /
    // element-width.
    auto byteOffset = subspanOp.getByteOffset();
    Location loc = subspanOp.getLoc();
    OpFoldResult elementOffset = rewriter.getIndexAttr(0);
    if (byteOffset && !matchPattern(byteOffset, m_Zero())) {
      elementOffset = convertByteOffsetToElementOffset(
          rewriter, loc, byteOffset, oldType.getElementType());
      AffineExpr s0, s1;
      bindSymbols(rewriter.getContext(), s0, s1);
      linearShape = affine::makeComposedFoldedAffineApply(
          rewriter, loc, s0 + s1, {linearShape, elementOffset});
    }

    SmallVector<int64_t, 1> staticShape;
    SmallVector<Value, 1> dynamicShape;
    dispatchIndexOpFoldResult(linearShape, dynamicShape, staticShape);
    auto newType =
        MemRefType::get(staticShape, oldType.getElementType(),
                        MemRefLayoutAttrInterface(), oldType.getMemorySpace());

    auto newOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto newOp = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp.getLoc(), newType, subspanOp.getSet(), subspanOp.getBinding(),
        subspanOp.getDescriptorType(), newOffset, dynamicShape,
        subspanOp.getAlignmentAttr(), subspanOp.getDescriptorFlagsAttr());

    Value replacement = newOp;
    if (!isConstantIntValue(elementOffset, 0)) {
      OpFoldResult stride = rewriter.getIndexAttr(1);
      MemRefType returnType =
          oldType.getRank() == 0
              ? memref::SubViewOp::inferRankReducedResultType(
                    {}, newType, elementOffset, linearShapeWithoutOffset,
                    stride)
                    .cast<MemRefType>()
              : nullptr;
      replacement = rewriter.create<memref::SubViewOp>(
          loc, returnType, newOp, elementOffset, linearShapeWithoutOffset,
          OpFoldResult(rewriter.getIndexAttr(1)));
    }

    rewriter.replaceOp(subspanOp, replacement);
    return success();
  }
};

/// Flatten `memref` operands and results of `memref.reinterpret_cast` op.
// TODO(ravishankarm): For now just handle the case where the result is 0D
// memref, and offset is 0. This is how void pointers are modeled. Generalize if
// necessary.
struct FlattenReinterpretCast
    : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::ReinterpretCastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getResultRank() != 0) {
      return rewriter.notifyMatchFailure(
          op, "unhandled op with non-zero rank memref return type");
    }

    if (!isConstantIntValue(op.getConstifiedMixedOffset(), 0)) {
      return rewriter.notifyMatchFailure(op, "unhandled non-zero offset");
    }

    rewriter.updateRootInPlace(op,
                               [&] { op->setOperand(0, adaptor.getSource()); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Linearizing Patterns
//===----------------------------------------------------------------------===//

/// Generates IR to perform index linearization with the given `indices`
/// indexing into the given memref `sourceValue`.
static Value linearizeIndices(Value sourceValue, ValueRange indices,
                              Location loc, OpBuilder &builder) {
  MemRefType sourceType = sourceValue.getType().cast<MemRefType>();
  assert(sourceType.hasRank());

  int64_t rank = sourceType.getRank();
  assert(rank > 0);

  // First try to get the strides from the MemRef type itself. This applies to
  // cases where we have static shapes and only the leading dimension is
  // dynamic.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (succeeded(getStridesAndOffset(sourceType, strides, offset))) {
    // The memref itself might have an offset, but we should not account for it
    // when computing the linearization. The original memref might be
    // `memref<?x?xf32, strided<[?, ?], offset: ?>`
    // where shape is `{%d0, %d1}`, strides are `{%s0, %s1}` and offset is
    // `%offset`. The interpretation of that is the actual memref starts at
    // `%offset` from the base pointer. After linearization, the offset remains,
    // but the shape is 1D. So build a map with the same strides, but 0 offset.
    AffineMap linearLayoutMap =
        makeStridedLinearLayoutMap(strides, 0, builder.getContext());
    // Dynamic strides/offset will create symbols. There should be none for the
    // static case.
    if (linearLayoutMap.getNumSymbols() == 0) {
      return affine::makeComposedAffineApply(builder, loc, linearLayoutMap,
                                             indices);
    }
  }

  // Then try to see if the source op carries the dynamic dimensions itself.
  // If so we can still get the strides for dimensions to linearize.
  Operation *sourceOp = sourceValue.getDefiningOp();
  SmallVector<Value, 4> dims;
  dims.reserve(rank);
  if (auto shapeAwareOp =
          dyn_cast<IREE::Util::ShapeAwareOpInterface>(sourceOp)) {
    dims = shapeAwareOp.buildResultValueShape(sourceValue, builder);
  } else {
    auto getDimValues = [&](MemRefType type, ValueRange dynamicDims) {
      auto shape = type.getShape();
      int dynamicDimIndex = 0;
      for (int i = 0; i < shape.size(); ++i) {
        if (ShapedType::isDynamic(shape[i])) {
          dims.push_back(dynamicDims[dynamicDimIndex++]);
        } else {
          dims.push_back(builder.create<arith::ConstantIndexOp>(loc, shape[i]));
        }
      }
    };

    if (auto allocOp = dyn_cast<memref::AllocOp>(sourceOp)) {
      getDimValues(sourceType, allocOp.getDynamicSizes());
    } else if (auto allocaOp = dyn_cast<memref::AllocaOp>(sourceOp)) {
      getDimValues(sourceType, allocaOp.getDynamicSizes());
    } else {
      if (sourceType.hasStaticShape()) {
        for (int64_t dim : sourceType.getShape()) {
          dims.push_back(builder.create<arith::ConstantIndexOp>(loc, dim));
        }
      } else {
        return nullptr;
      }
    }
  }

  AffineExpr sym0, sym1, sym2;
  bindSymbols(builder.getContext(), sym0, sym1, sym2);
  MLIRContext *context = builder.getContext();
  auto mulAddMap = AffineMap::get(0, 3, {sym0 * sym1 + sym2}, context);

  Value linearIndex = indices.front();
  for (int i = 1; i < indices.size(); ++i) {
    linearIndex = builder.create<affine::AffineApplyOp>(
        loc, mulAddMap, ValueRange{linearIndex, dims[i], indices[i]});
  }
  return linearIndex;
}

/// Flattens memref subspan ops with more than 1 dimensions to 1 dimension.
struct FlattenSubView final : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::SubViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(adaptor.getSource().getType())) {
      return rewriter.notifyMatchFailure(
          op, "expected converted memref of rank <= 1");
    }
    Type neededResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!neededResultType || !isRankZeroOrOneMemRef(neededResultType))
      return failure();
    Value size = createTotalElementCountValue(op.getType(), op.sizes(),
                                              op.getLoc(), rewriter);
    SmallVector<Value> offsets = mlir::getValueOrCreateConstantIndexOp(
        rewriter, op.getLoc(), op.getMixedOffsets());
    Value linearOffset =
        linearizeIndices(op.getSource(), offsets, op.getLoc(), rewriter);
    Value stride = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    Value newSubView = rewriter.create<memref::SubViewOp>(
        op.getLoc(), adaptor.getSource(), ValueRange({linearOffset}),
        ValueRange({size}), ValueRange({stride}));
    rewriter.replaceOpWithNewOp<memref::CastOp>(op, neededResultType,
                                                newSubView);
    return success();
  }
};

/// Linearizes indices in memref.load ops.
struct LinearizeLoadIndices final : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::LoadOp loadOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(adaptor.getMemref().getType())) {
      return rewriter.notifyMatchFailure(
          loadOp, "expected converted memref of rank <= 1");
    }

    Value linearIndex = linearizeIndices(
        loadOp.getMemref(), loadOp.getIndices(), loadOp.getLoc(), rewriter);
    if (!linearIndex) {
      return loadOp.emitOpError() << "failed to linearize index";
    }

    rewriter.replaceOpWithNewOp<memref::LoadOp>(loadOp, adaptor.getMemref(),
                                                linearIndex);
    return success();
  }
};

/// Linearizes indices in gpu.subgroup_mma_load_matrix ops.
struct LinearizeMMALoadIndices final
    : public OpConversionPattern<gpu::SubgroupMmaLoadMatrixOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      gpu::SubgroupMmaLoadMatrixOp loadOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(adaptor.getSrcMemref().getType())) {
      return rewriter.notifyMatchFailure(
          loadOp, "expected converted memref of rank <= 1");
    }

    Value linearIndex = linearizeIndices(
        loadOp.getSrcMemref(), loadOp.getIndices(), loadOp.getLoc(), rewriter);
    if (!linearIndex) {
      return loadOp.emitOpError() << "failed to linearize index";
    }

    rewriter.replaceOpWithNewOp<gpu::SubgroupMmaLoadMatrixOp>(
        loadOp, loadOp.getType(), adaptor.getSrcMemref(), linearIndex,
        loadOp.getLeadDimension(), loadOp.getTransposeAttr());
    return success();
  }
};

/// Linearizes indices in memref.store ops.
struct LinearizeStoreIndices final
    : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::StoreOp storeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(adaptor.getMemref().getType())) {
      return rewriter.notifyMatchFailure(
          storeOp, "expected converted memref of rank <= 1");
    }

    Value linearIndex = linearizeIndices(
        storeOp.getMemref(), storeOp.getIndices(), storeOp.getLoc(), rewriter);
    if (!linearIndex) {
      return storeOp.emitOpError() << "failed to linearize index";
    }

    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        storeOp, adaptor.getValue(), adaptor.getMemref(), linearIndex);
    return success();
  }
};

/// Linearizes indices in gpu.subgroup_mma_store_matrix ops.
struct LinearizeMMAStoreIndices final
    : public OpConversionPattern<gpu::SubgroupMmaStoreMatrixOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      gpu::SubgroupMmaStoreMatrixOp storeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(adaptor.getDstMemref().getType())) {
      return rewriter.notifyMatchFailure(
          storeOp, "expected converted memref of rank <= 1");
    }

    Value linearIndex =
        linearizeIndices(storeOp.getDstMemref(), storeOp.getIndices(),
                         storeOp.getLoc(), rewriter);
    if (!linearIndex) {
      return storeOp.emitOpError() << "failed to linearize index";
    }

    rewriter.replaceOpWithNewOp<gpu::SubgroupMmaStoreMatrixOp>(
        storeOp, adaptor.getSrc(), adaptor.getDstMemref(), linearIndex,
        storeOp.getLeadDimension(), storeOp.getTransposeAttr());
    return success();
  }
};

/// Linearizes indices in vector.transfer_read ops.
struct LinearizeTransferReadIndices final
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferReadOp transferReadOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!transferReadOp.getPermutationMap().isMinorIdentity()) {
      return rewriter.notifyMatchFailure(
          transferReadOp,
          "cannot convert op with non-minor identity "
          "map");
    }
    if (!isRankZeroOrOneMemRef(adaptor.getSource().getType())) {
      return rewriter.notifyMatchFailure(
          transferReadOp, "expected converted memref of rank <= 1");
    }
    Value linearIndex = linearizeIndices(transferReadOp.getSource(),
                                         transferReadOp.getIndices(),
                                         transferReadOp.getLoc(), rewriter);
    if (!linearIndex) {
      return transferReadOp.emitOpError() << "failed to linearize index";
    }

    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        transferReadOp, transferReadOp.getVectorType(), adaptor.getSource(),
        linearIndex, AffineMapAttr::get(rewriter.getDimIdentityMap()),
        transferReadOp.getPadding(), /*mask=*/Value(),
        transferReadOp.getInBoundsAttr());
    return success();
  }
};

/// Linearizes indices in vector.transfer_write ops.
struct LinearizeTransferWriteIndices final
    : public OpConversionPattern<vector::TransferWriteOp> {
  using OpConversionPattern<vector::TransferWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferWriteOp transferWriteOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!transferWriteOp.getPermutationMap().isMinorIdentity()) {
      return rewriter.notifyMatchFailure(
          transferWriteOp,
          "cannot convert op with non-minor identity "
          "map");
    }
    if (!isRankZeroOrOneMemRef(adaptor.getSource().getType())) {
      return rewriter.notifyMatchFailure(
          transferWriteOp, "expected converted memref of rank <= 1");
    }
    Value linearIndex = linearizeIndices(transferWriteOp.getSource(),
                                         transferWriteOp.getIndices(),
                                         transferWriteOp.getLoc(), rewriter);
    if (!linearIndex) {
      return transferWriteOp.emitOpError() << "failed to linearize index";
    }

    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        transferWriteOp, adaptor.getVector(), adaptor.getSource(), linearIndex,
        AffineMapAttr::get(rewriter.getDimIdentityMap()),
        transferWriteOp.getInBoundsAttr());
    return success();
  }
};

/// Adjusts unrealized_conversion_cast ops' inputs to flattened memref values.
struct AdjustConversionCast final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      UnrealizedConversionCastOp castOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (castOp->getNumOperands() != 1) return failure();

    Value input = adaptor.getOperands().front();
    // We only want to handle cases where the cast op handles memref types.
    if (!input.getType().isa<BaseMemRefType>()) return failure();

    if (!isRankZeroOrOneMemRef(input.getType())) {
      return rewriter.notifyMatchFailure(
          castOp, "expected converted memref of rank <= 1");
    }
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        castOp, castOp.getResultTypes(), input);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Folding Patterns
//===----------------------------------------------------------------------===//

/// Removes MemRef reshape ops given that we'll linearize both the source and
/// target type to the same one.
template <typename ReshapeOpTy>
struct FoldMemRefReshape final : public OpConversionPattern<ReshapeOpTy> {
  using OpConversionPattern<ReshapeOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReshapeOpTy op, typename ReshapeOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto typeConverter = OpConversionPattern<ReshapeOpTy>::typeConverter;
    if (!isRankZeroOrOneMemRef(adaptor.getSrc().getType())) {
      return rewriter.notifyMatchFailure(
          op, "expected converted memref of rank <= 1");
    }

    // If the types are the same, just elide. Otherwise, introduce a cast
    // so long as both are 1D. This is most often the result of needing
    // to become more static (i.e. memref<?xf32> -> memref<5xf32>).
    // Refuse to match if the cast would be illegal.
    Type newSourceType = adaptor.getSrc().getType();
    Type neededResultType =
        typeConverter->convertType(op.getResult().getType());
    if (!neededResultType) return failure();
    if (newSourceType == neededResultType) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    } else if (isRankZeroOrOneMemRef(neededResultType)) {
      rewriter.replaceOpWithNewOp<memref::CastOp>(op, neededResultType,
                                                  adaptor.getSrc());
      return success();
    }
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &d) {
      d << "replacement type incompatible (" << newSourceType << " vs "
        << neededResultType << ")";
    });
  };
};

/// Returns the number of bytes of the given `type`. Returns std::nullopt if
/// cannot deduce.
///
/// Note that this should be kept consistent with how the byte offset was
/// calculated in the subspan ops!
std::optional<int64_t> getNumBytes(Type type) {
  if (type.isIntOrFloat()) return IREE::Util::getRoundedElementByteWidth(type);
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    auto elementBytes = getNumBytes(vectorType.getElementType());
    if (!elementBytes) return std::nullopt;
    return elementBytes.value() * vectorType.getNumElements();
  }
  return std::nullopt;
}

/// Erase alignment hints.
struct RemoveAssumeAlignOp
    : public OpRewritePattern<memref::AssumeAlignmentOp> {
 public:
  using OpRewritePattern<memref::AssumeAlignmentOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AssumeAlignmentOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Removes memref.cast that turns static shapes into dynamic shapes.
struct RemoveDynamicCastOp final : public OpRewritePattern<memref::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto srcType = castOp.getSource().getType().cast<MemRefType>();
    auto dstType = castOp.getType().cast<MemRefType>();
    // Restrict to the cases we generate in this pass--1-D static shape to 1-D
    // dynamic shape.
    if (srcType.getRank() == 1 && srcType.hasStaticShape() &&
        dstType.getRank() == 1 && !dstType.hasStaticShape()) {
      rewriter.replaceOp(castOp, castOp.getSource());
      return success();
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct FlattenMemRefSubspanPass
    : public FlattenMemRefSubspanBase<FlattenMemRefSubspanPass> {
  FlattenMemRefSubspanPass() {}
  FlattenMemRefSubspanPass(const FlattenMemRefSubspanPass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    // First flatten the dimensions of subspan op and their consumer load/store
    // ops. This requires setting up conversion targets with type converter.

    MLIRContext *context = &getContext();

    // This pass currently doesn't support alignment hints so remove them first.
    RewritePatternSet patterns(context);
    patterns.add<RemoveAssumeAlignOp>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));

    RewritePatternSet flattenPatterns(context);

    // Interface binding subspan ops represents allocations from the runtime. We
    // convert all MemRef types "generated" by them to 1-D one, static for
    // uniform buffers and dynamic for storage buffers. This matches how IREE
    // models runtime buffers nicely.
    FlattenMemRefTypeConverter interfaceTypeConverter;
    interfaceTypeConverter.addConversion(
        [](MemRefType type) -> std::optional<Type> {
          // 0-D MemRef types can be used to represent raw pointers for
          // micro-kernel ABI purposes. Specially allow it.
          if (isRankZeroMemRef(type)) return type;

          // Fall back to the default conversion flow.
          return std::nullopt;
        });
    flattenPatterns.add<FlattenBindingSubspan>(interfaceTypeConverter, context);

    // Other ops generate MemRef values representing internal allocations (e.g.,
    // on stack for GPU, in shared memory for GPU) or data embedded in the
    // kernel. We may not be able to go fully dynamic (e.g., memref::GlobalOp).
    // Still convert everything to 1-D though.
    FlattenMemRefTypeConverter internalTypeConverter;
    internalTypeConverter.addConversion(
        [](MemRefType type) -> std::optional<Type> {
          // 0-D or 1-D MemRef types are okay.
          if (isRankZeroOrOneMemRef(type)) return type;

          // Fall back to the default conversion flow.
          return std::nullopt;
        });
    flattenPatterns.add<
        FlattenAlloc<memref::AllocaOp>, FlattenAlloc<memref::AllocOp>,
        FlattenGlobal, FlattenGetGlobal, FlattenReinterpretCast,
        LinearizeLoadIndices, LinearizeMMALoadIndices, LinearizeStoreIndices,
        LinearizeMMAStoreIndices, LinearizeTransferReadIndices,
        LinearizeTransferWriteIndices, AdjustConversionCast, FlattenSubView,
        FoldMemRefReshape<memref::CollapseShapeOp>,
        FoldMemRefReshape<memref::ExpandShapeOp>>(internalTypeConverter,
                                                  context);

    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addDynamicallyLegalOp<memref::AllocaOp, memref::AllocOp,
                                 memref::GetGlobalOp>([](Operation *op) {
      return isRankZeroOrOneMemRef(op->getResultTypes().front());
    });
    target
        .addDynamicallyLegalOp<memref::CollapseShapeOp, memref::ExpandShapeOp>(
            [](Operation *op) {
              return isRankZeroOrOneMemRef(op->getResultTypes().front()) &&
                     isRankZeroOrOneMemRef(op->getOperandTypes().front());
            });
    target.addDynamicallyLegalOp<IREE::HAL::InterfaceBindingSubspanOp>(
        [&](IREE::HAL::InterfaceBindingSubspanOp op) {
          if (!isRankZeroOrOneMemRef(op.getType())) {
            return false;
          }
          auto byteOffset = op.getByteOffset();
          return !byteOffset || matchPattern(byteOffset, m_Zero());
        });
    target.addDynamicallyLegalOp<memref::GlobalOp>([](memref::GlobalOp op) {
      return isRankZeroOrOneMemRef(op.getType());
    });
    target.addDynamicallyLegalOp<memref::LoadOp>([](memref::LoadOp loadOp) {
      return isRankZeroOrOneMemRef(loadOp.getMemRefType());
    });
    target.addDynamicallyLegalOp<memref::ReinterpretCastOp>(
        [](memref::ReinterpretCastOp castOp) {
          return isRankZeroOrOneMemRef(castOp.getSource().getType());
        });
    target.addDynamicallyLegalOp<gpu::SubgroupMmaLoadMatrixOp>(
        [](gpu::SubgroupMmaLoadMatrixOp loadOp) {
          return isRankZeroOrOneMemRef(loadOp.getSrcMemref().getType());
        });
    target.addDynamicallyLegalOp<memref::StoreOp>([](memref::StoreOp storeOp) {
      return isRankZeroOrOneMemRef(storeOp.getMemRefType());
    });
    target.addDynamicallyLegalOp<gpu::SubgroupMmaStoreMatrixOp>(
        [](gpu::SubgroupMmaStoreMatrixOp storeOp) {
          return isRankZeroOrOneMemRef(storeOp.getDstMemref().getType());
        });
    target.addDynamicallyLegalOp<vector::TransferReadOp>(
        [](vector::TransferReadOp readOp) {
          return isRankZeroOrOneMemRef(
              readOp.getSource().getType().cast<MemRefType>());
        });
    target.addDynamicallyLegalOp<vector::TransferWriteOp>(
        [](vector::TransferWriteOp writeOp) {
          return isRankZeroOrOneMemRef(
              writeOp.getSource().getType().cast<MemRefType>());
        });
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](UnrealizedConversionCastOp castOp) {
          if (castOp->getNumOperands() != 1) return false;

          Type inputType = castOp->getOperandTypes().front();
          return !inputType.isa<BaseMemRefType>() ||
                 isRankZeroOrOneMemRef(inputType);
        });
    target.addDynamicallyLegalOp<memref::SubViewOp>([](memref::SubViewOp op) {
      return isRankZeroOrOneMemRef(op.getType());
    });

    // Use partial conversion here so that we can ignore allocations created
    // by promotion and their load/store ops.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(flattenPatterns)))) {
      return signalPassFailure();
    }

    // Fold subviews if any new oportuinity has been created.
    RewritePatternSet foldSubviewPatterns(context);
    memref::populateFoldMemRefAliasOpPatterns(foldSubviewPatterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(foldSubviewPatterns)))) {
      return signalPassFailure();
    }

    // Finally run canonicalization patterns for memref allocations. This helps
    // to fold, e.g., `memref.alloc(%c24) : memref<?xf32, 3>` into
    // `memref.alloc() : memref<24xf32, 3>` for later steps depending on static
    // shaped memref allocations.
    RewritePatternSet cleanupPatterns(context);
    memref::AllocOp::getCanonicalizationPatterns(cleanupPatterns, context);
    memref::AllocaOp::getCanonicalizationPatterns(cleanupPatterns, context);
    memref::SubViewOp::getCanonicalizationPatterns(cleanupPatterns, context);
    cleanupPatterns.add<RemoveDynamicCastOp>(context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(cleanupPatterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createFlattenMemRefSubspanPass() {
  return std::make_unique<FlattenMemRefSubspanPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
