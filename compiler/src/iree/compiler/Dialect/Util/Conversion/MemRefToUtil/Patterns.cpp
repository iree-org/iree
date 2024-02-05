// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Conversion/MemRefToUtil/Patterns.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

/// Returns true if the given `type` is a MemRef of rank 0 or 1.
static bool isRankZeroOrOneMemRef(Type type) {
  if (auto memrefType = llvm::dyn_cast<MemRefType>(type)) {
    return memrefType.hasRank() && memrefType.getRank() <= 1 &&
           memrefType.getLayout().isIdentity();
  }
  return false;
}

static Value getElementTypeByteSize(OpBuilder &builder, Location loc,
                                    Value memrefValue) {
  auto elementType =
      llvm::cast<ShapedType>(memrefValue.getType()).getElementType();
  return builder.createOrFold<IREE::Util::SizeOfOp>(loc, elementType);
}

/// Returns the offset, in bytes, of an index within a linearized dense buffer.
/// Expects that the |memrefValue| has been linearized already. This function
/// only takes a `ValueRange indices` because that's more convenient for callers
/// but in practice it only uses `indices[0]`.
///
static Value getByteOffsetForIndices(OpBuilder &builder, Location loc,
                                     Value memrefValue, ValueRange indices,
                                     Value elementTypeByteSize) {
  auto memrefType = llvm::cast<MemRefType>(memrefValue.getType());
  if (memrefType.getRank() == 0) {
    // Rank 0 buffers (like memref<i32>) have only a single valid offset at 0.
    return builder.createOrFold<arith::ConstantIndexOp>(loc, 0);
  }
  if (memrefType.getRank() != 1) {
    emitError(loc, "memrefs should have been flattened");
    return {};
  }
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memrefType, strides, offset)) ||
      strides[0] != 1) {
    emitError(loc, "expected memref stride 1");
    return {};
  }

  // Rank 1 memrefs are just offset by their element width by the offset.
  auto elementCount = indices[0];
  return builder.create<arith::MulIOp>(loc, elementTypeByteSize, elementCount);
}

static Value getByteLength(OpBuilder &builder, Location loc,
                           Value memrefValue) {
  auto memrefType = llvm::cast<MemRefType>(memrefValue.getType());
  if (memrefType.getRank() == 0) {
    return getElementTypeByteSize(builder, loc, memrefValue);
  }
  if (memrefType.getRank() != 1) {
    emitError(loc, "memrefs should have been flattened");
    return {};
  }
  Value size = builder.create<memref::DimOp>(loc, memrefValue, 0);
  Value elementTypeByteSize = getElementTypeByteSize(builder, loc, memrefValue);
  return getByteOffsetForIndices(builder, loc, memrefValue, {size},
                                 elementTypeByteSize);
}

/// Pattern to lower operations that become a no-ops at this level.
/// Passes through operands to results.
template <typename OpTy>
struct FoldAsNoOp final : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Pattern to lower operations that become a no-ops at this level.
/// Erases the op entirely.
template <typename OpTy>
struct ElideNoOp final : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertMemRefGlobalOp : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GlobalOp globalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(globalOp.getType())) {
      return rewriter.notifyMatchFailure(
          globalOp,
          "only rank-0 and rank-1 memrefs are supported; flatten first");
    }

    // For mutable values we'd want to either have a RwdataOp or a global
    // !vm.buffer that we initialized with rodata.
    if (!globalOp.getConstant()) {
      return rewriter.notifyMatchFailure(
          globalOp, "mutable global memrefs not yet implemented");
    }

    auto newOp = rewriter.replaceOpWithNewOp<IREE::Util::GlobalOp>(
        globalOp, globalOp.getSymName(), /*isMutable=*/false,
        rewriter.getType<IREE::Util::BufferType>());
    newOp.setPrivate();

    auto initializerOp =
        rewriter.create<IREE::Util::InitializerOp>(globalOp.getLoc());
    auto initializerBuilder =
        OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
    auto alignmentAttr = globalOp.getAlignmentAttr()
                             ? initializerBuilder.getIndexAttr(
                                   globalOp.getAlignmentAttr().getInt())
                             : IntegerAttr{};
    auto constantOp = initializerBuilder.create<IREE::Util::BufferConstantOp>(
        globalOp.getLoc(), /*name=*/nullptr, globalOp.getInitialValueAttr(),
        alignmentAttr, /*mimeType=*/nullptr);
    initializerBuilder.create<IREE::Util::GlobalStoreOp>(
        globalOp.getLoc(), constantOp.getResult(), newOp.getName());
    initializerBuilder.create<IREE::Util::ReturnOp>(globalOp.getLoc());

    return success();
  }
};

struct ConvertMemRefGetGlobalOp
    : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GetGlobalOp getOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(getOp.getResult().getType())) {
      return rewriter.notifyMatchFailure(
          getOp, "only rank-0 and rank-1 memrefs are supported; flatten first");
    }
    rewriter.replaceOpWithNewOp<IREE::Util::GlobalLoadOp>(
        getOp, rewriter.getType<IREE::Util::BufferType>(), getOp.getName());
    return success();
  }
};

struct ConvertMemRefAllocaOp : public OpConversionPattern<memref::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::AllocaOp allocaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = allocaOp.getLoc();
    auto allocationSize = getByteLength(rewriter, loc, allocaOp.getMemref());
    uint64_t alignment = allocaOp.getAlignment().value_or(0);
    rewriter.replaceOpWithNewOp<IREE::Util::BufferAllocOp>(
        allocaOp, rewriter.getType<IREE::Util::BufferType>(), allocationSize,
        alignment ? rewriter.getIndexAttr(alignment) : IntegerAttr{});
    return success();
  }
};

struct ConvertMemRefDimOp : public OpConversionPattern<memref::DimOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::DimOp dimOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(dimOp.getSource().getType())) {
      return rewriter.notifyMatchFailure(
          dimOp, "only rank-0 and rank-1 memrefs are supported; flatten first");
    }
    auto elementType =
        llvm::cast<MemRefType>(dimOp.getSource().getType()).getElementType();
    Value elementSize = rewriter.createOrFold<IREE::Util::SizeOfOp>(
        dimOp.getLoc(), elementType);
    Value bufferSize = rewriter.create<IREE::Util::BufferSizeOp>(
        dimOp.getLoc(), rewriter.getIndexType(), adaptor.getSource());
    rewriter.replaceOpWithNewOp<arith::FloorDivSIOp>(dimOp, bufferSize,
                                                     elementSize);
    return success();
  }
};

struct ConvertMemRefLoadOp : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(loadOp.getMemref().getType())) {
      return rewriter.notifyMatchFailure(
          loadOp,
          "only rank-0 and rank-1 memrefs are supported; flatten first");
    }
    auto oldType = loadOp.getResult().getType();
    auto newType = getTypeConverter()->convertType(oldType);
    Location loc = loadOp.getLoc();
    auto memRefSize = rewriter.createOrFold<IREE::Util::BufferSizeOp>(
        loc, rewriter.getIndexType(), adaptor.getMemref());
    auto elementTypeByteSize =
        getElementTypeByteSize(rewriter, loc, loadOp.getMemref());
    auto byteOffset =
        getByteOffsetForIndices(rewriter, loc, loadOp.getMemref(),
                                loadOp.getIndices(), elementTypeByteSize);
    Value loaded = rewriter.create<IREE::Util::BufferLoadOp>(
        loc, oldType, adaptor.getMemref(), memRefSize, byteOffset,
        elementTypeByteSize);
    if (newType != oldType) {
      // Since the BufferLoadOp semantics include its result type (i.e. a load
      // of an i8 is different than a load of an i32), in the presence of type
      // conversion, we must preserve the original type and emit an unrealized
      // conversion cast for downstreams. In this case, further legalizations
      // will be required to resolve it. This comes up in A->B->C lowerings
      // where the BufferLoad is an intermediate stage.
      loaded = rewriter.create<UnrealizedConversionCastOp>(loc, newType, loaded)
                   .getResult(0);
    }
    rewriter.replaceOp(loadOp, loaded);
    return success();
  }
};

struct ConvertMemRefStoreOp : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(storeOp.getMemref().getType())) {
      return rewriter.notifyMatchFailure(
          storeOp,
          "only rank-0 and rank-1 memrefs are supported; flatten first");
    }
    Location loc = storeOp.getLoc();
    auto memRefSize = rewriter.createOrFold<IREE::Util::BufferSizeOp>(
        loc, rewriter.getIndexType(), adaptor.getMemref());
    auto elementTypeByteSize =
        getElementTypeByteSize(rewriter, loc, storeOp.getMemref());
    auto byteOffset =
        getByteOffsetForIndices(rewriter, loc, storeOp.getMemref(),
                                storeOp.getIndices(), elementTypeByteSize);
    Value newValue = adaptor.getValue();
    if (newValue.getType() != storeOp.getValue().getType()) {
      // In combination with type conversion, the elemental type may change,
      // and this is load bearing with respect to buffer_store op semantics
      // (i.e. storing of an i32 is different from an i8, even if the
      // conversion target widens). Insert an unrealized conversion cast to
      // preserve the original semantic. Presumably, something will clear this
      // with additional lowering.
      newValue = rewriter
                     .create<UnrealizedConversionCastOp>(
                         loc, storeOp.getValue().getType(), newValue)
                     .getResult(0);
    }
    rewriter.replaceOpWithNewOp<IREE::Util::BufferStoreOp>(
        storeOp, newValue, adaptor.getMemref(), memRefSize, byteOffset,
        elementTypeByteSize);
    return success();
  }
};

// Make `reinterpret_cast` a no-op.
struct ConvertMemRefReinterpretCastOp
    : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(castOp, adaptor.getSource());
    return success();
  }
};

} // namespace

void populateMemRefToUtilPatterns(MLIRContext *context,
                                  ConversionTarget &conversionTarget,
                                  TypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  Type convertedBufferType) {
  conversionTarget.addIllegalDialect<memref::MemRefDialect>();

  typeConverter.addConversion(
      [convertedBufferType](MemRefType type) -> std::optional<Type> {
        if (isRankZeroOrOneMemRef(type)) {
          if (convertedBufferType) {
            return convertedBufferType;
          } else {
            return IREE::Util::BufferType::get(type.getContext());
          }
        }
        return std::nullopt;
      });

  patterns
      .insert<FoldAsNoOp<bufferization::ToMemrefOp>,
              ElideNoOp<memref::AssumeAlignmentOp>, FoldAsNoOp<memref::CastOp>>(
          typeConverter, context);
  patterns
      .insert<ConvertMemRefGlobalOp, ConvertMemRefGetGlobalOp,
              ConvertMemRefAllocaOp, ConvertMemRefDimOp, ConvertMemRefLoadOp,
              ConvertMemRefStoreOp, ConvertMemRefReinterpretCastOp>(
          typeConverter, context);
}

} // namespace mlir::iree_compiler
