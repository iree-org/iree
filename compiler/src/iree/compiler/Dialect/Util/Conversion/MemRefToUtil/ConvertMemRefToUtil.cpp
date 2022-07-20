// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Conversion/MemRefToUtil/ConvertMemRefToUtil.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Returns true if the given `type` is a MemRef of rank 0 or 1.
static bool isRankZeroOrOneMemRef(Type type) {
  if (auto memrefType = type.dyn_cast<MemRefType>()) {
    return memrefType.hasRank() && memrefType.getRank() <= 1 &&
           memrefType.getLayout().isIdentity();
  }
  return false;
}

/// Returns the offset, in bytes, of an index within a linearized dense buffer.
/// Expects that the |memrefValue| has been linearized already.
static Value getBufferOffset(Location loc, Value memrefValue,
                             ValueRange indices, Type elementType,
                             ConversionPatternRewriter &rewriter) {
  auto memrefType = memrefValue.getType().cast<ShapedType>();
  if (memrefType.getRank() == 0) {
    // Rank 0 buffers (like memref<i32>) have only a single valid offset at 0.
    return rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
  }
  assert(memrefType.getRank() == 1 && "memrefs should have been flattened");

  // Element type byte length as the base.
  auto elementSize = rewriter.createOrFold<arith::ConstantIndexOp>(
      loc, IREE::Util::getRoundedElementByteWidth(elementType));

  // Rank 1 memrefs are just offset by their element width by the offset.
  auto elementCount = indices.front();
  return rewriter.create<arith::MulIOp>(loc, elementSize, elementCount);
}

/// Pattern to lower operations that become a no-ops at this level.
/// Passes through operands to results.
template <typename OpTy>
struct FoldAsNoOp final : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
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
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertMemRefGlobalOp : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      memref::GlobalOp globalOp, OpAdaptor adaptor,
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
        globalOp.getLoc(), initializerBuilder.getType<IREE::Util::BufferType>(),
        globalOp.getInitialValueAttr(), alignmentAttr);
    initializerBuilder.create<IREE::Util::GlobalStoreOp>(
        globalOp.getLoc(), constantOp.getResult(), newOp.getName());
    initializerBuilder.create<IREE::Util::InitializerReturnOp>(
        globalOp.getLoc());

    return success();
  }
};

struct ConvertMemRefGetGlobalOp
    : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      memref::GetGlobalOp getOp, OpAdaptor adaptor,
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
  LogicalResult matchAndRewrite(
      memref::AllocaOp allocaOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto type = allocaOp.getType().cast<ShapedType>();
    if (!type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          allocaOp, "unable to create buffers for dynamic shapes");
    }
    int64_t memRefLength =
        type.getNumElements() *
        IREE::Util::getRoundedElementByteWidth(type.getElementType());
    Value allocationSize = rewriter.create<arith::ConstantIndexOp>(
        allocaOp.getLoc(), memRefLength);
    rewriter.replaceOpWithNewOp<IREE::Util::BufferAllocOp>(
        allocaOp, rewriter.getType<IREE::Util::BufferType>(), allocationSize);
    return success();
  }
};

struct ConvertMemRefDimOp : public OpConversionPattern<memref::DimOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      memref::DimOp dimOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(dimOp.getSource().getType())) {
      return rewriter.notifyMatchFailure(
          dimOp, "only rank-0 and rank-1 memrefs are supported; flatten first");
    }
    auto newElementType = getTypeConverter()->convertType(
        dimOp.getSource().getType().cast<MemRefType>().getElementType());
    if (!newElementType) {
      return rewriter.notifyMatchFailure(dimOp, "unsupported element type");
    }
    Value elementSize = rewriter.create<arith::ConstantIndexOp>(
        dimOp.getLoc(), IREE::Util::getRoundedElementByteWidth(newElementType));
    Value bufferSize = rewriter.create<IREE::Util::BufferSizeOp>(
        dimOp.getLoc(), rewriter.getIndexType(), adaptor.getSource());
    rewriter.replaceOpWithNewOp<arith::FloorDivSIOp>(dimOp, bufferSize,
                                                     elementSize);
    return success();
  }
};

struct ConvertMemRefLoadOp : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      memref::LoadOp loadOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(loadOp.getMemref().getType())) {
      return rewriter.notifyMatchFailure(
          loadOp,
          "only rank-0 and rank-1 memrefs are supported; flatten first");
    }
    auto oldType = loadOp.getResult().getType();
    auto newType = getTypeConverter()->convertType(oldType);
    auto newElementType = getTypeConverter()->convertType(
        loadOp.getMemRef().getType().cast<MemRefType>().getElementType());
    if (!newElementType) {
      return rewriter.notifyMatchFailure(loadOp, "unsupported element type");
    }
    auto memRefSize = rewriter.createOrFold<IREE::Util::BufferSizeOp>(
        loadOp.getLoc(), rewriter.getIndexType(), adaptor.getMemref());
    auto byteOffset =
        getBufferOffset(loadOp.getLoc(), loadOp.getMemref(),
                        loadOp.getIndices(), newElementType, rewriter);
    rewriter.replaceOpWithNewOp<IREE::Util::BufferLoadOp>(
        loadOp, newType, adaptor.getMemref(), memRefSize, byteOffset);
    return success();
  }
};

struct ConvertMemRefStoreOp : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      memref::StoreOp storeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(storeOp.getMemref().getType())) {
      return rewriter.notifyMatchFailure(
          storeOp,
          "only rank-0 and rank-1 memrefs are supported; flatten first");
    }
    auto newElementType = getTypeConverter()->convertType(
        storeOp.getMemRef().getType().cast<MemRefType>().getElementType());
    if (!newElementType) {
      return rewriter.notifyMatchFailure(storeOp, "unsupported element type");
    }
    auto memRefSize = rewriter.createOrFold<IREE::Util::BufferSizeOp>(
        storeOp.getLoc(), rewriter.getIndexType(), adaptor.getMemref());
    auto byteOffset =
        getBufferOffset(storeOp.getLoc(), storeOp.getMemref(),
                        storeOp.getIndices(), newElementType, rewriter);
    rewriter.replaceOpWithNewOp<IREE::Util::BufferStoreOp>(
        storeOp, adaptor.getValue(), adaptor.getMemref(), memRefSize,
        byteOffset);
    return success();
  }
};

}  // namespace

void populateMemRefToUtilPatterns(MLIRContext *context,
                                  ConversionTarget &conversionTarget,
                                  TypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  conversionTarget.addIllegalDialect<memref::MemRefDialect>();

  typeConverter.addConversion([&](MemRefType type) -> llvm::Optional<Type> {
    if (isRankZeroOrOneMemRef(type)) {
      return IREE::Util::BufferType::get(type.getContext());
    }
    return llvm::None;
  });

  // Unranked memrefs are emitted for library call integration when we just
  // need void* semantics. An unranked memref is basically just a (pointer,
  // memory-space, element-type).
  typeConverter.addConversion(
      [&](UnrankedMemRefType type) -> llvm::Optional<Type> {
        return IREE::Util::BufferType::get(type.getContext());
      });

  patterns
      .insert<FoldAsNoOp<bufferization::ToMemrefOp>,
              ElideNoOp<memref::AssumeAlignmentOp>, FoldAsNoOp<memref::CastOp>>(
          typeConverter, context);
  patterns.insert<ConvertMemRefGlobalOp, ConvertMemRefGetGlobalOp,
                  ConvertMemRefAllocaOp, ConvertMemRefDimOp,
                  ConvertMemRefLoadOp, ConvertMemRefStoreOp>(typeConverter,
                                                             context);
}

}  // namespace iree_compiler
}  // namespace mlir
