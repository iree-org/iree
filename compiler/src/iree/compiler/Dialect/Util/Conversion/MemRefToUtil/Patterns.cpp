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
                             ValueRange indices,
                             ConversionPatternRewriter &rewriter) {
  auto memrefType = memrefValue.getType().cast<ShapedType>();
  if (memrefType.getRank() == 0) {
    // Rank 0 buffers (like memref<i32>) have only a single valid offset at 0.
    return rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
  }
  assert(memrefType.getRank() == 1 && "memrefs should have been flattened");

  // Element type byte length as the base. Note that this is the unconverted
  // element type. Since these are storage types within a buffer, they are
  // not subject to general type conversion (i.e. a general type converter
  // may elect to represent all i8 registers as i32, but this does not mean
  // that all memrefs are widened from i8 to i32).
  auto elementType = memrefType.getElementType();
  auto elementSize =
      rewriter.createOrFold<IREE::Util::SizeOfOp>(loc, elementType);

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
        globalOp.getLoc(), /*name=*/nullptr, globalOp.getInitialValueAttr(),
        alignmentAttr, /*mimeType=*/nullptr);
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
    auto numElements = rewriter.create<arith::ConstantIndexOp>(
        allocaOp.getLoc(), type.getNumElements());
    auto elementSize = rewriter.createOrFold<IREE::Util::SizeOfOp>(
        allocaOp.getLoc(), type.getElementType());
    auto allocationSize = rewriter.createOrFold<arith::MulIOp>(
        allocaOp.getLoc(), numElements, elementSize);
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
    auto elementType =
        dimOp.getSource().getType().cast<MemRefType>().getElementType();
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
    auto memRefSize = rewriter.createOrFold<IREE::Util::BufferSizeOp>(
        loadOp.getLoc(), rewriter.getIndexType(), adaptor.getMemref());
    auto byteOffset = getBufferOffset(loadOp.getLoc(), loadOp.getMemref(),
                                      loadOp.getIndices(), rewriter);
    Value loaded = rewriter.create<IREE::Util::BufferLoadOp>(
        loadOp.getLoc(), oldType, adaptor.getMemref(), memRefSize, byteOffset);
    if (newType != oldType) {
      // Since the BufferLoadOp semantics include its result type (i.e. a load
      // of an i8 is different than a load of an i32), in the presence of type
      // conversion, we must preserve the original type and emit an unrealized
      // conversion cast for downstreams. In this case, further legalizations
      // will be required to resolve it. This comes up in A->B->C lowerings
      // where the BufferLoad is an intermediate stage.
      loaded = rewriter
                   .create<UnrealizedConversionCastOp>(loadOp.getLoc(), newType,
                                                       loaded)
                   .getResult(0);
    }
    rewriter.replaceOp(loadOp, loaded);
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
    auto memRefSize = rewriter.createOrFold<IREE::Util::BufferSizeOp>(
        storeOp.getLoc(), rewriter.getIndexType(), adaptor.getMemref());
    auto byteOffset = getBufferOffset(storeOp.getLoc(), storeOp.getMemref(),
                                      storeOp.getIndices(), rewriter);
    Value newValue = adaptor.getValue();
    if (newValue.getType() != storeOp.getValue().getType()) {
      // In combination with type conversion, the elemental type may change,
      // and this is load bearing with respect to buffer_store op semantics
      // (i.e. storing of an i32 is different from an i8, even if the
      // conversion target widens). Insert an unrealized conversion cast to
      // preserve the original semantic. Presumably, something will clear this
      // with additional lowering.
      newValue =
          rewriter
              .create<UnrealizedConversionCastOp>(
                  storeOp.getLoc(), storeOp.getValue().getType(), newValue)
              .getResult(0);
    }
    rewriter.replaceOpWithNewOp<IREE::Util::BufferStoreOp>(
        storeOp, newValue, adaptor.getMemref(), memRefSize, byteOffset);
    return success();
  }
};

}  // namespace

void populateMemRefToUtilPatterns(MLIRContext *context,
                                  ConversionTarget &conversionTarget,
                                  TypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  Type convertedBufferType) {
  conversionTarget.addIllegalDialect<memref::MemRefDialect>();

  typeConverter.addConversion(
      [convertedBufferType](MemRefType type) -> llvm::Optional<Type> {
        if (isRankZeroOrOneMemRef(type)) {
          if (convertedBufferType) {
            return convertedBufferType;
          } else {
            return IREE::Util::BufferType::get(type.getContext());
          }
        }
        return llvm::None;
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
