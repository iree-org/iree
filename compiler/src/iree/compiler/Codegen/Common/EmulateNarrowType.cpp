// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_EMULATENARROWTYPEPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

struct ConvertHalInterfaceBindingSubspan final
    : OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto currentType = dyn_cast<MemRefType>(op.getType());
    if (!currentType) {
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "unhandled non-memref types");
    }
    auto newResultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(currentType));
    if (!newResultType) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to legalize memref type: {0}", op.getType()));
    }
    Location loc = op.getLoc();
    OpFoldResult zero = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> indices(currentType.getRank(), zero);

    // Get linearized type.
    int srcBits = currentType.getElementType().getIntOrFloatBitWidth();
    int dstBits = newResultType.getElementType().getIntOrFloatBitWidth();
    OpFoldResult elementOffset;
    Value byteOffset = adaptor.getByteOffset();
    if (byteOffset && !matchPattern(byteOffset, m_Zero())) {
      elementOffset = convertByteOffsetToElementOffset(
          rewriter, loc, byteOffset, currentType.getElementType());
    } else {
      elementOffset = rewriter.getIndexAttr(0);
    }
    SmallVector<OpFoldResult> sizes = getMixedValues(
        currentType.getShape(), adaptor.getDynamicDims(), rewriter);
    memref::LinearizedMemRefInfo linearizedMemRefInfo =
        memref::getLinearizedMemRefOffsetAndSize(rewriter, loc, srcBits,
                                                 dstBits, elementOffset, sizes);

    SmallVector<Value> dynamicLinearizedSize;
    if (newResultType.getRank() > 0 && !newResultType.hasStaticShape()) {
      dynamicLinearizedSize.push_back(getValueOrCreateConstantIndexOp(
          rewriter, loc, linearizedMemRefInfo.linearizedSize));
    }

    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        op, newResultType, adaptor.getLayout(), adaptor.getBinding(),
        byteOffset, dynamicLinearizedSize, adaptor.getAlignmentAttr(),
        adaptor.getDescriptorFlagsAttr());
    return success();
  }
};

static void populateIreeNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &converter,
    RewritePatternSet &patterns) {
  patterns.add<ConvertHalInterfaceBindingSubspan>(converter,
                                                  patterns.getContext());
}

static bool isByteAligned(ShapedType type) {
  unsigned elementBits = type.getElementType().getIntOrFloatBitWidth();
  auto numElements = type.getNumElements();
  return (numElements * elementBits) % 8 == 0;
}

struct PadSubbyteTransferWritePattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const final {
    auto target = writeOp.getVector();
    auto targetType = cast<VectorType>(target.getType());
    if (isByteAligned(targetType)) {
      return failure();
    }

    auto source = writeOp.getSource();
    auto sourceType = cast<ShapedType>(source.getType());
    auto elemType = targetType.getElementType();
    unsigned elementBits = targetType.getElementType().getIntOrFloatBitWidth();
    auto numElements = targetType.getNumElements();

    SmallVector<int64_t> strides;
    SmallVector<int64_t> offsets;
    for (unsigned i = 0; i < sourceType.getRank(); ++i) {
      strides.push_back(1);
      offsets.push_back(0);
    }

    // TODO: we should keep the source and sink ... otherwise we are
    // overwriting some part of the source tensor

    SmallVector<int64_t> newShape = SmallVector<int64_t>(targetType.getShape());
    newShape.back() += (8 - (numElements * elementBits) % 8) / elementBits;
    auto newTargetType = VectorType::get(newShape, elemType);

    // create an empty vector of the correct size
    SmallVector<bool> zeroValues;
    for (unsigned i = 0; i < newTargetType.getNumElements(); ++i) {
      zeroValues.push_back(false);
    }
    auto zeroVector = rewriter.create<arith::ConstantOp>(
        writeOp.getLoc(), DenseIntElementsAttr::get(newTargetType, zeroValues));

    auto extendedOp = rewriter.create<vector::InsertStridedSliceOp>(
        writeOp->getLoc(), target, zeroVector, offsets, strides);

    writeOp.getVectorMutable().assign(extendedOp);
    return success();
  }
};

struct PadSubbyteTransferReadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const final {
    auto resultType = cast<VectorType>(readOp.getResult().getType());
    if (isByteAligned(resultType)) {
      return failure();
    }

    unsigned elementBits = resultType.getElementType().getIntOrFloatBitWidth();
    auto numElements = resultType.getNumElements();

    // pad the type to be byte aligned
    SmallVector<int64_t> newShape = SmallVector<int64_t>(resultType.getShape());
    newShape.back() += (8 - (numElements * elementBits) % 8) / elementBits;
    // Create a new vector type with the padded shape
    auto newType = VectorType::get(newShape, resultType.getElementType());

    // Create a new transfer read op with the new type
    auto paddingValue = rewriter.create<arith::ConstantOp>(
        readOp.getLoc(), resultType.getElementType(),
        rewriter.getZeroAttr(resultType.getElementType()));

    // use a vector extract to extract the original vector
    SmallVector<int64_t> offsets, strides;
    for (unsigned i = 0; i < resultType.getRank(); ++i) {
      offsets.push_back(0);
      strides.push_back(1);
    }

    auto newTransferReadOp = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), newType, readOp.getSource(), readOp.getIndices(),
        paddingValue);

    rewriter.replaceOpWithNewOp<vector::ExtractStridedSliceOp>(
        readOp, newTransferReadOp, offsets, resultType.getShape(), strides);
    return success();
  }
};

struct PadSubbyteVectorLoadPattern : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const final {
    auto result = loadOp.getResult();
    auto resultType = mlir::cast<VectorType>(result.getType());
    if (isByteAligned(resultType)) {
      return failure();
    }

    unsigned elementBits = resultType.getElementType().getIntOrFloatBitWidth();
    auto numElements = resultType.getNumElements();

    SmallVector<int64_t> newShape = SmallVector<int64_t>(resultType.getShape());
    newShape.back() += (8 - (numElements * elementBits) % 8) / elementBits;
    auto newTargetType = VectorType::get(newShape, resultType.getElementType());

    // create a new vector load op with the new type
    auto newVectorLoad = rewriter.create<vector::LoadOp>(
        loadOp.getLoc(), newTargetType, loadOp.getBase(), loadOp.getIndices());

    auto newNumElements = newTargetType.getNumElements();
    SmallVector<bool> zeroValues;
    for (unsigned i = 0; i < newNumElements; ++i) {
      zeroValues.push_back(false);
    }

    // extract strided slice
    SmallVector<int64_t> offsets, strides;
    for (unsigned i = 0; i < resultType.getRank(); ++i) {
      offsets.push_back(0);
      strides.push_back(1);
    }

    rewriter.replaceOpWithNewOp<vector::ExtractStridedSliceOp>(
        loadOp, newVectorLoad, offsets, resultType.getShape(), strides);
    return success();
  }
};

struct PadSubbyteVectorStorePattern : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::StoreOp storeOp,
                                PatternRewriter &rewriter) const final {
    auto storeValue = storeOp.getValueToStore();
    auto valueType = mlir::cast<ShapedType>(storeValue.getType());
    if (isByteAligned(valueType)) {
      return failure();
    }

    auto target = storeOp.getBase();
    auto targetType = mlir::cast<ShapedType>(target.getType());
    // check that the type size is byte aligned
    auto elemType = valueType.getElementType();
    unsigned elementBits = valueType.getElementType().getIntOrFloatBitWidth();
    auto numElements = valueType.getNumElements();

    SmallVector<int64_t> newShape = SmallVector<int64_t>(valueType.getShape());
    newShape.back() += (8 - (numElements * elementBits) % 8) / elementBits;
    auto newValueType = VectorType::get(newShape, elemType);

    SmallVector<int64_t> strides;
    SmallVector<int64_t> offsets;
    for (unsigned i = 0; i < targetType.getRank(); ++i) {
      strides.push_back(1);
      offsets.push_back(0);
    }

    // create an empty vector of the correct size
    SmallVector<bool> zeroValues;
    for (unsigned i = 0; i < newValueType.getNumElements(); ++i) {
      zeroValues.push_back(false);
    }
    auto zeroVector = rewriter.create<arith::ConstantOp>(
        storeOp.getLoc(), DenseIntElementsAttr::get(newValueType, zeroValues));

    auto extendedOp = rewriter.create<vector::InsertStridedSliceOp>(
        storeOp->getLoc(), storeValue, zeroVector, offsets, strides);

    // create a mask and use masked store:
    SmallVector<Value> maskShape;
    for (auto dim : valueType.getShape()) {
      maskShape.push_back(
          rewriter.create<arith::ConstantIndexOp>(storeOp.getLoc(), dim));
    }
    auto mask = rewriter.create<vector::CreateMaskOp>(storeOp.getLoc(),
                                                      newValueType, maskShape);

    rewriter.replaceOpWithNewOp<vector::MaskedStoreOp>(
        storeOp, target, storeOp.getIndices(), mask, extendedOp);
    return success();
  }
};

static void populateSubbyteTypeHandlingPatterns(RewritePatternSet &patterns) {
  patterns.add<PadSubbyteTransferReadPattern, PadSubbyteTransferWritePattern,
               PadSubbyteVectorLoadPattern, PadSubbyteVectorStorePattern>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct EmulateNarrowTypePass final
    : impl::EmulateNarrowTypePassBase<EmulateNarrowTypePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    affine::AffineDialect, IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    // The number of bits used in a load/store op.
    constexpr unsigned kLoadStoreEmulateBitwidth = 8;
    static_assert(
        llvm::isPowerOf2_32(kLoadStoreEmulateBitwidth) &&
        "only power of 2 is supported for narrow type load/store emulation");

    MLIRContext *ctx = &getContext();

    arith::NarrowTypeEmulationConverter typeConverter(
        kLoadStoreEmulateBitwidth);
    memref::populateMemRefNarrowTypeEmulationConversions(typeConverter);

    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](Operation *op) {
      return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
    });
    auto opLegalCallback = [&typeConverter](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(opLegalCallback);
    target.addDynamicallyLegalDialect<
        arith::ArithDialect, vector::VectorDialect, memref::MemRefDialect,
        affine::AffineDialect, IREE::HAL::HALDialect>(opLegalCallback);

    RewritePatternSet patterns(ctx);
    populateSubbyteTypeHandlingPatterns(patterns);
    arith::populateArithNarrowTypeEmulationPatterns(typeConverter, patterns);
    memref::populateMemRefNarrowTypeEmulationPatterns(typeConverter, patterns);
    populateIREEResolveExtractStridedMetadataPatterns(ctx, patterns);
    vector::populateVectorNarrowTypeEmulationPatterns(typeConverter, patterns);
    populateIreeNarrowTypeEmulationPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      getOperation()->emitOpError("failed to emulate bit width");
      return signalPassFailure();
    }

    RewritePatternSet sinkBroadcast(ctx);
    vector::populateSinkVectorOpsPatterns(sinkBroadcast);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(sinkBroadcast)))) {
      getOperation()->emitOpError("failed in sinking of broadcasts");
      return signalPassFailure();
    }

    // Also do the `bitcast -> extui/extsi` rewrite.
    RewritePatternSet foldExtPatterns(ctx);
    vector::populateVectorNarrowTypeRewritePatterns(foldExtPatterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(foldExtPatterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
