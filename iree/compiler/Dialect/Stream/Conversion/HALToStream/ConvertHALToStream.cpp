// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/HALToStream/ConvertHALToStream.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/Conversion/PatternUtils.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
namespace iree_compiler {

namespace {

// %1 = hal.tensor.import %0 : !hal.buffer_view -> tensor<4xf32>
// ->
// %1 = stream.tensor.import %0 : !hal.buffer_view ->
//                                tensor<4xf32> in !stream.resource<*>
struct ConvertTensorImportOp
    : public OpConversionPattern<IREE::HAL::TensorImportOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::TensorImportOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceType = op.source().getType();
    auto targetType = op.target().getType();
    if (!sourceType.isa<IREE::HAL::BufferType>() &&
        !sourceType.isa<IREE::HAL::BufferViewType>()) {
      return rewriter.notifyMatchFailure(op, "unsupported HAL cast conversion");
    }

    // Import (buffer view to stream resource).
    auto resultType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::External);
    auto resultSize = rewriter.createOrFold<IREE::Stream::TensorSizeOfOp>(
        op.getLoc(), rewriter.getIndexType(),
        TypeAttr::get(op.target().getType()), adaptor.target_dims(),
        /*affinity=*/nullptr);
    auto newOp = rewriter.create<IREE::Stream::TensorImportOp>(
        op.getLoc(), resultType, adaptor.source(), TypeAttr::get(targetType),
        adaptor.target_dims(), resultSize,
        /*affinity=*/nullptr);

    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncTransferOp>(
        op, unknownType, newOp.result(), resultSize, resultSize,
        /*source_affinity=*/nullptr,
        /*result_affinity=*/nullptr);
    return success();
  }
};

// %1 = hal.tensor.export %0 : tensor<4xf32> -> !hal.buffer_view
// ->
// %1 = stream.tensor.export %0 : tensor<4xf32> in !stream.resource<*> ->
//                                !hal.buffer_view
struct ConvertTensorExportOp
    : public OpConversionPattern<IREE::HAL::TensorExportOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::TensorExportOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceType = op.source().getType();
    auto targetType = op.target().getType();
    if (!targetType.isa<IREE::HAL::BufferType>() &&
        !targetType.isa<IREE::HAL::BufferViewType>()) {
      return rewriter.notifyMatchFailure(op, "unsupported HAL cast conversion");
    }

    auto source = consumeTensorOperand(op.getLoc(), adaptor.source(), rewriter);
    auto externalType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::External);
    auto exportSource = adaptor.source();
    if (source.resource.getType() != externalType) {
      exportSource = rewriter.create<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), externalType, source.resource, source.resourceSize,
          source.resourceSize,
          /*source_affinity=*/nullptr,
          /*result_affinity=*/nullptr);
    }

    // Export (stream resource to buffer view).
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorExportOp>(
        op, targetType, exportSource, TypeAttr::get(sourceType),
        adaptor.source_dims(), source.resourceSize,
        /*affinity=*/nullptr);
    return success();
  }
};

}  // namespace

void populateHALToStreamConversionPatterns(MLIRContext *context,
                                           TypeConverter &typeConverter,
                                           OwningRewritePatternList &patterns) {
  typeConverter.addConversion(
      [](IREE::HAL::BufferViewType type) { return type; });
  patterns.insert<ConvertTensorImportOp>(typeConverter, context);
  patterns.insert<ConvertTensorExportOp>(typeConverter, context);
}

void populateHALToStreamConversionPatterns(MLIRContext *context,
                                           ConversionTarget &conversionTarget,
                                           TypeConverter &typeConverter,
                                           OwningRewritePatternList &patterns) {
  conversionTarget.addDynamicallyLegalOp<IREE::HAL::TensorImportOp>(
      [&](IREE::HAL::TensorImportOp op) {
        return typeConverter.isLegal(op.source().getType()) &&
               typeConverter.isLegal(op.target().getType());
      });
  conversionTarget.addDynamicallyLegalOp<IREE::HAL::TensorExportOp>(
      [&](IREE::HAL::TensorExportOp op) {
        return typeConverter.isLegal(op.source().getType()) &&
               typeConverter.isLegal(op.target().getType());
      });

  populateHALToStreamConversionPatterns(context, typeConverter, patterns);
}

}  // namespace iree_compiler
}  // namespace mlir
