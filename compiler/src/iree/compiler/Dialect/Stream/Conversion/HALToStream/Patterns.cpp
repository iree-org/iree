// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/HALToStream/Patterns.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/Conversion/PatternUtils.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::iree_compiler {

namespace {

// %1 = hal.tensor.import %0 : !hal.buffer_view -> tensor<4xf32>
// ->
// %1 = stream.tensor.import %0 : !hal.buffer_view ->
//                                tensor<4xf32> in !stream.resource<*>
struct ConvertTensorImportOp
    : public OpConversionPattern<IREE::HAL::TensorImportOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::TensorImportOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = op.getSource().getType();
    auto targetType = op.getTargetEncoding();
    if (!llvm::isa<IREE::HAL::BufferType>(sourceType) &&
        !llvm::isa<IREE::HAL::BufferViewType>(sourceType)) {
      return rewriter.notifyMatchFailure(op, "unsupported HAL cast conversion");
    }

    // Assert the shape of the buffer view matches the expected encoding
    // shape. We can only do this when we are importing a buffer view as that's
    // what carries the information we need to validate.
    if (llvm::isa<IREE::HAL::BufferViewType>(sourceType)) {
      // NOTE: we do this before the other checks as it's the most likely
      // mistake and it's better to know of a shape mismatch than just buffer
      // byte length difference.
      if (auto tensorType = llvm::dyn_cast<RankedTensorType>(targetType)) {
        if (failed(buildEncodingAssertions(op.getLoc(), adaptor.getSource(),
                                           op.getNameAttr(), tensorType,
                                           op.getTargetDims(), rewriter))) {
          return rewriter.notifyMatchFailure(op, "unsupported tensor type");
        }
      }
    }

    // Import (buffer view to stream resource).
    auto affinityAttr = IREE::Stream::AffinityAttr::lookup(op);
    auto resultType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::External);
    auto resultSize = rewriter.createOrFold<IREE::Stream::TensorSizeOfOp>(
        op.getLoc(), rewriter.getIndexType(),
        TypeAttr::get(op.getTarget().getType()), adaptor.getTargetDims(),
        affinityAttr);
    Value resource = rewriter.create<IREE::Stream::TensorImportOp>(
        op.getLoc(), resultType, adaptor.getSource(), TypeAttr::get(targetType),
        adaptor.getTargetDims(), resultSize, affinityAttr);

    // Await the fence, if needed. When not specified the resource is assumed to
    // be immediately available.
    if (auto waitFence = op.getWaitFence()) {
      Value waitTimepoint = rewriter.create<IREE::Stream::TimepointImportOp>(
          op.getLoc(), rewriter.getType<IREE::Stream::TimepointType>(),
          ValueRange{waitFence}, affinityAttr);
      resource = rewriter
                     .create<IREE::Stream::TimepointAwaitOp>(
                         op.getLoc(), ValueRange{resource},
                         ValueRange{resultSize}, waitTimepoint)
                     .getResult(0);
    }

    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncTransferOp>(
        op, unknownType, resource, resultSize, resultSize, affinityAttr,
        affinityAttr);
    return success();
  }

  // Inserts IR to assert that the buffer view shape and encoding matches the
  // expected encoding we have in the program. This ensures that the user didn't
  // pass a 4x8xf32 when they originally compiled the model for a 2x8x1xi8.
  static LogicalResult buildEncodingAssertions(Location loc, Value bufferView,
                                               StringAttr message,
                                               RankedTensorType tensorType,
                                               ValueRange dynamicDims,
                                               OpBuilder &builder) {
    auto expectedElementType = builder.create<IREE::HAL::ElementTypeOp>(
        loc, tensorType.getElementType());
    auto expectedEncodingType = builder.create<IREE::HAL::EncodingTypeOp>(
        loc, tensorType.getEncoding());

    SmallVector<Value> shapeDims;
    if (tensorType.getRank() > 0) {
      unsigned dynamicIdx = 0;
      for (int64_t idx = 0; idx < tensorType.getRank(); ++idx) {
        Value expectedDim;
        if (tensorType.isDynamicDim(idx)) {
          expectedDim = dynamicDims[dynamicIdx++];
        } else {
          expectedDim = builder.create<arith::ConstantIndexOp>(
              loc, tensorType.getDimSize(idx));
        }
        shapeDims.push_back(expectedDim);
      }
    }

    builder.create<IREE::HAL::BufferViewAssertOp>(
        loc, bufferView, message ? message : builder.getStringAttr("tensor"),
        expectedElementType, expectedEncodingType, shapeDims);
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
  LogicalResult
  matchAndRewrite(IREE::HAL::TensorExportOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = op.getSourceEncoding();
    auto targetType = op.getTarget().getType();
    if (!llvm::isa<IREE::HAL::BufferType>(targetType) &&
        !llvm::isa<IREE::HAL::BufferViewType>(targetType)) {
      return rewriter.notifyMatchFailure(op, "unsupported HAL cast conversion");
    }

    auto affinityAttr = IREE::Stream::AffinityAttr::lookup(op);
    auto source =
        consumeTensorOperand(op.getLoc(), adaptor.getSource(), rewriter);

    // Exporting a produced value - transfer our source value to an externally
    // usable resource and directly export it. This will cause an allocation.
    auto exportSource = adaptor.getSource();
    auto externalType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::External);
    if (source.resource.getType() != externalType) {
      exportSource = rewriter.create<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), externalType, source.resource, source.resourceSize,
          source.resourceSize, affinityAttr, affinityAttr);
    }

    // Export (stream resource to buffer view).
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorExportOp>(
        op, targetType, exportSource, TypeAttr::get(sourceType),
        adaptor.getSourceDims(), source.resourceSize, affinityAttr);
    return success();
  }
};

// Imports the storage to alias as a resource, copies the source value into it,
// and slices out the source value. This should allow allocation placement to
// elide the update (and subsequently the slice) if possible and otherwise will
// turn into a copy.
//
// Effectively:
//   %2 = hal.tensor.alias %0 : tensor<4xf32> to %1 : !hal.buffer_view
// ->
//   %storage = stream.tensor.import %1 : !hal.buffer -> tensor<...>
//   %update = stream.async.update %0, %storage[...]
//   %2 = stream.async.slice %update[...]
struct ConvertTensorAliasOp
    : public OpConversionPattern<IREE::HAL::TensorAliasOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::TensorAliasOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = op.getSource().getType();
    auto source =
        consumeTensorOperand(op.getLoc(), adaptor.getSource(), rewriter);

    // All operations (if any) will happen on the device specified by the alias
    // as that indicates the affinity of the storage.
    auto affinityAttr = IREE::Stream::AffinityAttr::lookup(op);

    // Query the target storage buffer length; we will only populate up to
    // what is required for the output.
    auto storageSize = rewriter.createOrFold<IREE::Stream::TensorSizeOfOp>(
        op.getLoc(), rewriter.getIndexType(),
        TypeAttr::get(op.getSource().getType()), adaptor.getSourceDims(),
        affinityAttr);

    // Import the target storage as a resource that we can use as an update
    // target. We overwrite the contents and just cast the storage to the
    // target type so we know we can update it.
    auto externalType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::External);
    auto importOp = rewriter.create<IREE::Stream::TensorImportOp>(
        op.getLoc(), externalType, adaptor.getStorage(),
        TypeAttr::get(sourceType), adaptor.getSourceDims(), storageSize,
        affinityAttr);

    // Await the fence, if needed. When not specified the storage is assumed to
    // be immediately available.
    Value storage = importOp.getResult();
    if (auto waitFence = op.getWaitFence()) {
      Value waitTimepoint = rewriter.create<IREE::Stream::TimepointImportOp>(
          op.getLoc(), rewriter.getType<IREE::Stream::TimepointType>(),
          ValueRange{waitFence}, affinityAttr);
      storage = rewriter
                    .create<IREE::Stream::TimepointAwaitOp>(
                        op.getLoc(), ValueRange{storage},
                        ValueRange{storageSize}, waitTimepoint)
                    .getResult(0);
    }

    // Copy the source value into the imported target storage.
    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto updateOp = rewriter.create<IREE::Stream::AsyncUpdateOp>(
        op.getLoc(), externalType, storage, storageSize, zeroOffset,
        source.resourceSize, source.resource, source.resourceSize,
        affinityAttr);

    // Slice out the value from the updated tensor.
    // This preserves the use-def chain but is almost always elided by aliasing
    // the input value later on.
    auto sliceOp = rewriter.create<IREE::Stream::AsyncSliceOp>(
        op.getLoc(), externalType, updateOp.getResult(),
        updateOp.getTargetSize(), zeroOffset, source.resourceSize,
        source.resourceSize, affinityAttr);

    // Transfer to match original lifetime (if needed).
    Value result = sliceOp.getResult();
    if (source.resource.getType() != result.getType()) {
      result = rewriter.create<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), source.resource.getType(), result, source.resourceSize,
          source.resourceSize, affinityAttr, affinityAttr);
    }
    rewriter.replaceOp(op, result);

    return success();
  }
};

// %r0b, %r1b = hal.tensor.barrier join(%r0a : tensor<4xf32>,
//                                      %r1a : tensor<1xi32>) => %fence
// ->
// %r0b, %t0 = stream.timepoint.barrier %r0a :
//                 tensor<4xf32> in !stream.resource<*> => !stream.timepoint
// %r1b, %t1 = stream.timepoint.barrier %r1a :
//                 tensor<1xi32> in !stream.resource<*> => !stream.timepoint
// %t01 = stream.timepoint.join max(%t0, %t1)
// stream.timepoint.export %t01 => %fence
struct ConvertTensorBarrierOp
    : public OpConversionPattern<IREE::HAL::TensorBarrierOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::TensorBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto affinityAttr = IREE::Stream::AffinityAttr::lookup(op);
    auto timepointType = rewriter.getType<IREE::Stream::TimepointType>();
    SmallVector<Value> signaledResources;
    SmallVector<Value> signaledTimepoints;
    for (auto sourceResource : adaptor.getSources()) {
      auto source = consumeTensorOperand(op.getLoc(), sourceResource, rewriter);
      auto barrierOp = rewriter.create<IREE::Stream::TimepointBarrierOp>(
          sourceResource.getLoc(), source.resource.getType(), timepointType,
          source.resource, source.resourceSize, affinityAttr);
      signaledResources.push_back(barrierOp.getResult());
      signaledTimepoints.push_back(barrierOp.getResultTimepoint());
    }
    Value joinedTimepoint = IREE::Stream::TimepointJoinOp::join(
        op.getLoc(), signaledTimepoints, rewriter);
    rewriter.create<IREE::Stream::TimepointChainExternalOp>(
        op.getLoc(), joinedTimepoint, ValueRange{adaptor.getSignalFence()},
        affinityAttr);
    rewriter.replaceOp(op, signaledResources);
    return success();
  }
};

} // namespace

void populateHALToStreamConversionPatterns(MLIRContext *context,
                                           TypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  typeConverter.addConversion(
      [](IREE::HAL::BufferViewType type) { return type; });
  patterns.insert<ConvertTensorImportOp>(typeConverter, context);
  patterns.insert<ConvertTensorExportOp>(typeConverter, context);
  patterns.insert<ConvertTensorAliasOp>(typeConverter, context);
  patterns.insert<ConvertTensorBarrierOp>(typeConverter, context);
}

void populateHALToStreamConversionPatterns(MLIRContext *context,
                                           ConversionTarget &conversionTarget,
                                           TypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  // Allow executables through without modification.
  conversionTarget.addLegalOp<IREE::HAL::ExecutableOp>();
  conversionTarget.markOpRecursivelyLegal<IREE::HAL::ExecutableOp>();

  conversionTarget.addDynamicallyLegalOp<IREE::HAL::TensorImportOp>(
      [&](IREE::HAL::TensorImportOp op) {
        return typeConverter.isLegal(op.getSource().getType()) &&
               typeConverter.isLegal(op.getTarget().getType());
      });
  conversionTarget.addDynamicallyLegalOp<IREE::HAL::TensorExportOp>(
      [&](IREE::HAL::TensorExportOp op) {
        return typeConverter.isLegal(op.getSource().getType()) &&
               typeConverter.isLegal(op.getTarget().getType());
      });

  populateHALToStreamConversionPatterns(context, typeConverter, patterns);
}

} // namespace mlir::iree_compiler
