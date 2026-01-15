// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/HALToStream/Patterns.h"

#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/Conversion/PatternUtils.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::iree_compiler {

namespace {

/// Flatten the given value ranges into a single vector of values.
static SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (const auto &vals : values) {
    llvm::append_range(result, vals);
  }
  return result;
}

// %1 = hal.tensor.import %0 : !hal.buffer_view -> tensor<4xf32>
// ->
// %1 = stream.tensor.import %0 : !hal.buffer_view ->
//                                tensor<4xf32> in !stream.resource<*>
struct ConvertTensorImportOp
    : public AffinityOpConversionPattern<IREE::HAL::TensorImportOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::HAL::TensorImportOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceType = op.getSource().getType();
    auto targetType = op.getTargetEncoding();
    if (!isa<IREE::HAL::BufferType>(sourceType) &&
        !isa<IREE::HAL::BufferViewType>(sourceType)) {
      return rewriter.notifyMatchFailure(op, "unsupported HAL cast conversion");
    }

    // Assert the shape of the buffer view matches the expected encoding
    // shape. We can only do this when we are importing a buffer view as that's
    // what carries the information we need to validate.
    if (isa<IREE::HAL::BufferViewType>(sourceType)) {
      // NOTE: we do this before the other checks as it's the most likely
      // mistake and it's better to know of a shape mismatch than just buffer
      // byte length difference.
      if (auto tensorType = dyn_cast<RankedTensorType>(targetType)) {
        if (failed(buildEncodingAssertions(
                op.getLoc(), adaptor.getSource().front(), op.getNameAttr(),
                tensorType, op.getTargetDims(), rewriter))) {
          return rewriter.notifyMatchFailure(op, "unsupported tensor type");
        }
      }
    }

    // Import (buffer view to stream resource).
    auto resultType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::External);
    Value resultSize = IREE::Stream::TensorSizeOfOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(),
        TypeAttr::get(op.getTarget().getType()),
        flattenValues(adaptor.getTargetDims()), executionAffinityAttr);
    Value resource = IREE::Stream::TensorImportOp::create(
        rewriter, op.getLoc(), resultType, adaptor.getSource().front(),
        targetType, flattenValues(adaptor.getTargetDims()), resultSize,
        op.getConsume(), executionAffinityAttr);

    // Await the fence, if needed. When not specified the resource is assumed to
    // be immediately available.
    if (auto waitFence = op.getWaitFence()) {
      Value waitTimepoint = IREE::Stream::TimepointImportOp::create(
          rewriter, op.getLoc(),
          rewriter.getType<IREE::Stream::TimepointType>(),
          ValueRange{waitFence}, executionAffinityAttr);
      resource = IREE::Stream::TimepointAwaitOp::create(
                     rewriter, op.getLoc(), ValueRange{resource},
                     ValueRange{resultSize}, waitTimepoint)
                     .getResult(0);
    }

    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    Value newImport = IREE::Stream::AsyncTransferOp::create(
        rewriter, op.getLoc(), unknownType, resource, resultSize, resultSize,
        /*source_affinity=*/executionAffinityAttr,
        /*target_affinity=*/executionAffinityAttr);
    rewriter.replaceOpWithMultiple(op, {{newImport, resultSize}});
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
    // If the encoding attr is about packed storage then we don't need
    // assertion, because packed storage attribute is about memory layout and it
    // doesn't affect the tensor shape.
    if (IREE::Encoding::hasPackedStorageAttr(tensorType)) {
      return success();
    }

    auto expectedElementType = IREE::HAL::ElementTypeOp::create(
        builder, loc, tensorType.getElementType());
    auto expectedEncodingType = IREE::HAL::EncodingTypeOp::create(
        builder, loc, tensorType.getEncoding());

    SmallVector<Value> shapeDims;
    if (tensorType.getRank() > 0) {
      unsigned dynamicIdx = 0;
      for (int64_t idx = 0; idx < tensorType.getRank(); ++idx) {
        Value expectedDim;
        if (tensorType.isDynamicDim(idx)) {
          expectedDim = dynamicDims[dynamicIdx++];
        } else {
          expectedDim = arith::ConstantIndexOp::create(
              builder, loc, tensorType.getDimSize(idx));
        }
        shapeDims.push_back(expectedDim);
      }
    }

    IREE::HAL::BufferViewAssertOp::create(
        builder, loc, bufferView,
        message ? message : builder.getStringAttr("tensor"),
        expectedElementType, expectedEncodingType, shapeDims);
    return success();
  }
};

// %1 = hal.tensor.export %0 : tensor<4xf32> -> !hal.buffer_view
// ->
// %1 = stream.tensor.export %0 : tensor<4xf32> in !stream.resource<*> ->
//                                !hal.buffer_view
struct ConvertTensorExportOp
    : public AffinityOpConversionPattern<IREE::HAL::TensorExportOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::HAL::TensorExportOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceType = op.getSourceEncoding();
    auto targetType = op.getTarget().getType();
    if (!isa<IREE::HAL::BufferType>(targetType) &&
        !isa<IREE::HAL::BufferViewType>(targetType)) {
      return rewriter.notifyMatchFailure(op, "unsupported HAL cast conversion");
    }

    auto source =
        transferTensorOperands(op.getLoc(), op.getSource(), adaptor.getSource(),
                               executionAffinityAttr, rewriter);

    // Exporting a produced value - transfer our source value to an externally
    // usable resource and directly export it. This will cause an allocation.
    Value exportSource = adaptor.getSource().front();
    auto externalType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::External);
    if (source.resource.getType() != externalType) {
      exportSource = IREE::Stream::AsyncTransferOp::create(
          rewriter, op.getLoc(), externalType, source.resource,
          source.resourceSize, source.resourceSize,
          /*source_affinity=*/source.affinity,
          /*target_affinity=*/executionAffinityAttr);
    }

    // Export (stream resource to buffer view).
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorExportOp>(
        op, targetType, exportSource, TypeAttr::get(sourceType),
        flattenValues(adaptor.getSourceDims()), source.resourceSize,
        executionAffinityAttr);
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
    : public AffinityOpConversionPattern<IREE::HAL::TensorAliasOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::HAL::TensorAliasOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceType = op.getSource().getType();
    auto source =
        transferTensorOperands(op.getLoc(), op.getSource(), adaptor.getSource(),
                               executionAffinityAttr, rewriter);

    // Query the target storage buffer length; we will only populate up to
    // what is required for the output.
    SmallVector<Value> convertedSourceDims =
        flattenValues(adaptor.getSourceDims());
    Value storageSize = IREE::Stream::TensorSizeOfOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(),
        TypeAttr::get(op.getSource().getType()), convertedSourceDims,
        executionAffinityAttr);

    // Import the target storage as a resource that we can use as an update
    // target. We overwrite the contents and just cast the storage to the
    // target type so we know we can update it.
    auto externalType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::External);
    auto importOp = IREE::Stream::TensorImportOp::create(
        rewriter, op.getLoc(), externalType, adaptor.getStorage().front(),
        TypeAttr::get(sourceType), convertedSourceDims, storageSize,
        /*consume=*/UnitAttr{}, executionAffinityAttr);

    // Await the fence, if needed. When not specified the storage is assumed to
    // be immediately available.
    Value storage = importOp.getResult();
    if (auto waitFence = op.getWaitFence()) {
      Value waitTimepoint = IREE::Stream::TimepointImportOp::create(
          rewriter, op.getLoc(),
          rewriter.getType<IREE::Stream::TimepointType>(),
          ValueRange{waitFence}, executionAffinityAttr);
      storage = IREE::Stream::TimepointAwaitOp::create(
                    rewriter, op.getLoc(), ValueRange{storage},
                    ValueRange{storageSize}, waitTimepoint)
                    .getResult(0);
    }

    // Copy the source value into the imported target storage.
    auto zeroOffset = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    auto updateOp = IREE::Stream::AsyncUpdateOp::create(
        rewriter, op.getLoc(), externalType, storage, storageSize, zeroOffset,
        source.resourceSize, source.resource, source.resourceSize,
        executionAffinityAttr);

    // Slice out the value from the updated tensor.
    // This preserves the use-def chain but is almost always elided by aliasing
    // the input value later on.
    auto sliceOp = IREE::Stream::AsyncSliceOp::create(
        rewriter, op.getLoc(), externalType, updateOp.getResult(),
        updateOp.getTargetSize(), zeroOffset, source.resourceSize,
        source.resourceSize, executionAffinityAttr);

    // Transfer to match original lifetime (if needed).
    Value result = sliceOp.getResult();
    if (source.resource.getType() != result.getType()) {
      result = IREE::Stream::AsyncTransferOp::create(
          rewriter, op.getLoc(), source.resource.getType(), result,
          source.resourceSize, source.resourceSize, executionAffinityAttr,
          executionAffinityAttr);
    }
    rewriter.replaceOpWithMultiple(op, {{result, source.resourceSize}});

    return success();
  }
};

// %1 = hal.tensor.transients %0 : tensor<4xf32>
//          from %storage : !hal.buffer (or !hal.buffer_view)
// ->
// %buffer = hal.buffer_view.buffer %storage (if buffer_view)
// %storage_size = hal.buffer.length %buffer
// %imported = stream.tensor.import %buffer : !hal.buffer -> tensor<?xi8>
//                 in !stream.resource<transient>{%storage_size}
// %0a, %t0 = stream.timepoint.barrier %0 : !stream.resource<*>{%size}
//                                      => !stream.timepoint
// %1a, %t1 = stream.resource.transients await(%t0) => %0a
//                                       : !stream.resource<*>{%size}
//                                       from %imported
//                                       :
//                                       !stream.resource<transient>{%storage_size}
//                                       => !stream.timepoint
// %1 = stream.timepoint.await %t1 => %1a : !stream.resource<*>{%size}
struct ConvertTensorTransientsOp
    : public AffinityOpConversionPattern<IREE::HAL::TensorTransientsOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::HAL::TensorTransientsOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto source =
        transferTensorOperands(op.getLoc(), op.getSource(), adaptor.getSource(),
                               executionAffinityAttr, rewriter);

    // Insert barrier to materialize timepoint from resource.
    auto timepointType = rewriter.getType<IREE::Stream::TimepointType>();
    auto barrierOp = IREE::Stream::TimepointBarrierOp::create(
        rewriter, op.getLoc(), source.resource.getType(), timepointType,
        source.resource, source.resourceSize, executionAffinityAttr);

    // Get storage (can be hal.buffer or hal.buffer_view) and extract buffer.
    Value storage = adaptor.getStorage().front();
    Value halBuffer = storage;
    if (llvm::isa<IREE::HAL::BufferViewType>(storage.getType())) {
      // If storage is a buffer_view, extract the underlying buffer.
      halBuffer = rewriter.createOrFold<IREE::HAL::BufferViewBufferOp>(
          op.getLoc(), rewriter.getType<IREE::HAL::BufferType>(), storage);
    }

    // Get buffer length.
    Value bufferLength = rewriter.createOrFold<IREE::HAL::BufferLengthOp>(
        op.getLoc(), rewriter.getIndexType(), halBuffer);

    // Import storage as stream.resource<transient>.
    // We use an arbitrary tensor type for the import - the actual type doesn't
    // matter as this is just raw storage for transient allocations.
    auto transientType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::Transient);
    auto tensorType =
        RankedTensorType::get({ShapedType::kDynamic}, rewriter.getI8Type());
    auto importOp = IREE::Stream::TensorImportOp::create(
        rewriter, op.getLoc(), transientType, halBuffer,
        TypeAttr::get(tensorType), ValueRange{bufferLength}, bufferLength,
        /*consume=*/UnitAttr{}, executionAffinityAttr);

    // Convert to stream.resource.transients with imported storage + size.
    auto transientsOp = IREE::Stream::ResourceTransientsOp::create(
        rewriter, op.getLoc(), source.resource.getType(), timepointType,
        barrierOp.getResult(), source.resourceSize, importOp.getResult(),
        bufferLength, barrierOp.getResultTimepoint(), executionAffinityAttr);

    // Await the result timepoint to resolve to plain resource.
    auto awaitOp = IREE::Stream::TimepointAwaitOp::create(
        rewriter, op.getLoc(), {transientsOp.getResult()},
        {source.resourceSize}, transientsOp.getResultTimepoint());
    rewriter.replaceOpWithMultiple(op, {{
                                           awaitOp.getResults().front(),
                                           source.resourceSize,
                                       }});
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
    : public AffinityAwareConversionPattern<IREE::HAL::TensorBarrierOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::TensorBarrierOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto timepointType = rewriter.getType<IREE::Stream::TimepointType>();
    IREE::Stream::AffinityAttr anyAffinityAttr;
    SmallVector<Value> signaledResources;
    SmallVector<Value> signaledResourceSizes;
    SmallVector<Value> signaledTimepoints;
    for (auto [sourceTensor, sourceResource] :
         llvm::zip_equal(op.getSources(), adaptor.getSources())) {
      auto source = resolveTensorOperands(op.getLoc(), sourceTensor,
                                          sourceResource, rewriter);
      auto barrierOp = IREE::Stream::TimepointBarrierOp::create(
          rewriter, sourceResource.front().getLoc(), source.resource.getType(),
          timepointType, source.resource, source.resourceSize, source.affinity);
      signaledResources.push_back(barrierOp.getResult());
      signaledResourceSizes.push_back(source.resourceSize);
      signaledTimepoints.push_back(barrierOp.getResultTimepoint());

      // When joining from multiple affinities we need to pick one to perform
      // the chain. For now we do the affinity of the last tensor with the hope
      // that we can perform the final signal on the affinity that is running.
      // We should instead probably change this to be set after timepoint
      // propagation such that we ensure it happens on the final signal when not
      // acting as a join.
      anyAffinityAttr = source.affinity;
    }
    Value joinedTimepoint = IREE::Stream::TimepointJoinOp::join(
        op.getLoc(), signaledTimepoints, rewriter);
    IREE::Stream::TimepointChainExternalOp::create(
        rewriter, op.getLoc(), joinedTimepoint,
        ValueRange{adaptor.getSignalFence()}, anyAffinityAttr);
    replaceOpWithMultiple(op, signaledResources, signaledResourceSizes,
                          rewriter);
    return success();
  }
};

} // namespace

void populateHALToStreamConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    IREE::Stream::AffinityAnalysis *affinityAnalysis,
    RewritePatternSet &patterns) {
  typeConverter.addConversion(
      [](IREE::HAL::BufferViewType type) { return type; });
  patterns.insert<ConvertTensorImportOp>(typeConverter, context,
                                         affinityAnalysis);
  patterns.insert<ConvertTensorExportOp>(typeConverter, context,
                                         affinityAnalysis);
  patterns.insert<ConvertTensorAliasOp>(typeConverter, context,
                                        affinityAnalysis);
  patterns.insert<ConvertTensorTransientsOp>(typeConverter, context,
                                             affinityAnalysis);
  patterns.insert<ConvertTensorBarrierOp>(typeConverter, context,
                                          affinityAnalysis);
}

void populateHALToStreamConversionPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter,
    IREE::Stream::AffinityAnalysis *affinityAnalysis,
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

  populateHALToStreamConversionPatterns(context, typeConverter,
                                        affinityAnalysis, patterns);
}

} // namespace mlir::iree_compiler
