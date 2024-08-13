// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/StreamToHAL/Patterns.h"

#include "iree/compiler/Dialect/HAL/Conversion/StreamToHAL/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

static llvm::cl::opt<bool> clIndirectCommandBuffers{
    "iree-hal-indirect-command-buffers",
    llvm::cl::desc("Whether to turn buffer bindings into indirect references "
                   "when recording command buffers."),
    llvm::cl::init(false),
};

// TODO(#18154): switch default to true and then remove.
static llvm::cl::opt<bool> clExperimentalDispatch2{
    "iree-hal-experimental-dispatch2",
    llvm::cl::desc("Whether to emit iree_hal_command_buffer_dispatch2 ops."),
    llvm::cl::init(false),
};

struct ContextResolveOpPattern
    : public StreamConversionPattern<IREE::Stream::ContextResolveOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ContextResolveOp resolveOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTypes = llvm::to_vector(resolveOp.getResultTypes());
    assert(!resultTypes.empty() && "must have at least one result");

    // Get the affinity from the op or an ancestor. Note that there may be no
    // affinity specified at all.
    auto affinityAttr = IREE::Stream::AffinityAttr::lookupOrDefault(resolveOp);

    // If no affinity was specified then resolve as 'any'.
    if (!affinityAttr) {
      rewriter.replaceOpWithNewOp<IREE::HAL::DeviceResolveOp>(
          resolveOp, resolveOp.getResultTypes(),
          IREE::HAL::DeviceAffinityAttr{});
      return success();
    }

    // We currently only handle HAL device affinities.
    // We could make this an interface to select the device and allow users to
    // provide their own affinities to convert to HAL. In the future users may
    // also want to provide devices as function arguments post-initialization.
    // For now we just have one way to specify device globals.
    if (auto deviceAffinityAttr =
            dyn_cast_if_present<IREE::HAL::DeviceAffinityAttr>(affinityAttr)) {
      rewriter.replaceOpWithNewOp<IREE::HAL::DeviceResolveOp>(
          resolveOp, resolveOp.getResultTypes(), deviceAffinityAttr);
      return success();
    }

    resolveOp.emitOpError() << "failed to resolve affinity: only HAL device "
                               "affinities are supported";
    return rewriter.notifyMatchFailure(
        resolveOp, "only HAL device affinities are supported");
  }
};

struct ResourceAllocOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceAllocOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceAllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [allocator, queueAffinity] =
        lookupAllocatorAndQueueAffinityFor(allocOp, rewriter);
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();

    auto resourceType =
        cast<IREE::Stream::ResourceType>(allocOp.getResult().getType());

    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
    if (failed(deriveAllowedResourceBufferBits(allocOp.getLoc(), resourceType,
                                               memoryTypes, bufferUsage))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IREE::HAL::AllocatorAllocateOp>(
        allocOp, bufferType, allocator, queueAffinity, memoryTypes, bufferUsage,
        adaptor.getStorageSize());
    return success();
  }
};

struct ResourceAllocaOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceAllocaOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceAllocaOp allocaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = allocaOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(allocaOp, rewriter);
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();

    auto resourceType =
        cast<IREE::Stream::ResourceType>(allocaOp.getResult().getType());
    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
    if (failed(deriveAllowedResourceBufferBits(loc, resourceType, memoryTypes,
                                               bufferUsage))) {
      return failure();
    }

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, allocaOp.getResultTimepoint(), rewriter);

    // Queue allocation.
    auto pool = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    auto allocateOp = rewriter.create<IREE::HAL::DeviceQueueAllocaOp>(
        loc, bufferType, device, queueAffinity, waitFence, signalFence, pool,
        memoryTypes, bufferUsage, adaptor.getStorageSize());

    rewriter.replaceOp(allocaOp, {allocateOp.getResult(), signalFence});
    return success();
  }
};

struct ResourceDeallocaOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceDeallocaOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceDeallocaOp deallocaOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = deallocaOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(deallocaOp, rewriter);

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, deallocaOp.getResultTimepoint(), rewriter);

    // Queue allocation.
    rewriter.create<IREE::HAL::DeviceQueueDeallocaOp>(
        loc, device, queueAffinity, waitFence, signalFence,
        adaptor.getOperand());

    rewriter.replaceOp(deallocaOp, {signalFence});
    return success();
  }
};

struct ResourceSizeOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceSizeOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceSizeOp sizeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferLengthOp>(
        sizeOp, rewriter.getIndexType(), adaptor.getOperand());
    return success();
  }
};

struct ResourceTryMapOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceTryMapOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceTryMapOp tryMapOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [allocator, queueAffinity] =
        lookupAllocatorAndQueueAffinityFor(tryMapOp, rewriter);
    auto resourceType =
        llvm::cast<IREE::Stream::ResourceType>(tryMapOp.getResult().getType());
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();

    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
    switch (resourceType.getLifetime()) {
    default:
      return tryMapOp.emitOpError()
             << "unsupported resource lifetime: "
             << IREE::Stream::stringifyLifetime(resourceType.getLifetime());
    case IREE::Stream::Lifetime::Constant:
      // Device local; copies required to get into external resources.
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
      bufferUsage =
          bufferUsage | IREE::HAL::BufferUsageBitfield::SharingImmutable;
      // TODO(benvanik): refine usage based on analysis.
      bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                    IREE::HAL::BufferUsageBitfield::DispatchStorage;
      break;
    case IREE::Stream::Lifetime::Variable:
      // Device local; copies required to get into external resources.
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
      // TODO(benvanik): refine usage based on analysis.
      bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                    IREE::HAL::BufferUsageBitfield::DispatchStorage;
      break;
    case IREE::Stream::Lifetime::Staging:
      // Host local; copies required to get into device resources.
      // We could vary this based on staging usage (upload/download) by
      // making it device-local|host-visible, but host-local means we have
      // a better chance of mapping it during uploads.
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::HostVisible |
                    IREE::HAL::MemoryTypeBitfield::DeviceVisible;
      bufferUsage =
          bufferUsage | IREE::HAL::BufferUsageBitfield::TransferSource;
      break;
    }

    rewriter.replaceOpWithNewOp<IREE::HAL::AllocatorImportOp>(
        tryMapOp, rewriter.getI1Type(), bufferType, allocator, queueAffinity,
        memoryTypes, bufferUsage, adaptor.getSource(),
        adaptor.getSourceOffset(), adaptor.getResultSize());
    return success();
  }
};

struct ResourceLoadOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceLoadOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loadType =
        getTypeConverter()->convertType(loadOp.getResult().getType());
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferLoadOp>(
        loadOp, loadType, adaptor.getSource(), adaptor.getSourceOffset());
    return success();
  }
};

struct ResourceStoreOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceStoreOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferStoreOp>(
        storeOp, adaptor.getValue(), adaptor.getTarget(),
        adaptor.getTargetOffset());
    return success();
  }
};

struct ResourceSubviewOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceSubviewOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceSubviewOp subviewOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();
    // NOTE: this aliases! We assume at this point all useful alias analysis
    // has been performed and it's fine to lose the tie information here.
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferSubspanOp>(
        subviewOp, bufferType, adaptor.getSource(), adaptor.getSourceOffset(),
        adaptor.getResultSize());
    return success();
  }
};

struct FileConstantOpPattern
    : public StreamConversionPattern<IREE::Stream::FileConstantOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::FileConstantOp constantOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(constantOp, rewriter);
    rewriter.replaceOpWithNewOp<IREE::HAL::ExFileFromMemoryOp>(
        constantOp, rewriter.getType<IREE::HAL::FileType>(), device,
        queueAffinity, IREE::HAL::MemoryAccessBitfield::Read,
        constantOp.getSource(), constantOp.getSourceOffset(),
        constantOp.getSourceLength(),
        rewriter.create<arith::ConstantIntOp>(constantOp.getLoc(), 0, 32));
    return success();
  }
};

struct FileReadOpPattern
    : public StreamConversionPattern<IREE::Stream::FileReadOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::FileReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = readOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(readOp, rewriter);

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, readOp.getResultTimepoint(), rewriter);

    // Queue read.
    rewriter.create<IREE::HAL::DeviceQueueReadOp>(
        loc, device, queueAffinity, waitFence, signalFence, adaptor.getSource(),
        adaptor.getSourceOffset(), adaptor.getTarget(),
        adaptor.getTargetOffset(), adaptor.getLength(),
        /*flags=*/0);

    rewriter.replaceOp(readOp, {signalFence});
    return success();
  }
};

struct FileWriteOpPattern
    : public StreamConversionPattern<IREE::Stream::FileWriteOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::FileWriteOp writeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = writeOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(writeOp, rewriter);

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, writeOp.getResultTimepoint(), rewriter);

    // Queue write.
    rewriter.create<IREE::HAL::DeviceQueueWriteOp>(
        loc, device, queueAffinity, waitFence, signalFence, adaptor.getSource(),
        adaptor.getSourceOffset(), adaptor.getTarget(),
        adaptor.getTargetOffset(), adaptor.getLength(),
        /*flags=*/0);

    rewriter.replaceOp(writeOp, {signalFence});
    return success();
  }
};

// Inserts IR to assert that the underlying buffer storage is compatible with
// the intended usage in the program. The allocator used to allocate the
// buffer must have compatibility with our target device allocator and the
// buffer must have at least the minimum expected size (additional padding is
// ok).
static LogicalResult
buildStorageAssertions(Location loc, Value buffer, StringAttr message,
                       Value allocator, Value minimumLength,
                       IREE::Stream::ResourceType resourceType,
                       OpBuilder &builder) {
  auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
  auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
  if (failed(deriveRequiredResourceBufferBits(loc, resourceType, memoryTypes,
                                              bufferUsage))) {
    return failure();
  }

  auto requiredTypes =
      IREE::HAL::MemoryTypeBitfieldAttr::get(builder.getContext(), memoryTypes);
  auto requiredUsage = IREE::HAL::BufferUsageBitfieldAttr::get(
      builder.getContext(), bufferUsage);

  builder.create<IREE::HAL::BufferAssertOp>(loc, buffer, message, allocator,
                                            minimumLength, requiredTypes,
                                            requiredUsage);
  return success();
}

struct TensorImportBufferOpPattern
    : public StreamConversionPattern<IREE::Stream::TensorImportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TensorImportOp importOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::isa<IREE::HAL::BufferType>(importOp.getSource().getType())) {
      return failure();
    }

    // TODO(benvanik): get a name for the tensor (argument name/etc).
    auto message = rewriter.getStringAttr("tensor");

    // Directly use the buffer.
    auto buffer = adaptor.getSource();
    rewriter.replaceOp(importOp, buffer);

    // Assert the storage is compatible with our expected device and usage.
    auto targetAllocator = lookupAllocatorFor(importOp, rewriter);
    auto resourceType =
        llvm::cast<IREE::Stream::ResourceType>(importOp.getResult().getType());
    if (failed(buildStorageAssertions(
            importOp.getLoc(), adaptor.getSource(), message, targetAllocator,
            adaptor.getResultSize(), resourceType, rewriter))) {
      return failure();
    }

    return success();
  }
};

struct TensorImportBufferViewOpPattern
    : public StreamConversionPattern<IREE::Stream::TensorImportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TensorImportOp importOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = importOp.getSource().getType();
    if (!llvm::isa<IREE::HAL::BufferViewType>(sourceType) &&
        !llvm::isa<TensorType>(sourceType)) {
      return failure();
    }

    auto loc = importOp.getLoc();

    // TODO(benvanik): get a name for the tensor (argument name/etc).
    auto message = rewriter.getStringAttr("tensor");

    auto bufferView = adaptor.getSource();
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();
    auto bufferOp = rewriter.replaceOpWithNewOp<IREE::HAL::BufferViewBufferOp>(
        importOp, bufferType, bufferView);

    // Assert the storage is compatible with our expected device and usage.
    auto targetAllocator = lookupAllocatorFor(importOp, rewriter);
    auto resourceType =
        llvm::cast<IREE::Stream::ResourceType>(importOp.getResult().getType());
    if (failed(buildStorageAssertions(loc, bufferOp.getResult(), message,
                                      targetAllocator, adaptor.getResultSize(),
                                      resourceType, rewriter))) {
      return failure();
    }

    return success();
  }
};

struct TensorExportBufferOpPattern
    : public StreamConversionPattern<IREE::Stream::TensorExportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TensorExportOp exportOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::isa<IREE::HAL::BufferType>(exportOp.getResult().getType())) {
      return failure();
    }
    rewriter.replaceOp(exportOp, adaptor.getSource());
    return success();
  }
};

struct TensorExportBufferViewOpPattern
    : public StreamConversionPattern<IREE::Stream::TensorExportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TensorExportOp exportOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto targetType = exportOp.getResult().getType();
    if (!llvm::isa<IREE::HAL::BufferViewType>(targetType) &&
        !llvm::isa<TensorType>(targetType)) {
      return failure();
    }

    auto loc = exportOp.getLoc();
    auto tensorType = llvm::cast<RankedTensorType>(adaptor.getSourceEncoding());
    auto dynamicDims = adaptor.getSourceEncodingDims();

    // NOTE: we should have verified supported encodings/types at entry into the
    // HAL pipeline.
    auto encodingType = rewriter.create<IREE::HAL::EncodingTypeOp>(
        loc, tensorType.getEncoding());
    auto elementType = rewriter.create<IREE::HAL::ElementTypeOp>(
        loc, tensorType.getElementType());

    // Flatten static + dynamic shape dimensions.
    SmallVector<Value> dims;
    unsigned dynamicIdx = 0;
    for (int64_t idx = 0; idx < tensorType.getRank(); ++idx) {
      if (tensorType.isDynamicDim(idx)) {
        dims.push_back(dynamicDims[dynamicIdx++]);
      } else {
        dims.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, tensorType.getDimSize(idx)));
      }
    }

    rewriter.replaceOpWithNewOp<IREE::HAL::BufferViewCreateOp>(
        exportOp, adaptor.getSource(),
        rewriter.create<arith::ConstantIndexOp>(loc, 0),
        adaptor.getSourceSize(), elementType, encodingType, dims);
    return success();
  }
};

struct TensorTraceOpPattern
    : public StreamConversionPattern<IREE::Stream::TensorTraceOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TensorTraceOp traceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> bufferViews;
    auto resourceEncodingDims = adaptor.getResourceEncodingDims();
    for (auto [resource, resourceSize, resourceEncoding] : llvm::zip_equal(
             adaptor.getResources(), adaptor.getResourceSizes(),
             adaptor.getResourceEncodings().getAsRange<TypeAttr>())) {
      int64_t dynamicDimCount =
          cast<ShapedType>(resourceEncoding.getValue()).getNumDynamicDims();
      bufferViews.push_back(rewriter.create<IREE::Stream::TensorExportOp>(
          traceOp.getLoc(), rewriter.getType<IREE::HAL::BufferViewType>(),
          resource, resourceEncoding,
          resourceEncodingDims.take_front(dynamicDimCount), resourceSize,
          /*affinity=*/IREE::Stream::AffinityAttr{}));
      resourceEncodingDims = resourceEncodingDims.drop_front(dynamicDimCount);
    }
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferViewTraceOp>(
        traceOp, traceOp.getKeyAttr(), bufferViews);
    return success();
  }
};

struct CmdFlushOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdFlushOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdFlushOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): HAL command buffer op for flush.
    rewriter.eraseOp(op);
    return success();
  }
};

struct CmdInvalidateOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdInvalidateOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdInvalidateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): HAL command buffer op for invalidate.
    rewriter.eraseOp(op);
    return success();
  }
};

struct CmdDiscardOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdDiscardOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdDiscardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): HAL command buffer op for discard.
    rewriter.eraseOp(op);
    return success();
  }
};

struct CmdFillOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdFillOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdFillOp fillOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto commandBufferMapping = mapping->lookupCommandBufferFor(fillOp);
    auto targetBinding = commandBufferMapping.resolveBinding(
        fillOp.getLoc(), fillOp.getTarget(), adaptor.getTarget(),
        adaptor.getTargetOffset(), adaptor.getTargetLength(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::HAL::CommandBufferFillBufferOp>(
        fillOp, commandBufferMapping.getHandle(), targetBinding.buffer,
        targetBinding.byteOffset, targetBinding.byteLength, adaptor.getValue());
    return success();
  }
};

struct CmdCopyOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdCopyOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto commandBufferMapping = mapping->lookupCommandBufferFor(op);
    auto sourceBinding = commandBufferMapping.resolveBinding(
        op.getLoc(), op.getSource(), adaptor.getSource(),
        adaptor.getSourceOffset(), adaptor.getLength(), rewriter);
    auto targetBinding = commandBufferMapping.resolveBinding(
        op.getLoc(), op.getTarget(), adaptor.getTarget(),
        adaptor.getTargetOffset(), adaptor.getLength(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::HAL::CommandBufferCopyBufferOp>(
        op, commandBufferMapping.getHandle(), sourceBinding.buffer,
        sourceBinding.byteOffset, targetBinding.buffer,
        targetBinding.byteOffset, adaptor.getLength());
    return success();
  }
};

// NOTE: this relies on the enums being the same today. Ew.
static IREE::HAL::CollectiveAttr
convertCollectiveAttr(IREE::Stream::CollectiveAttr sourceAttr) {
  auto convertReductionOp =
      [](std::optional<IREE::Stream::CollectiveReductionOp> op)
      -> std::optional<IREE::HAL::CollectiveReductionOp> {
    if (!op.has_value())
      return std::nullopt;
    return static_cast<IREE::HAL::CollectiveReductionOp>(op.value());
  };
  return IREE::HAL::CollectiveAttr::get(
      sourceAttr.getContext(),
      static_cast<IREE::HAL::CollectiveKind>(sourceAttr.getKind()),
      convertReductionOp(sourceAttr.getReduction()),
      static_cast<IREE::HAL::CollectiveElementType>(
          sourceAttr.getElementType()));
}

struct CmdCollectiveOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdCollectiveOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdCollectiveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto commandBufferMapping = mapping->lookupCommandBufferFor(op);

    IREE::HAL::BindingValue sendBinding;
    IREE::HAL::BindingValue recvBinding;
    switch (adaptor.getOp().getKind()) {
    default:
      assert(adaptor.getResources().size() == 2 && "should have verified");
      sendBinding = commandBufferMapping.resolveBinding(
          op.getLoc(), op.getResources()[0], adaptor.getResources()[0],
          adaptor.getResourceOffsets()[0], adaptor.getResourceLengths()[0],
          rewriter);
      recvBinding = commandBufferMapping.resolveBinding(
          op.getLoc(), op.getResources()[1], adaptor.getResources()[1],
          adaptor.getResourceOffsets()[1], adaptor.getResourceLengths()[1],
          rewriter);
      break;
    case IREE::Stream::CollectiveKind::Send:
      assert(adaptor.getResources().size() == 1 && "should have verified");
      sendBinding = commandBufferMapping.resolveBinding(
          op.getLoc(), op.getResources()[0], adaptor.getResources()[0],
          adaptor.getResourceOffsets()[0], adaptor.getResourceLengths()[0],
          rewriter);
      break;
    case IREE::Stream::CollectiveKind::Recv:
      assert(adaptor.getResources().size() == 1 && "should have verified");
      recvBinding = commandBufferMapping.resolveBinding(
          op.getLoc(), op.getResources()[0], adaptor.getResources()[0],
          adaptor.getResourceOffsets()[0], adaptor.getResourceLengths()[0],
          rewriter);
      break;
    }

    rewriter.replaceOpWithNewOp<IREE::HAL::CommandBufferCollectiveOp>(
        op, commandBufferMapping.getHandle(), adaptor.getChannel(),
        convertCollectiveAttr(adaptor.getOp()), adaptor.getElementCount(),
        adaptor.getParam(), sendBinding.buffer, sendBinding.byteOffset,
        sendBinding.byteLength, recvBinding.buffer, recvBinding.byteOffset,
        recvBinding.byteLength);
    return success();
  }
};

// TODO(#18154): switch to dispatch2.
struct CmdDispatchOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdDispatchOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdDispatchOp dispatchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = dispatchOp.getLoc();
    auto commandBufferMapping = mapping->lookupCommandBufferFor(dispatchOp);

    // TODO(multi-device): reusable command buffers done at the stream level may
    // make this difficult. For now we assume each stream region being lowered
    // has a singular affinity that may itself reference multiple devices in the
    // future but currently uniquely identifies a device.
    auto affinityAttr = IREE::Stream::AffinityAttr::lookupOrDefault(dispatchOp);

    // Get the device handle we're executing against in this execution region.
    // Note that this is a dynamic value: we have to treat the device as unknown
    // here.
    Value device = rewriter.create<IREE::HAL::CommandBufferDeviceOp>(
        loc, rewriter.getType<IREE::HAL::DeviceType>(),
        commandBufferMapping.getHandle());

    // Prepare for variant switch table by gathering the conditions selecting
    // each variant.
    SmallVector<int64_t> caseIndices;
    SmallVector<std::pair<SymbolRefAttr, IREE::HAL::ExecutableExportOp>>
        caseExportOps;
    dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
      // NOTE: slow lookup!
      auto exportOp =
          SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableExportOp>(
              dispatchOp, entryPointAttr);
      assert(exportOp && "dispatch target export not found");
      caseIndices.push_back(caseIndices.size());
      caseExportOps.push_back(std::make_pair(entryPointAttr, exportOp));
    });

    auto recordDispatch = [&](SymbolRefAttr entryPointAttr,
                              IREE::HAL::ExecutableExportOp exportOp,
                              OpBuilder &builder) {
      // Record push constants and buffer bindings.
      recordParameters(loc, affinityAttr, device, commandBufferMapping,
                       exportOp, dispatchOp, adaptor, builder);

      // Dispatch with a target-specific workgroup count.
      auto workgroupCount = exportOp.calculateWorkgroupCount(
          loc, device, adaptor.getWorkload(), builder);
      Value executable = builder.create<IREE::HAL::ExecutableLookupOp>(
          loc, builder.getType<IREE::HAL::ExecutableType>(), device,
          entryPointAttr.getRootReference().getValue());
      Value ordinal = builder.create<IREE::HAL::ExecutableExportOrdinalOp>(
          loc, builder.getIndexType(), entryPointAttr);
      auto flags = builder.getAttr<IREE::HAL::DispatchFlagsAttr>(
          IREE::HAL::DispatchFlags::None);
      return builder.create<IREE::HAL::CommandBufferDispatchOp>(
          loc, commandBufferMapping.getHandle(), executable, ordinal,
          workgroupCount[0], workgroupCount[1], workgroupCount[2], flags);
    };

    // If there is only one variant we can emit that directly without a
    // conditional check. The same result should occur later on but it saves
    // a lot of IR during generation if we know we can avoid it.
    if (caseExportOps.size() == 1) {
      auto [entryPointAttr, exportOp] = caseExportOps.front();
      rewriter.replaceOp(dispatchOp,
                         recordDispatch(entryPointAttr, exportOp, rewriter));
    } else {
      // Select the variant index.
      Value selectedIndex = buildIfElseTree(
          loc, caseExportOps.size(),
          [&](Location loc, size_t i, OpBuilder &builder) {
            auto exportOp = caseExportOps[i].second;
            auto variantOp =
                exportOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
            return variantOp.buildCondition(device, rewriter);
          },
          rewriter);

      // Allow each variant to define how it is dispatched.
      auto switchOp = rewriter.create<scf::IndexSwitchOp>(
          loc, TypeRange{}, selectedIndex, caseIndices, caseIndices.size());
      for (size_t i = 0; i < caseExportOps.size(); ++i) {
        auto [entryPointAttr, exportOp] = caseExportOps[i];
        auto &caseBlock = switchOp.getCaseRegions()[i].emplaceBlock();
        auto caseBuilder = OpBuilder::atBlockBegin(&caseBlock);
        recordDispatch(entryPointAttr, exportOp, caseBuilder);
        caseBuilder.create<scf::YieldOp>(loc);
      }

      // Fallback for no available variant. Today we just no-op as executable
      // loading should have already failed.
      auto &defaultBlock = switchOp.getDefaultRegion().emplaceBlock();
      auto defaultBuilder = OpBuilder::atBlockBegin(&defaultBlock);
      defaultBuilder.create<scf::YieldOp>(loc);

      rewriter.replaceOp(dispatchOp, switchOp);
    }

    return success();
  }

  void recordParameters(Location loc, IREE::Stream::AffinityAttr affinityAttr,
                        Value device,
                        CommandBufferConversionMapping &commandBufferMapping,
                        IREE::HAL::ExecutableExportOp exportOp,
                        IREE::Stream::CmdDispatchOp dispatchOp,
                        OpAdaptor adaptor, OpBuilder &builder) const {
    auto layoutAttr = exportOp.getLayout();
    auto pipelineLayout =
        builder
            .create<IREE::HAL::PipelineLayoutLookupOp>(
                loc, IREE::HAL::PipelineLayoutType::get(loc.getContext()),
                device, layoutAttr)
            .getResult();

    // Push constant values.
    // TODO(#5322): symbolic push constant names on the hal.interface so we can
    // sparsely pack these.
    if (!adaptor.getUniformOperands().empty()) {
      int pushConstantBase = 0; // always 0 today
      SmallVector<Value> pushConstants;
      for (auto operand : adaptor.getUniformOperands()) {
        assert(
            operand.getType().isInteger(32) &&
            "expected only i32 values after iree-hal-pack-dispatch-operands");
        pushConstants.push_back(operand);
      }
      builder.create<IREE::HAL::CommandBufferPushConstantsOp>(
          loc, commandBufferMapping.getHandle(), pipelineLayout,
          builder.getIndexAttr(pushConstantBase), pushConstants);
    }

    // Push descriptor bindings set by set.
    // We build a table of all sets in the layout then populate the bindings as
    // we walk the flattened/unordered resource list. After we've collected all
    // of the bindings we issue the command for that set.
    auto bindingAttrs = IREE::HAL::getInterfaceBindingAttrs(
        exportOp, dispatchOp.getResources().size());
    int64_t maxSet = llvm::max_element(bindingAttrs, [](auto lhs, auto rhs) {
                       return lhs.getSet() < rhs.getSet();
                     })->getSet();
    SmallVector<SmallVector<IREE::HAL::DescriptorSetBindingValue>> setBindings;
    setBindings.resize(maxSet + 1);
    for (auto [i, bindingAttr] : llvm::enumerate(bindingAttrs)) {
      auto setLayoutFlags =
          layoutAttr.getSetLayout(bindingAttr.getSet())
              .getFlags()
              .value_or(IREE::HAL::DescriptorSetLayoutFlags::None);
      IREE::HAL::DescriptorSetBindingValue binding;
      binding.ordinal =
          builder.create<arith::ConstantIndexOp>(loc, bindingAttr.getBinding());
      if (bitEnumContainsAll(setLayoutFlags,
                             IREE::HAL::DescriptorSetLayoutFlags::Indirect)) {
        // Indirect binding resolved through the cached command buffer binding
        // table. The buffer recorded in the descriptor is a slot ordinal into
        // the binding table. Note that the range may be adjusted based on the
        // range bound to the slot in the table.
        auto resolvedBinding = commandBufferMapping.resolveBinding(
            loc, dispatchOp.getResources()[i], adaptor.getResources()[i],
            adaptor.getResourceOffsets()[i], adaptor.getResourceLengths()[i],
            builder);
        binding.buffer = resolvedBinding.buffer;
        binding.byteOffset = resolvedBinding.byteOffset;
        binding.byteLength = resolvedBinding.byteLength;
      } else {
        // Direct binding referencing the buffer and range provided on the op.
        binding.buffer = adaptor.getResources()[i];
        binding.byteOffset = adaptor.getResourceOffsets()[i];
        binding.byteLength = adaptor.getResourceLengths()[i];
      }
      setBindings[bindingAttr.getSet()].push_back(binding);
    }
    for (auto [set, bindings] : llvm::enumerate(setBindings)) {
      if (!bindings.empty()) {
        builder.create<IREE::HAL::CommandBufferPushDescriptorSetOp>(
            loc, commandBufferMapping.getHandle(), pipelineLayout, set,
            bindings);
      }
    }
  }
};

struct CmdDispatch2OpPattern
    : public StreamConversionPattern<IREE::Stream::CmdDispatchOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdDispatchOp dispatchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = dispatchOp.getLoc();
    auto commandBufferMapping = mapping->lookupCommandBufferFor(dispatchOp);

    // TODO(multi-device): reusable command buffers done at the stream level may
    // make this difficult. For now we assume each stream region being lowered
    // has a singular affinity that may itself reference multiple devices in the
    // future but currently uniquely identifies a device.
    auto affinityAttr = IREE::Stream::AffinityAttr::lookupOrDefault(dispatchOp);

    // Get the device handle we're executing against in this execution region.
    // Note that this is a dynamic value: we have to treat the device as unknown
    // here.
    Value device = rewriter.create<IREE::HAL::CommandBufferDeviceOp>(
        loc, rewriter.getType<IREE::HAL::DeviceType>(),
        commandBufferMapping.getHandle());

    // Prepare for variant switch table by gathering the conditions selecting
    // each variant.
    SmallVector<int64_t> caseIndices;
    SmallVector<std::pair<SymbolRefAttr, IREE::HAL::ExecutableExportOp>>
        caseExportOps;
    dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
      // NOTE: slow lookup!
      auto exportOp =
          SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableExportOp>(
              dispatchOp, entryPointAttr);
      assert(exportOp && "dispatch target export not found");
      caseIndices.push_back(caseIndices.size());
      caseExportOps.push_back(std::make_pair(entryPointAttr, exportOp));
    });

    // If there is only one variant we can emit that directly without a
    // conditional check. The same result should occur later on but it saves
    // a lot of IR during generation if we know we can avoid it.
    if (caseExportOps.size() == 1) {
      auto [entryPointAttr, exportOp] = caseExportOps.front();
      rewriter.replaceOp(dispatchOp,
                         emitDispatchOp(loc, affinityAttr, device,
                                        commandBufferMapping, exportOp,
                                        entryPointAttr, dispatchOp, adaptor,
                                        rewriter));
    } else {
      // Select the variant index.
      Value selectedIndex = buildIfElseTree(
          loc, caseExportOps.size(),
          [&](Location loc, size_t i, OpBuilder &builder) {
            auto exportOp = caseExportOps[i].second;
            auto variantOp =
                exportOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
            return variantOp.buildCondition(device, rewriter);
          },
          rewriter);

      // Allow each variant to define how it is dispatched.
      auto switchOp = rewriter.create<scf::IndexSwitchOp>(
          loc, TypeRange{}, selectedIndex, caseIndices, caseIndices.size());
      for (size_t i = 0; i < caseExportOps.size(); ++i) {
        auto [entryPointAttr, exportOp] = caseExportOps[i];
        auto &caseBlock = switchOp.getCaseRegions()[i].emplaceBlock();
        auto caseBuilder = OpBuilder::atBlockBegin(&caseBlock);
        emitDispatchOp(loc, affinityAttr, device, commandBufferMapping,
                       exportOp, entryPointAttr, dispatchOp, adaptor,
                       caseBuilder);
        caseBuilder.create<scf::YieldOp>(loc);
      }

      // Fallback for no available variant. Today we just no-op as executable
      // loading should have already failed.
      auto &defaultBlock = switchOp.getDefaultRegion().emplaceBlock();
      auto defaultBuilder = OpBuilder::atBlockBegin(&defaultBlock);
      defaultBuilder.create<scf::YieldOp>(loc);

      rewriter.replaceOp(dispatchOp, switchOp);
    }

    return success();
  }

  Operation *emitDispatchOp(
      Location loc, IREE::Stream::AffinityAttr affinityAttr, Value device,
      CommandBufferConversionMapping &commandBufferMapping,
      IREE::HAL::ExecutableExportOp exportOp, SymbolRefAttr entryPointAttr,
      IREE::Stream::CmdDispatchOp dispatchOp, OpAdaptor adaptor,
      OpBuilder &builder) const {
    auto workgroupCount = exportOp.calculateWorkgroupCount(
        loc, device, adaptor.getWorkload(), builder);

    Value executable = builder.create<IREE::HAL::ExecutableLookupOp>(
        loc, builder.getType<IREE::HAL::ExecutableType>(), device,
        entryPointAttr.getRootReference().getValue());
    Value ordinal = builder.create<IREE::HAL::ExecutableExportOrdinalOp>(
        loc, builder.getIndexType(), entryPointAttr);

    // TODO(#18154): simplify bindings by removing descriptor sets.
    auto layoutAttr = exportOp.getLayout();
    auto bindingAttrs = IREE::HAL::getInterfaceBindingAttrs(
        exportOp, dispatchOp.getResources().size());
    SmallVector<IREE::HAL::BindingValue> bindings;
    for (auto [i, bindingAttr] : llvm::enumerate(bindingAttrs)) {
      auto descriptorFlags = layoutAttr.getSetLayout(bindingAttr.getSet())
                                 .getBinding(i)
                                 .getFlags();
      IREE::HAL::BindingValue binding;
      if (bitEnumContainsAll(descriptorFlags,
                             IREE::HAL::DescriptorFlags::Indirect)) {
        // Indirect binding resolved through the cached command buffer binding
        // table. The buffer recorded in the descriptor is a slot ordinal into
        // the binding table. Note that the range may be adjusted based on the
        // range bound to the slot in the table.
        auto resolvedBinding = commandBufferMapping.resolveBinding(
            loc, dispatchOp.getResources()[i], adaptor.getResources()[i],
            adaptor.getResourceOffsets()[i], adaptor.getResourceLengths()[i],
            builder);
        binding.buffer = resolvedBinding.buffer;
        binding.byteOffset = resolvedBinding.byteOffset;
        binding.byteLength = resolvedBinding.byteLength;
      } else {
        // Direct binding referencing the buffer and range provided on the op.
        binding.buffer = adaptor.getResources()[i];
        binding.byteOffset = adaptor.getResourceOffsets()[i];
        binding.byteLength = adaptor.getResourceLengths()[i];
      }
      bindings.push_back(binding);
    }

    auto flags = IREE::HAL::DispatchFlags::None;

    return builder.create<IREE::HAL::CommandBufferDispatch2Op>(
        loc, commandBufferMapping.getHandle(), executable, ordinal,
        workgroupCount, adaptor.getUniformOperands(), bindings, flags);
  }
};

struct CmdFuncOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdFuncOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdFuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newArgTypes;
    SmallVector<DictionaryAttr> newArgAttrs;
    newArgTypes.push_back(rewriter.getType<IREE::HAL::CommandBufferType>());
    newArgAttrs.push_back(rewriter.getDictionaryAttr({})); // command buffer
    funcOp.getAllArgAttrs(newArgAttrs);
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(funcOp.getArgumentTypes(),
                                                newArgTypes)) ||
        failed(getTypeConverter()->convertTypes(funcOp.getResultTypes(),
                                                newResultTypes))) {
      return rewriter.notifyMatchFailure(funcOp, "failed to convert types");
    }
    auto newOp = rewriter.replaceOpWithNewOp<IREE::Util::FuncOp>(
        funcOp, funcOp.getNameAttr(),
        rewriter.getFunctionType(newArgTypes, newResultTypes),
        /*tied_operands=*/ArrayAttr{}, funcOp.getSymVisibilityAttr(),
        rewriter.getArrayAttr(
            ArrayRef<Attribute>(newArgAttrs.data(), newArgAttrs.size())),
        funcOp.getAllResultAttrs(),
        /*inlining_policy=*/IREE::Util::InliningPolicyAttrInterface{});
    newOp->setDialectAttrs(funcOp->getDialectAttrs());
    return success();
  }
};

struct CmdCallOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdCallOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdCallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto commandBufferMapping = mapping->lookupCommandBufferFor(callOp);

    // Always pass the command buffer as the first arg.
    SmallVector<Value> operands;
    operands.push_back(commandBufferMapping.getHandle());
    size_t resourceIndex = 0;
    for (auto [originalOperand, convertedOperand] : llvm::zip_equal(
             callOp.getResourceOperands(), adaptor.getResourceOperands())) {
      if (llvm::isa<IREE::Stream::ResourceType>(originalOperand.getType())) {
        // Resource type, add offset/length.
        operands.push_back(convertedOperand);
        operands.push_back(adaptor.getResourceOperandOffsets()[resourceIndex]);
        operands.push_back(adaptor.getResourceOperandLengths()[resourceIndex]);
        ++resourceIndex;
      } else {
        // Primitive/custom type.
        operands.push_back(convertedOperand);
      }
    }

    SmallVector<Type> resultTypes;
    for (auto result : callOp.getResults()) {
      SmallVector<Type> convertedTypes;
      if (failed(getTypeConverter()->convertType(result.getType(),
                                                 convertedTypes))) {
        return rewriter.notifyMatchFailure(callOp.getLoc(),
                                           "unconvertable result type");
      }
      llvm::append_range(resultTypes, convertedTypes);
    }

    rewriter.replaceOpWithNewOp<IREE::Util::CallOp>(
        callOp, resultTypes, callOp.getCallee(), operands,
        /*tied_operands=*/ArrayAttr{});
    return success();
  }
};

static void insertSerializationBarriers(Location loc, Block &block,
                                        Value commandBuffer,
                                        OpBuilder builder) {
  // TODO(benvanik): derive based on the type of operations that surround the
  // barriers. Can use deriveCommandCategories on the ranges to see what kind
  // of ops happen above and below, but really some analysis is required.
  auto sourceStage = IREE::HAL::ExecutionStageBitfield::CommandRetire |
                     IREE::HAL::ExecutionStageBitfield::Transfer |
                     IREE::HAL::ExecutionStageBitfield::Dispatch;
  auto targetStage = IREE::HAL::ExecutionStageBitfield::CommandIssue |
                     IREE::HAL::ExecutionStageBitfield::Transfer |
                     IREE::HAL::ExecutionStageBitfield::Dispatch;
  auto flags = IREE::HAL::ExecutionBarrierFlagBitfield::None;

  // Insert barriers after every op.
  // Note that we can't mutate the block while iterating it so we first grab
  // all the original ops.
  SmallVector<Operation *> serialOps;
  for (auto &op : block)
    serialOps.push_back(&op);
  for (auto *op : serialOps) {
    if (op->hasTrait<OpTrait::IsTerminator>())
      continue;
    builder.setInsertionPointAfter(op);
    builder.create<IREE::HAL::CommandBufferExecutionBarrierOp>(
        loc, commandBuffer, sourceStage, targetStage, flags);
  }
}

struct CmdExecuteOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdExecuteOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdExecuteOp executeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = executeOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(executeOp, rewriter);

    // Calculate the indirect buffer references used within the command buffer
    // by analyzing captured resources. This analysis will be used by subsequent
    // conversion to decide between embedding the direct buffer references or
    // indirect ones. We only do this if the execution region is reused.
    IndexSet indexSet(loc, rewriter);
    BindingTable bindingTable;
    if (!executeOp.getOnce() && clIndirectCommandBuffers) {
      bindingTable = BindingTable(executeOp, adaptor.getResourceOperands(),
                                  adaptor.getResourceOperandSizes(), indexSet);
    }
    auto bindingTableValues = llvm::to_vector(bindingTable.getValues());

    // If the execute op is one-shot or there's no indirect bindings then mark
    // the command buffer one-shot.
    IREE::HAL::CommandBufferModeBitfield modes =
        IREE::HAL::CommandBufferModeBitfield::None;
    if (!bindingTable.isSupported() || bindingTable.empty()) {
      modes = modes | IREE::HAL::CommandBufferModeBitfield::OneShot;
      if (!executeOp.getAwaitTimepoint()) {
        modes =
            modes | IREE::HAL::CommandBufferModeBitfield::AllowInlineExecution;
      }
    }

    // Derive the command buffer type based on the kind of operations present.
    // This can help the submission get routed to appropriate hardware queues
    // (like dedicated DMA controllers).
    auto commandCategories = deriveCommandCategories(executeOp.getBody());

    // Create, record, and finalize a command buffer at the current rewriter
    // insertion point. Returns the command buffer handle.
    auto recordCommandBuffer =
        [&](Value device, Value queueAffinity,
            ConversionPatternRewriter &rewriter) -> Value {
      // Create a new command buffer for recording.
      Value bindingTableCapacity =
          bindingTable.empty() ? Value{}
                               : rewriter.create<arith::ConstantIndexOp>(
                                     loc, bindingTable.size());
      Value commandBuffer = rewriter.create<IREE::HAL::CommandBufferCreateOp>(
          loc, rewriter.getType<IREE::HAL::CommandBufferType>(), device, modes,
          commandCategories, queueAffinity, bindingTableCapacity);
      mapping->mapCommandBuffer(executeOp, commandBuffer,
                                std::move(bindingTable));

      // Run through the execution region and serialize execution by inserting
      // barriers. Nested regions may elide barriers as needed.
      auto &bodyBlock = executeOp.getBody().front();
      insertSerializationBarriers(loc, bodyBlock, commandBuffer,
                                  OpBuilder::atBlockBegin(&bodyBlock));

      // Begin/end recording and inline the execution region between them.
      auto endOp = rewriter.create<IREE::HAL::CommandBufferFinalizeOp>(
          loc, commandBuffer);
      rewriter.inlineBlockBefore(&executeOp.getBody().front(), endOp,
                                 adaptor.getResourceOperands());

      // Return the command buffer handle.
      return commandBuffer;
    };

    // If reusable then we can memoize the command buffer by nesting it within
    // a memoization region and otherwise we inline the recording directly into
    // the original execution site.
    Value commandBuffer;
    if (!bitEnumContainsAll(modes,
                            IREE::HAL::CommandBufferModeBitfield::OneShot)) {
      auto memoizeOp = rewriter.create<IREE::HAL::DeviceMemoizeOp>(
          loc, rewriter.getType<IREE::HAL::CommandBufferType>(), device,
          queueAffinity);
      auto ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(&memoizeOp.getBody().emplaceBlock());
      rewriter.create<IREE::HAL::ReturnOp>(
          loc, recordCommandBuffer(device, queueAffinity, rewriter));
      rewriter.restoreInsertionPoint(ip);
      commandBuffer = memoizeOp.getResult(0);
    } else {
      commandBuffer = recordCommandBuffer(device, queueAffinity, rewriter);
    }

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, executeOp.getResultTimepoint(), rewriter);

    // Queue execution.
    if (bindingTableValues.empty()) {
      rewriter.create<IREE::HAL::DeviceQueueExecuteOp>(
          loc, device, queueAffinity, waitFence, signalFence,
          ValueRange{commandBuffer});
    } else {
      rewriter.create<IREE::HAL::DeviceQueueExecuteIndirectOp>(
          loc, device, queueAffinity, waitFence, signalFence, commandBuffer,
          bindingTableValues);
    }

    rewriter.replaceOp(executeOp, signalFence);
    return success();
  }
};

struct CmdSerialOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdSerialOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdSerialOp serialOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto commandBufferMapping = mapping->lookupCommandBufferFor(serialOp);

    // Run through the execution region and serialize execution by inserting
    // barriers. Nested regions may elide barriers as needed.
    auto &bodyBlock = serialOp.getBody().front();
    insertSerializationBarriers(serialOp.getLoc(), bodyBlock,
                                commandBufferMapping.getHandle(),
                                OpBuilder::atBlockBegin(&bodyBlock));

    // Inline the serial execution region.
    rewriter.inlineBlockBefore(&serialOp.getBody().front(), serialOp);
    rewriter.eraseOp(serialOp);
    return success();
  }
};

struct CmdConcurrentOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdConcurrentOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdConcurrentOp concurrentOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Inline the concurrent execution region.
    // TODO(benvanik): split barriers (event set/wait) when nesting.
    rewriter.inlineBlockBefore(&concurrentOp.getBody().front(), concurrentOp);
    rewriter.eraseOp(concurrentOp);
    return success();
  }
};

struct TimepointImmediateOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointImmediateOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointImmediateOp immediateOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Util::NullOp>(
        immediateOp, rewriter.getType<IREE::HAL::FenceType>());
    return success();
  }
};

struct TimepointImportOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointImportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointImportOp importOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle imports from HAL semaphores _or_ fences.
    auto operands = adaptor.getOperands();
    if (operands.size() == 1 &&
        llvm::isa<IREE::HAL::FenceType>(operands[0].getType())) {
      rewriter.replaceOp(importOp, operands[0]);
      return success();
    } else {
      return rewriter.notifyMatchFailure(
          importOp, "only imports from HAL fences are supported");
    }
  }
};

struct TimepointExportOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointExportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointExportOp exportOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle exports into HAL fences.
    if (exportOp.getNumResults() != 1 ||
        !llvm::isa<IREE::HAL::FenceType>(exportOp.getResult(0).getType())) {
      return rewriter.notifyMatchFailure(
          exportOp, "only exports to HAL fences are supported");
    }
    rewriter.replaceOp(exportOp, adaptor.getAwaitTimepoint());
    return success();
  }
};

struct TimepointChainExternalOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointChainExternalOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointChainExternalOp exportOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle exports into HAL fences.
    auto externalValues = exportOp.getExternalValues();
    if (externalValues.size() != 1 ||
        !llvm::isa<IREE::HAL::FenceType>(externalValues[0].getType())) {
      return rewriter.notifyMatchFailure(
          exportOp, "only exports to HAL fences are supported");
    }
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(exportOp, rewriter);
    rewriter.replaceOpWithNewOp<IREE::HAL::DeviceQueueExecuteOp>(
        exportOp, device, queueAffinity,
        /*wait_fence=*/adaptor.getAwaitTimepoint(),
        /*signal_fence=*/externalValues[0], /*command_buffers=*/ValueRange{});
    return success();
  }
};

struct TimepointJoinOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointJoinOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointJoinOp joinOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::FenceJoinOp>(
        joinOp, rewriter.getType<IREE::HAL::FenceType>(),
        adaptor.getAwaitTimepoints());
    return success();
  }
};

struct TimepointBarrierOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointBarrierOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointBarrierOp barrierOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace with a signaled fence.
    // NOTE: this assumes that if this op still exists the input resource is
    // already available. If it isn't then timepoint propagation should have
    // replaced the signal op with the producing timepoint.
    Value nullFence = rewriter.create<IREE::Util::NullOp>(
        barrierOp.getLoc(), rewriter.getType<IREE::HAL::FenceType>());
    rewriter.replaceOp(barrierOp, {adaptor.getResource(), nullFence});
    return success();
  }
};

struct TimepointAwaitOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointAwaitOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointAwaitOp awaitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = awaitOp.getLoc();

    // Perform the blocking wait.
    Value timeoutMillis = rewriter.create<arith::ConstantIntOp>(loc, -1, 32);
    auto fenceOp = rewriter.create<IREE::HAL::FenceAwaitOp>(
        loc, rewriter.getI32Type(), timeoutMillis, adaptor.getAwaitTimepoint());
    rewriter.create<IREE::Util::StatusCheckOkOp>(loc, fenceOp.getStatus(),
                                                 "failed to wait on timepoint");

    // Pass along operands.
    rewriter.replaceOp(awaitOp, adaptor.getResourceOperands());
    return success();
  }
};

struct ChannelCreateOpPattern
    : public StreamConversionPattern<IREE::Stream::ChannelCreateOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ChannelCreateOp createOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(createOp, rewriter);
    Value neg1I32;
    auto getDefault = [&]() {
      if (!neg1I32) {
        neg1I32 =
            rewriter.create<arith::ConstantIntOp>(createOp.getLoc(), -1, 32);
      }
      return neg1I32;
    };
    Value id = adaptor.getId();
    if (!id) {
      id = rewriter.create<IREE::Util::NullOp>(
          createOp.getLoc(), rewriter.getType<IREE::Util::BufferType>());
    }
    Value group =
        adaptor.getGroupAttr()
            ? rewriter
                  .create<IREE::Util::BufferConstantOp>(
                      createOp.getLoc(),
                      /*name=*/StringAttr{}, /*value=*/adaptor.getGroupAttr(),
                      /*alignment=*/IntegerAttr{}, /*mime_type=*/StringAttr{})
                  .getResult()
            : rewriter
                  .create<IREE::Util::NullOp>(
                      createOp.getLoc(),
                      rewriter.getType<IREE::Util::BufferType>())
                  .getResult();
    Value rank =
        adaptor.getRank()
            ? rewriter.create<arith::IndexCastOp>(
                  createOp.getLoc(), rewriter.getI32Type(), adaptor.getRank())
            : getDefault();
    Value count =
        adaptor.getRank()
            ? rewriter.create<arith::IndexCastOp>(
                  createOp.getLoc(), rewriter.getI32Type(), adaptor.getCount())
            : getDefault();
    rewriter.replaceOpWithNewOp<IREE::HAL::ChannelCreateOp>(
        createOp, rewriter.getType<IREE::HAL::ChannelType>(), device,
        queueAffinity, /*flags=*/rewriter.getI32IntegerAttr(0), id, group, rank,
        count);
    return success();
  }
};

struct ChannelSplitOpPattern
    : public StreamConversionPattern<IREE::Stream::ChannelSplitOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ChannelSplitOp splitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value color = rewriter.create<arith::IndexCastOp>(
        splitOp.getLoc(), rewriter.getI32Type(), adaptor.getColor());
    Value key = rewriter.create<arith::IndexCastOp>(
        splitOp.getLoc(), rewriter.getI32Type(), adaptor.getKey());
    rewriter.replaceOpWithNewOp<IREE::HAL::ChannelSplitOp>(
        splitOp, rewriter.getType<IREE::HAL::ChannelType>(),
        adaptor.getChannel(), color, key,
        /*flags=*/rewriter.getI32IntegerAttr(0));
    return success();
  }
};

struct ChannelRankOpPattern
    : public StreamConversionPattern<IREE::Stream::ChannelRankOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ChannelRankOp rankOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<IREE::HAL::ChannelRankAndCountOp>(
        rankOp.getLoc(), rewriter.getI32Type(), rewriter.getI32Type(),
        adaptor.getChannel());
    Value indexRank = rewriter.create<arith::IndexCastOp>(
        rankOp.getLoc(), rewriter.getIndexType(), newOp.getRank());
    rewriter.replaceOp(rankOp, indexRank);
    return success();
  }
};

struct ChannelCountOpPattern
    : public StreamConversionPattern<IREE::Stream::ChannelCountOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ChannelCountOp countOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<IREE::HAL::ChannelRankAndCountOp>(
        countOp.getLoc(), rewriter.getI32Type(), rewriter.getI32Type(),
        adaptor.getChannel());
    Value indexCount = rewriter.create<arith::IndexCastOp>(
        countOp.getLoc(), rewriter.getIndexType(), newOp.getCount());
    rewriter.replaceOp(countOp, indexCount);
    return success();
  }
};

struct ElideYieldOpPattern
    : public StreamConversionPattern<IREE::Stream::YieldOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::YieldOp yieldOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(yieldOp);
    return success();
  }
};

// Annoying we have to have this here, but there's no attribute converter
// equivalent we have access to so that we could do it in a generic way.
struct GlobalTimepointConversionPattern
    : public OpConversionPattern<IREE::Util::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto initialValue = op.getInitialValue();
    if (!initialValue.has_value())
      return failure();
    if (!llvm::isa<IREE::Stream::TimepointAttr>(*initialValue))
      return failure();
    rewriter.modifyOpInPlace(op, [&]() { op.removeInitialValueAttr(); });
    return success();
  }
};

} // namespace

void populateStreamToHALPatterns(MLIRContext *context,
                                 ConversionTarget &conversionTarget,
                                 TypeConverter &typeConverter,
                                 RewritePatternSet &patterns) {
  conversionTarget.addIllegalDialect<IREE::Stream::StreamDialect>();

  typeConverter.addConversion(
      [=](IREE::Stream::ChannelType type, SmallVectorImpl<Type> &results) {
        // Collective channels are 1:1.
        results.push_back(IREE::HAL::ChannelType::get(context));
        return success();
      });

  typeConverter.addConversion(
      [=](IREE::Stream::FileType type, SmallVectorImpl<Type> &results) {
        results.push_back(IREE::HAL::FileType::get(context));
        return success();
      });

  typeConverter.addConversion(
      [=](IREE::Stream::ResourceType type, SmallVectorImpl<Type> &results) {
        // Resources are just buffers (no shape/encoding/etc).
        results.push_back(IREE::HAL::BufferType::get(context));
        return success();
      });

  typeConverter.addConversion(
      [=](IREE::Stream::TimepointType type, SmallVectorImpl<Type> &results) {
        results.push_back(IREE::HAL::FenceType::get(context));
        return success();
      });

  // Spooky action at a distance:
  patterns.insert<GlobalTimepointConversionPattern>(typeConverter, context);

  auto mapping = std::make_shared<StreamConversionMapping>();

  patterns.insert<ContextResolveOpPattern>(mapping, typeConverter, context);

  patterns.insert<ResourceAllocOpPattern, ResourceAllocaOpPattern,
                  ResourceDeallocaOpPattern, ResourceSizeOpPattern,
                  ResourceTryMapOpPattern, ResourceLoadOpPattern,
                  ResourceStoreOpPattern, ResourceSubviewOpPattern>(
      mapping, typeConverter, context);

  patterns.insert<FileConstantOpPattern, FileReadOpPattern, FileWriteOpPattern>(
      mapping, typeConverter, context);

  patterns.insert<TensorImportBufferOpPattern, TensorImportBufferViewOpPattern,
                  TensorExportBufferOpPattern, TensorExportBufferViewOpPattern,
                  TensorTraceOpPattern>(mapping, typeConverter, context);
  patterns
      .insert<CmdFlushOpPattern, CmdInvalidateOpPattern, CmdDiscardOpPattern,
              CmdFillOpPattern, CmdCopyOpPattern, CmdCollectiveOpPattern,
              CmdFuncOpPattern, CmdCallOpPattern, CmdExecuteOpPattern,
              CmdSerialOpPattern, CmdConcurrentOpPattern>(
          mapping, typeConverter, context);
  // TODO(#18154): drop existing pattern.
  if (clExperimentalDispatch2) {
    patterns.insert<CmdDispatch2OpPattern>(mapping, typeConverter, context);
  } else {
    patterns.insert<CmdDispatchOpPattern>(mapping, typeConverter, context);
  }
  patterns.insert<TimepointImmediateOpPattern, TimepointImportOpPattern,
                  TimepointExportOpPattern, TimepointChainExternalOpPattern,
                  TimepointJoinOpPattern, TimepointBarrierOpPattern,
                  TimepointAwaitOpPattern>(mapping, typeConverter, context);
  patterns.insert<ChannelCreateOpPattern, ChannelSplitOpPattern,
                  ChannelRankOpPattern, ChannelCountOpPattern>(
      mapping, typeConverter, context);
  patterns.insert<ElideYieldOpPattern>(mapping, typeConverter, context);
}

} // namespace mlir::iree_compiler
