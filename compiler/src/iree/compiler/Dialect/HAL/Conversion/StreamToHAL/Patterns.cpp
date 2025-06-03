// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/StreamToHAL/Patterns.h"

#include "iree/compiler/Dialect/HAL/Analysis/Captures.h"
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
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler {

namespace {

static llvm::cl::opt<bool> clIndirectCommandBuffers{
    "iree-hal-indirect-command-buffers",
    llvm::cl::desc("Whether to turn buffer bindings into indirect references "
                   "when recording command buffers."),
    llvm::cl::init(true),
};

// TODO(benvanik): remove when we support capturing dynamic values for reuse.
static llvm::cl::opt<bool> clForceIndirectCommandBuffers{
    "iree-hal-force-indirect-command-buffers",
    llvm::cl::desc("Forces indirect command buffers when they would otherwise "
                   "not be chosen due to the values they capture. They may not "
                   "be reusable but will still be outlined."),
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

    // TODO(multi-device): policy for selecting the appropriate affinity.
    // Today we only support optimal affinities for certain ops as there needs
    // to be some runtime policy hooks to choose otherwise. For any op that
    // ends up here we select the first device in the optimal set.
    if (auto deviceOptimalAttr =
            dyn_cast_if_present<IREE::HAL::DeviceOptimalAttr>(affinityAttr)) {
      affinityAttr = deviceOptimalAttr.getAffinities().front();
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
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();
    auto resourceType =
        cast<IREE::Stream::ResourceType>(allocOp.getResult().getType());
    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
    if (failed(deriveAllowedResourceBufferBits(allocOp.getLoc(), resourceType,
                                               memoryTypes, bufferUsage))) {
      return failure();
    }

    // Lookup the appropriate allocator/queue for allocation based on the buffer
    // propreties.
    auto [allocator, queueAffinity] = lookupAllocatorAndQueueAffinityFor(
        allocOp, memoryTypes, bufferUsage, rewriter);

    auto memoryTypeOp = rewriter.create<IREE::HAL::MemoryTypeOp>(
        allocOp.getLoc(), memoryTypes);
    auto bufferUsageOp = rewriter.create<IREE::HAL::BufferUsageOp>(
        allocOp.getLoc(), bufferUsage);

    rewriter.replaceOpWithNewOp<IREE::HAL::AllocatorAllocateOp>(
        allocOp, bufferType, allocator, queueAffinity, memoryTypeOp, bufferUsageOp,
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

    // Derive buffer propreties from the resource type.
    auto resourceType =
        cast<IREE::Stream::ResourceType>(allocaOp.getResult().getType());
    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
    if (failed(deriveAllowedResourceBufferBits(loc, resourceType, memoryTypes,
                                               bufferUsage))) {
      return failure();
    }
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();

    // Lookup the appropriate device/queue for allocation based on the buffer
    // propreties.
    auto [device, queueAffinity] = lookupDeviceAndQueueAffinityFor(
        allocaOp, memoryTypes, bufferUsage, rewriter);

    // Behavior flags.
    IREE::HAL::AllocaFlagBitfield flags = IREE::HAL::AllocaFlagBitfield::None;
    if (allocaOp.getIndeterminateLifetime()) {
      flags = flags | IREE::HAL::AllocaFlagBitfield::IndeterminateLifetime;
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
        memoryTypes, bufferUsage, adaptor.getStorageSize(), flags);

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

    // Derive buffer propreties from the resource type. This must match the
    // original allocation. If we're uncertain if it does we have to switch to
    // prefer-origin mode.
    auto resourceType =
        cast<IREE::Stream::ResourceType>(deallocaOp.getOperand().getType());
    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
    bool preferOrigin = deallocaOp.getPreferOrigin();
    if (failed(deriveAllowedResourceBufferBits(loc, resourceType, memoryTypes,
                                               bufferUsage))) {
      preferOrigin = true;
    }
    auto [device, queueAffinity] = lookupDeviceAndQueueAffinityFor(
        deallocaOp, memoryTypes, bufferUsage, rewriter);

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, deallocaOp.getResultTimepoint(), rewriter);

    // Route to the origin of the allocation (if available).
    IREE::HAL::DeallocaFlagBitfield flags =
        IREE::HAL::DeallocaFlagBitfield::None;
    if (preferOrigin) {
      flags = flags | IREE::HAL::DeallocaFlagBitfield::PreferOrigin;
    }

    // Queue deallocation.
    rewriter.create<IREE::HAL::DeviceQueueDeallocaOp>(
        loc, device, queueAffinity, waitFence, signalFence,
        adaptor.getOperand(), flags);

    rewriter.replaceOp(deallocaOp, {signalFence});
    return success();
  }
};

struct ResourceRetainOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceRetainOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceRetainOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferAllocationPreserveOp>(
        op, adaptor.getOperand());
    return success();
  }
};

struct ResourceReleaseOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceReleaseOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceReleaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferAllocationDiscardOp>(
        op, rewriter.getI1Type(), adaptor.getOperand());
    return success();
  }
};

struct ResourceIsTerminalOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceIsTerminalOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceIsTerminalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferAllocationIsTerminalOp>(
        op, rewriter.getI1Type(), adaptor.getOperand());
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

    // Lookup the appropriate allocator/queue for allocation based on the buffer
    // propreties.
    auto [allocator, queueAffinity] = lookupAllocatorAndQueueAffinityFor(
        tryMapOp, memoryTypes, bufferUsage, rewriter);

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
        rewriter.getAttr<IREE::HAL::ReadFlagBitfieldAttr>(
            IREE::HAL::ReadFlagBitfield::None));

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
        rewriter.getAttr<IREE::HAL::WriteFlagBitfieldAttr>(
            IREE::HAL::WriteFlagBitfield::None));

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
        targetBinding.byteOffset, targetBinding.byteLength, adaptor.getValue(),
        IREE::HAL::FillFlagBitfield::None);
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
        targetBinding.byteOffset, adaptor.getLength(),
        IREE::HAL::CopyFlagBitfield::None);
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

  // Returns the fully qualified export name (@executable::@variant::@export).
  SymbolRefAttr getExportRef(IREE::HAL::ExecutableExportOp exportOp) const {
    auto variantOp =
        exportOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
    auto executableOp = variantOp->getParentOfType<IREE::HAL::ExecutableOp>();
    return SymbolRefAttr::get(executableOp.getSymNameAttr(),
                              {
                                  FlatSymbolRefAttr::get(variantOp),
                                  FlatSymbolRefAttr::get(exportOp),
                              });
  }

  // Selects the ordinal for the given |baseExportOp| and calculates its
  // workgroup count. The ordinal may be different than the ordinal of the
  // export itself if any fallbacks are specified. Each export condition will be
  // evaluated and the first that matches will be returned.
  //
  // As an example, a fallback chain of @0 -> @1 -> @2 (with @0 being the
  // highest priority) would result in a decision tree:
  //   %ordinal, %workgroups = scf.if %cond0 {
  //     %ordinal0 = hal.executable.export.ordinal @0
  //     %workgroups0 = calculate for @0
  //     scf.yield %ordinal0, %workgroups0
  //   } else {
  //     %ordinal12, %workgroups12 = scf.if %cond1 {
  //       %ordinal1 = hal.executable.export.ordinal @1
  //       %workgroups1 = calculate for @1
  //       scf.yield %ordinal1, %workgroups1
  //     } else {
  //       %ordinal2 = hal.executable.export.ordinal @2
  //       %workgroups2 = calculate for @2
  //       scf.yield %ordinal2, %workgroups2
  //     }
  //     scf.yield %ordinal12, %workgroups12
  //   }
  std::tuple<Value, std::array<Value, 3>>
  selectExport(Location loc, IREE::HAL::ExecutableExportOp baseExportOp,
               Value device, ValueRange workload, OpBuilder &builder) const {
    if (!baseExportOp.getConditionBody()) {
      // No fallback - fast path to just the base export.
      Value ordinal = builder.create<IREE::HAL::ExecutableExportOrdinalOp>(
          loc, builder.getIndexType(), getExportRef(baseExportOp));
      auto workgroupCount =
          baseExportOp.calculateWorkgroupCount(loc, device, workload, builder);
      return {ordinal, workgroupCount};
    }
    // Recursively build the selection decision tree.
    auto fallbackExportOp =
        SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableExportOp>(
            baseExportOp, baseExportOp.getConditionFallbackAttr());
    return buildExportSelection(loc, baseExportOp, fallbackExportOp, device,
                                workload, builder);
  }
  std::tuple<Value, std::array<Value, 3>>
  buildExportSelection(Location loc, IREE::HAL::ExecutableExportOp tryExportOp,
                       IREE::HAL::ExecutableExportOp fallbackExportOp,
                       Value device, ValueRange workload,
                       OpBuilder &builder) const {
    // Inline the condition logic.
    Value tryCondition =
        tryExportOp.calculateCondition(loc, device, workload, builder);

    // Create an scf.if: the then region will simply return the
    // ordinal (condition matches) and the else region will contain the rest of
    // the decision tree.
    Type indexType = builder.getIndexType();
    auto ifOp = builder.create<scf::IfOp>(
        loc, TypeRange{indexType, indexType, indexType, indexType},
        tryCondition,
        /*addThenBlock=*/true, /*addElseBlock=*/true);
    {
      auto thenBuilder = ifOp.getThenBodyBuilder();
      Value tryOrdinal =
          thenBuilder.create<IREE::HAL::ExecutableExportOrdinalOp>(
              loc, thenBuilder.getIndexType(), getExportRef(tryExportOp));
      auto tryWorkgroupCount = tryExportOp.calculateWorkgroupCount(
          loc, device, workload, thenBuilder);
      thenBuilder.create<scf::YieldOp>(loc, ValueRange{
                                                tryOrdinal,
                                                tryWorkgroupCount[0],
                                                tryWorkgroupCount[1],
                                                tryWorkgroupCount[2],
                                            });
    }
    {
      auto elseBuilder = ifOp.getElseBodyBuilder();
      if (fallbackExportOp.getConditionBody()) {
        // Recursively chain to the next fallback-enabled export.
        auto chainExportOp =
            SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableExportOp>(
                fallbackExportOp, fallbackExportOp.getConditionFallbackAttr());
        auto [chainOrdinal, chainWorkgroupCount] =
            buildExportSelection(loc, fallbackExportOp, chainExportOp, device,
                                 workload, elseBuilder);
        elseBuilder.create<scf::YieldOp>(loc, ValueRange{
                                                  chainOrdinal,
                                                  chainWorkgroupCount[0],
                                                  chainWorkgroupCount[1],
                                                  chainWorkgroupCount[2],
                                              });
      } else {
        // Tail of recursion; fallback has no fallback.
        Value fallbackOrdinal =
            elseBuilder.create<IREE::HAL::ExecutableExportOrdinalOp>(
                loc, indexType, getExportRef(fallbackExportOp));
        auto fallbackWorkgroupCount = fallbackExportOp.calculateWorkgroupCount(
            loc, device, workload, elseBuilder);
        elseBuilder.create<scf::YieldOp>(loc, ValueRange{
                                                  fallbackOrdinal,
                                                  fallbackWorkgroupCount[0],
                                                  fallbackWorkgroupCount[1],
                                                  fallbackWorkgroupCount[2],
                                              });
      }
    }
    return {ifOp.getResult(0),
            {
                ifOp.getResult(1),
                ifOp.getResult(2),
                ifOp.getResult(3),
            }};
  }

  Operation *emitDispatchOp(
      Location loc, IREE::Stream::AffinityAttr affinityAttr, Value device,
      CommandBufferConversionMapping &commandBufferMapping,
      IREE::HAL::ExecutableExportOp exportOp, SymbolRefAttr entryPointAttr,
      IREE::Stream::CmdDispatchOp dispatchOp, OpAdaptor adaptor,
      OpBuilder &builder) const {
    Value executable = builder.create<IREE::HAL::ExecutableLookupOp>(
        loc, builder.getType<IREE::HAL::ExecutableType>(), device,
        entryPointAttr.getRootReference().getValue());

    // Select the export and calculate its workgroup count.
    auto [ordinal, workgroupCount] =
        selectExport(loc, exportOp, device, adaptor.getWorkload(), builder);

    auto layoutAttr = exportOp.getLayout();
    SmallVector<IREE::HAL::BindingValue> bindings;
    for (auto [i, bindingAttr] : llvm::enumerate(layoutAttr.getBindings())) {
      auto descriptorFlags = bindingAttr.getFlags();
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

    return builder.create<IREE::HAL::CommandBufferDispatchOp>(
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
    SmallVector<DictionaryAttr> oldArgAttrs;
    funcOp.getAllArgAttrs(oldArgAttrs);
    SmallVector<Type> newArgTypes;
    SmallVector<Attribute> newArgAttrs;
    newArgTypes.push_back(rewriter.getType<IREE::HAL::CommandBufferType>());
    newArgAttrs.push_back(rewriter.getDictionaryAttr({})); // command buffer
    for (auto [i, oldType] : llvm::enumerate(funcOp.getArgumentTypes())) {
      if (isa<IREE::Stream::ResourceType>(oldType)) {
        // Resource converted into a (binding ordinal, buffer) pair.
        newArgTypes.push_back(rewriter.getIndexType());
        newArgAttrs.push_back(rewriter.getDictionaryAttr({}));
        newArgTypes.push_back(rewriter.getType<IREE::HAL::BufferType>());
        newArgAttrs.push_back(oldArgAttrs[i]);
      } else {
        // Primitive/other pass-through.
        // Support expansion by preserving the arg attr on the first expanded
        // type and filling in empty attrs for the remainder.
        size_t oldCount = newArgTypes.size();
        if (failed(getTypeConverter()->convertType(oldType, newArgTypes))) {
          return rewriter.notifyMatchFailure(funcOp,
                                             "failed to convert arg types");
        }
        size_t typeCount = newArgTypes.size() - oldCount;
        newArgAttrs.push_back(oldArgAttrs[i]);
        newArgAttrs.append(typeCount - 1, rewriter.getDictionaryAttr({}));
      }
    }
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(funcOp.getResultTypes(),
                                                newResultTypes))) {
      return rewriter.notifyMatchFailure(funcOp,
                                         "failed to convert result types");
    }
    auto newOp = rewriter.replaceOpWithNewOp<IREE::Util::FuncOp>(
        funcOp, funcOp.getNameAttr(),
        rewriter.getFunctionType(newArgTypes, newResultTypes),
        /*tied_operands=*/ArrayAttr{}, funcOp.getSymVisibilityAttr(),
        rewriter.getArrayAttr(newArgAttrs), funcOp.getAllResultAttrs(),
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

    // Memoized dummy values.
    Value zeroIndex;
    auto getZeroIndex = [&]() {
      if (!zeroIndex) {
        zeroIndex = rewriter.create<arith::ConstantIndexOp>(callOp.getLoc(), 0);
      }
      return zeroIndex;
    };
    Value nullBuffer;
    auto getNullBuffer = [&]() {
      if (!nullBuffer) {
        nullBuffer = rewriter.create<IREE::Util::NullOp>(
            callOp.getLoc(), rewriter.getType<IREE::HAL::BufferType>());
      }
      return nullBuffer;
    };

    // Always pass the command buffer as the first arg.
    SmallVector<Value> operands;
    operands.push_back(commandBufferMapping.getHandle());
    size_t resourceIndex = 0;
    for (auto [originalOperand, convertedOperand] : llvm::zip_equal(
             callOp.getResourceOperands(), adaptor.getResourceOperands())) {
      if (llvm::isa<IREE::Stream::ResourceType>(originalOperand.getType())) {
        // Resource type, pass binding index or buffer and offset/length.
        auto binding = commandBufferMapping.resolveBinding(
            callOp.getLoc(), originalOperand, convertedOperand,
            adaptor.getResourceOperandOffsets()[resourceIndex],
            adaptor.getResourceOperandLengths()[resourceIndex], rewriter);
        if (binding.buffer.getType().isIndex()) {
          operands.push_back(binding.buffer);
          operands.push_back(getNullBuffer());
        } else {
          operands.push_back(getZeroIndex());
          operands.push_back(binding.buffer);
        }
        operands.push_back(binding.byteOffset);
        operands.push_back(binding.byteLength);
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
        /*tied_operands=*/ArrayAttr{}, callOp.getArgAttrsAttr(),
        callOp.getResAttrsAttr());
    return success();
  }
};

// Returns true if any primitive uniform value (i32, index, etc) captured within
// |op| (but not _by_ op) is a dynamic value (mutable global, calculated, etc).
// Returns false if all values are derived from constants or immutable globals.
static bool regionCapturesDynamicUniformValues(Operation *op) {
  auto isDynamicUniform = [](Value value) {
    if (value.getType().isIntOrIndexOrFloat()) {
      switch (IREE::HAL::categorizeValue(value)) {
      default:
      case IREE::HAL::ValueOrigin::Unknown:
      case IREE::HAL::ValueOrigin::MutableGlobal:
        return true;
      case IREE::HAL::ValueOrigin::LocalConstant:
      case IREE::HAL::ValueOrigin::ImmutableGlobal:
        return false;
      }
    }
    return false;
  };
  for (auto operand : op->getOperands()) {
    if (isDynamicUniform(operand)) {
      // Today this usually indicates a dynamic buffer size. We could perform
      // some tricks to adjust the size based on usage instead of requiring that
      // this size match however it's safer to treat dynamically sized buffers
      // as fully dynamic for now.
      return true;
    }
  }
  SetVector<Value> capturedValues;
  mlir::getUsedValuesDefinedAbove(op->getRegions(), capturedValues);
  for (auto capturedValue : capturedValues) {
    if (isDynamicUniform(capturedValue)) {
      return true;
    }
  }
  return false;
}

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

// Checks if |executeOp| contains only a single transfer operation and returns
// it. Non-transfer/dispatch operations like cache control will be ignored.
//
// Intended to match things like:
//   stream.cmd.execute ... {
//     stream.cmd.invalidate
//     stream.cmd.fill        <----- returned
//     stream.cmd.flush
//   }
// And not:
//   stream.cmd.execute ... {
//     stream.cmd.invalidate
//     stream.cmd.fill
//     stream.cmd.flush
//     stream.cmd.dispatch
//   }
static Operation *matchSingleTransferOp(IREE::Stream::CmdExecuteOp executeOp) {
  Operation *foundOp = nullptr;
  for (auto &block : executeOp.getBodyRegion()) {
    for (auto &op : block) {
      if (!TypeSwitch<Operation *, bool>(&op)
               // Ignore non-transfer/dispatch ops.
               .Case<IREE::Stream::CmdInvalidateOp, IREE::Stream::CmdFlushOp,
                     IREE::Stream::CmdDiscardOp, IREE::Stream::YieldOp>(
                   [&](auto metaOp) { return true; })
               .Case<IREE::Stream::CmdFillOp, IREE::Stream::CmdCopyOp>(
                   [&](auto transferOp) {
                     if (!foundOp) {
                       foundOp = &op; // first found
                       return true;
                     } else {
                       return false; // more than one transfer op
                     }
                   })
               // Dispatch/collective/etc fail the search.
               .Default([&](auto otherOp) { return false; })) {
        return nullptr;
      }
    }
  }
  return foundOp;
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

    // If the command buffer only contains a single transfer command we may be
    // able to convert it to a queue operation instead. This will have
    // significantly less overhead than a command buffer especially if we are
    // not able to memoize it.
    if (auto *singleTransferOp = matchSingleTransferOp(executeOp)) {
      // Gather wait/signal fence, which are optional.
      Value waitFence =
          getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
      Value signalFence = getOrCreateSignalFence(
          loc, device, executeOp.getResultTimepoint(), rewriter);

      // Replace the op with the queue operation.
      // Note that since we are matching an op nested within the region we have
      // to get the corresponding externally captured operand and lookup the
      // remapped value from the conversion state.
      //
      // Example:
      //   stream.cmd.execute ... with(%operand as %capture: !stream.resource)
      //     stream.cmd.fill ... %capture
      //  ->
      //   hal.device.queue.fill ... target(%operand : !hal.buffer)
      if (auto fillOp = dyn_cast<IREE::Stream::CmdFillOp>(*singleTransferOp)) {
        auto fillTargetBuffer = rewriter.getRemappedValue(
            executeOp.getClosureCapturedValue(fillOp.getTarget()));
        rewriter.create<IREE::HAL::DeviceQueueFillOp>(
            loc, device, queueAffinity, waitFence, signalFence,
            fillTargetBuffer, fillOp.getTargetOffset(),
            fillOp.getTargetLength(), fillOp.getValue(),
            IREE::HAL::FillFlagBitfield::None);
      } else if (auto copyOp =
                     dyn_cast<IREE::Stream::CmdCopyOp>(*singleTransferOp)) {
        auto copySourceBuffer = rewriter.getRemappedValue(
            executeOp.getClosureCapturedValue(copyOp.getSource()));
        auto copyTargetBuffer = rewriter.getRemappedValue(
            executeOp.getClosureCapturedValue(copyOp.getTarget()));
        rewriter.create<IREE::HAL::DeviceQueueCopyOp>(
            loc, device, queueAffinity, waitFence, signalFence,
            copySourceBuffer, copyOp.getSourceOffset(), copyTargetBuffer,
            copyOp.getTargetOffset(), copyOp.getLength(),
            IREE::HAL::CopyFlagBitfield::None);
      }

      rewriter.replaceOp(executeOp, signalFence);
      return success();
    }

    // Until uniform buffers are implemented we can't reuse command buffers that
    // contain non-constant uniform values (i32, index, etc). We'll have a pass
    // that runs prior to conversion that creates new stream resources and
    // changes dispatches to use them for any dispatch we can - note that there
    // may still be some that slip through due to custom executables.
    const bool capturesDynamicUniformValues =
        clForceIndirectCommandBuffers
            ? false
            : regionCapturesDynamicUniformValues(executeOp);

    // Calculate the indirect buffer references used within the command buffer
    // by analyzing captured resources. This analysis will be used by subsequent
    // conversion to decide between embedding the direct buffer references or
    // indirect ones. We only do this if the execution region is reused.
    IndexSet indexSet(loc, rewriter);
    BindingTable bindingTable;
    if (!executeOp.getOnce() && !capturesDynamicUniformValues &&
        clIndirectCommandBuffers) {
      bindingTable = BindingTable(executeOp, adaptor.getResourceOperands(),
                                  adaptor.getResourceOperandSizes(), indexSet);
    }

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
      bindingTable = {};
    }

    // Cache the binding table values for use with the indirect execute.
    auto bindingTableValues = llvm::to_vector(bindingTable.getValues());

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
    IREE::HAL::ExecuteFlagBitfield flags = IREE::HAL::ExecuteFlagBitfield::None;
    if (bindingTableValues.empty()) {
      rewriter.create<IREE::HAL::DeviceQueueExecuteOp>(
          loc, device, queueAffinity, waitFence, signalFence, commandBuffer,
          flags);
    } else {
      rewriter.create<IREE::HAL::DeviceQueueExecuteIndirectOp>(
          loc, device, queueAffinity, waitFence, signalFence, commandBuffer,
          bindingTableValues, flags);
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
    rewriter.replaceOpWithNewOp<IREE::HAL::DeviceQueueBarrierOp>(
        exportOp, device, queueAffinity,
        /*wait_fence=*/adaptor.getAwaitTimepoint(),
        /*signal_fence=*/externalValues[0],
        IREE::HAL::ExecuteFlagBitfield::None);
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
        IREE::HAL::FenceFlagBitfield::None, adaptor.getAwaitTimepoints());
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
        loc, rewriter.getI32Type(), timeoutMillis,
        IREE::HAL::WaitFlagBitfield::None, adaptor.getAwaitTimepoint());
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
        queueAffinity, IREE::HAL::ChannelFlagBitfield::None, id, group, rank,
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
        adaptor.getChannel(), color, key, IREE::HAL::ChannelFlagBitfield::None);
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
                  ResourceDeallocaOpPattern, ResourceRetainOpPattern,
                  ResourceReleaseOpPattern, ResourceIsTerminalOpPattern,
                  ResourceSizeOpPattern, ResourceTryMapOpPattern,
                  ResourceLoadOpPattern, ResourceStoreOpPattern,
                  ResourceSubviewOpPattern>(mapping, typeConverter, context);

  patterns.insert<FileConstantOpPattern, FileReadOpPattern, FileWriteOpPattern>(
      mapping, typeConverter, context);

  patterns.insert<TensorImportBufferOpPattern, TensorImportBufferViewOpPattern,
                  TensorExportBufferOpPattern, TensorExportBufferViewOpPattern,
                  TensorTraceOpPattern>(mapping, typeConverter, context);
  patterns
      .insert<CmdFlushOpPattern, CmdInvalidateOpPattern, CmdDiscardOpPattern,
              CmdFillOpPattern, CmdCopyOpPattern, CmdCollectiveOpPattern,
              CmdDispatchOpPattern, CmdFuncOpPattern, CmdCallOpPattern,
              CmdExecuteOpPattern, CmdSerialOpPattern, CmdConcurrentOpPattern>(
          mapping, typeConverter, context);
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
