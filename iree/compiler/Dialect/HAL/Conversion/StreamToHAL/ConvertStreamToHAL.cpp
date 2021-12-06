// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/StreamToHAL/ConvertStreamToHAL.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/DeviceSwitchBuilder.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

static Value makeElementType(Location loc, Type elementType,
                             OpBuilder &builder) {
  auto i32Value = IREE::HAL::getElementTypeValue(elementType);
  assert(i32Value.hasValue() && "unhandled element type for allocation");
  auto constantValue =
      builder.createOrFold<arith::ConstantIntOp>(loc, i32Value.getValue(), 32);
  return constantValue;
}

static Value makeEncodingType(Location loc, Attribute encodingType,
                              OpBuilder &builder) {
  auto i32Value = IREE::HAL::getEncodingTypeValue(encodingType);
  assert(i32Value.hasValue() && "unhandled encoding type for allocation");
  auto constantValue =
      builder.createOrFold<arith::ConstantIntOp>(loc, i32Value.getValue(), 32);
  return constantValue;
}

static Value lookupExecutableLayout(Location loc, Value device,
                                    IREE::HAL::InterfaceOp interfaceOp,
                                    OpBuilder &builder) {
  auto lookupOp = builder.create<IREE::HAL::ExecutableLayoutLookupOp>(
      loc, IREE::HAL::ExecutableLayoutType::get(loc.getContext()), device,
      interfaceOp.push_constantsAttr(),
      interfaceOp.getExecutableSetLayoutsAttr());
  return lookupOp.result();
}

static Value lookupDeviceFor(Operation *op, OpBuilder &builder) {
  // TODO(benvanik): make this do multi-device lookup and other fancy things.
  auto lookupOp = builder.create<IREE::HAL::ExSharedDeviceOp>(op->getLoc());
  return lookupOp.result();
}

static Value lookupAllocatorFor(Operation *op, OpBuilder &builder) {
  auto device = lookupDeviceFor(op, builder);
  auto allocatorOp =
      builder.create<IREE::HAL::DeviceAllocatorOp>(op->getLoc(), device);
  return allocatorOp.result();
}

// Scans all of the stream.cmd.* ops in the region to derive a command category.
static IREE::HAL::CommandCategoryBitfield deriveCommandCategories(
    Region &region) {
  auto bits = IREE::HAL::CommandCategoryBitfield::None;
  for (auto &block : region) {
    for (auto &op : block) {
      if (isa<IREE::Stream::CmdDispatchOp>(op)) {
        bits = bits | IREE::HAL::CommandCategoryBitfield::Dispatch;
      } else {
        bits = bits | IREE::HAL::CommandCategoryBitfield::Transfer;
      }
      for (auto &nestedRegion : op.getRegions()) {
        bits = bits | deriveCommandCategories(nestedRegion);
      }
    }
  }
  return bits;
}

// Maps a resource type to the corresponding HAL memory types and buffer usage.
// This will fail if the resource type is not directly mappable to HAL bits.
// The bits set here are those that must be set for the buffer to be used as the
// buffer within the program with its defined resource lifetime.
static LogicalResult deriveRequiredResourceBufferBits(
    Location loc, IREE::Stream::ResourceType resourceType,
    IREE::HAL::MemoryTypeBitfield &memoryTypes,
    IREE::HAL::BufferUsageBitfield &bufferUsage) {
  memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
  bufferUsage = IREE::HAL::BufferUsageBitfield::None;
  switch (resourceType.getLifetime()) {
    default:
      return mlir::emitError(loc)
             << "unsupported resource lifetime: "
             << IREE::Stream::stringifyLifetime(resourceType.getLifetime());
    case IREE::Stream::Lifetime::Constant:
      // Device local; copies required to get into external resources.
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
      bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Constant;
      break;
    case IREE::Stream::Lifetime::Variable:
      // Device local; copies required to get into external resources.
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
      break;
    case IREE::Stream::Lifetime::External:
      // We only require device-visible for external buffers (as we don't today
      // do anything else with them on the host). They may be mappable for user
      // convenience. Ideally they would have been placed in device-local memory
      // but so long as they are device visible the program will execute
      // correctly.
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceVisible;
      break;
    case IREE::Stream::Lifetime::Staging:
      // Host local; copies required to get into device resources.
      // We could vary this based on staging usage (upload/download) by
      // making it device-local|host-visible, but host-local means we have
      // a better chance of mapping it during uploads.
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::HostLocal |
                    IREE::HAL::MemoryTypeBitfield::DeviceVisible;
      bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                    IREE::HAL::BufferUsageBitfield::Mapping;
      break;
    case IREE::Stream::Lifetime::Transient:
      // Device local; copies required to get into external resources.
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal |
                    IREE::HAL::MemoryTypeBitfield::Transient;
      break;
  }

  // TODO(benvanik): refine usage based on analysis.
  bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                IREE::HAL::BufferUsageBitfield::Dispatch;

  return success();
}

// Maps a resource type to the corresponding HAL memory types and buffer usage.
// This will fail if the resource type is not directly mappable to HAL bits.
// The bits set here represent the superset of required and allowed bits and
// are useful for providing buffers back to users via the ABI that may need to
// be used for more than just what the internal program requires.
static LogicalResult deriveAllowedResourceBufferBits(
    Location loc, IREE::Stream::ResourceType resourceType,
    IREE::HAL::MemoryTypeBitfield &memoryTypes,
    IREE::HAL::BufferUsageBitfield &bufferUsage) {
  memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
  bufferUsage = IREE::HAL::BufferUsageBitfield::None;
  if (failed(deriveRequiredResourceBufferBits(loc, resourceType, memoryTypes,
                                              bufferUsage))) {
    return failure();
  }
  switch (resourceType.getLifetime()) {
    default:
      break;
    case IREE::Stream::Lifetime::External:
      // #yolo; these come from/go to outside the program.
      // Today we assume they are device-local|host-visible just for
      // practical purposes but that does not have to be true. We really
      // want this to be something we analyze and handle on the edges
      // (transfering devices/etc if needed).
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal |
                    IREE::HAL::MemoryTypeBitfield::HostVisible;
      // NOTE: we may not map it but users may after they get them back.
      // Another reason we should annotate this - having a buffer be
      // mappable is potentially expensive (may get a 2nd copy in memory!).
      bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Mapping;
      break;
  }
  return success();
}

class StreamConversionMapping {
 public:
  // Maps the stream dialect |executeOp| to the hal dialect |commandBuffer|
  // value used during recording. Patterns can use this to find the SSA value
  // they need to make hal.command_buffer.* ops.
  void mapCommandBuffer(IREE::Stream::CmdExecuteOp executeOp,
                        Value commandBuffer) {
    assert(commandBuffers.insert(std::make_pair(executeOp, commandBuffer))
               .second &&
           "multiple command buffers cannot be registered for the same op");

    // Map all ops nested within the command buffer so we can query later.
    executeOp.walk([&](Operation *op) {
      commandBuffers.insert(std::make_pair(op, commandBuffer));
      return WalkResult::advance();
    });
  }

  // Looks up a mapped command buffer SSA value that can be used by the given
  // stream.cmd.* op.
  Value lookupCommandBufferFor(Operation *cmdOp) const {
    auto it = commandBuffers.find(cmdOp);
    assert(it != commandBuffers.end() &&
           "command buffer must have been registered during conversion");
    return it->second;
  }

 private:
  // Ops within stream.cmd.execute ops -> !hal.command_buffer.
  DenseMap<Operation *, Value> commandBuffers;
};

template <typename OpT>
struct StreamConversionPattern : public OpConversionPattern<OpT> {
  StreamConversionPattern(std::shared_ptr<StreamConversionMapping> mapping,
                          TypeConverter &typeConverter, MLIRContext *context,
                          PatternBenefit benefit = 1)
      : OpConversionPattern<OpT>(typeConverter, context, benefit),
        mapping(std::move(mapping)) {}

  std::shared_ptr<StreamConversionMapping> mapping;
};

struct ResourceAllocOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceAllocOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceAllocOp allocOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto allocator = lookupAllocatorFor(allocOp, rewriter);
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();

    SmallVector<Value> results;
    for (auto it : llvm::zip(allocOp.results(), allocOp.storage_sizes())) {
      auto resourceResult = std::get<0>(it);
      auto resourceType =
          resourceResult.getType().cast<IREE::Stream::ResourceType>();
      auto storageSize = std::get<1>(it);

      auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
      auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
      if (failed(deriveAllowedResourceBufferBits(allocOp.getLoc(), resourceType,
                                                 memoryTypes, bufferUsage))) {
        return failure();
      }

      auto allocateOp = rewriter.create<IREE::HAL::AllocatorAllocateOp>(
          allocOp.getLoc(), bufferType, allocator, memoryTypes, bufferUsage,
          storageSize);
      results.push_back(allocateOp.result());
    }

    rewriter.replaceOp(allocOp, results);
    return success();
  }
};

struct ResourceAllocaOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceAllocaOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceAllocaOp allocaOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto allocator = lookupAllocatorFor(allocaOp, rewriter);
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();

    // Transient allocations are device-local. Copies are required to get their
    // contents back on the host/another device.
    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::DeviceLocal |
                       IREE::HAL::MemoryTypeBitfield::Transient;

    // TODO(benvanik): refine usage.
    // We know by construction that transient buffers are not host visible and
    // as such can only be used for device commands. We should be able to more
    // closely limit to just dispatch or transfer though.
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::Dispatch |
                       IREE::HAL::BufferUsageBitfield::Transfer;

    auto allocateOp = rewriter.create<IREE::HAL::AllocatorAllocateOp>(
        allocaOp.getLoc(), bufferType, allocator, memoryTypes, bufferUsage,
        allocaOp.storage_size());

    // TODO(benvanik): stream ordered allocations.
    auto resolvedTimepoint =
        rewriter.create<arith::ConstantIndexOp>(allocaOp.getLoc(), 0)
            .getResult();

    rewriter.replaceOp(allocaOp, {allocateOp.result(), resolvedTimepoint});
    return success();
  }
};

struct ResourceDeallocaOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceDeallocaOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceDeallocaOp deallocaOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): stream ordered allocations.
    auto resolvedTimepoint =
        rewriter.create<arith::ConstantIndexOp>(deallocaOp.getLoc(), 0)
            .getResult();
    rewriter.replaceOp(deallocaOp, {resolvedTimepoint});
    return success();
  }
};

struct ResourceSizeOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceSizeOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceSizeOp sizeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferLengthOp>(
        sizeOp, rewriter.getIndexType(), adaptor.operand());
    return success();
  }
};

struct ResourceMapOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceMapOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceMapOp mapOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto allocator = lookupAllocatorFor(mapOp, rewriter);
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();

    // We know this is a staging buffer. We could refine usage here by seeing
    // whether this was upload or download.
    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::HostLocal |
                       IREE::HAL::MemoryTypeBitfield::DeviceVisible;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::Mapping |
                       IREE::HAL::BufferUsageBitfield::Transfer;

    rewriter.replaceOpWithNewOp<IREE::HAL::AllocatorMapOp>(
        mapOp, bufferType, allocator, memoryTypes, bufferUsage,
        adaptor.source(), adaptor.source_offset(), adaptor.result_size());
    return success();
  }
};

struct ResourceTryMapOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceTryMapOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceTryMapOp tryMapOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto allocator = lookupAllocatorFor(tryMapOp, rewriter);
    auto resourceType =
        tryMapOp.result().getType().cast<IREE::Stream::ResourceType>();
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
        bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Constant;
        break;
      case IREE::Stream::Lifetime::Staging:
        // Host local; copies required to get into device resources.
        // We could vary this based on staging usage (upload/download) by
        // making it device-local|host-visible, but host-local means we have
        // a better chance of mapping it during uploads.
        memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::HostLocal |
                      IREE::HAL::MemoryTypeBitfield::DeviceVisible;
        bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                      IREE::HAL::BufferUsageBitfield::Mapping;
        break;
    }

    // TODO(benvanik): refine usage based on analysis.
    bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                  IREE::HAL::BufferUsageBitfield::Dispatch;

    rewriter.replaceOpWithNewOp<IREE::HAL::AllocatorTryMapOp>(
        tryMapOp, rewriter.getI1Type(), bufferType, allocator, memoryTypes,
        bufferUsage, adaptor.source(), adaptor.source_offset(),
        adaptor.result_size());
    return success();
  }
};

struct ResourceLoadOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceLoadOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceLoadOp loadOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loadType = getTypeConverter()->convertType(loadOp.result().getType());
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferLoadOp>(
        loadOp, loadType, adaptor.source(), adaptor.source_offset());
    return success();
  }
};

struct ResourceStoreOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceStoreOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceStoreOp storeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.create<IREE::HAL::BufferStoreOp>(storeOp.getLoc(), adaptor.value(),
                                              adaptor.target(),
                                              adaptor.target_offset());
    rewriter.replaceOp(storeOp, adaptor.target());
    return success();
  }
};

struct ResourceSubviewOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceSubviewOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceSubviewOp subviewOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();
    // NOTE: this aliases! We assume at this point all useful alias analysis
    // has been performed and it's fine to lose the tie information here.
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferSubspanOp>(
        subviewOp, bufferType, adaptor.source(), adaptor.source_offset(),
        adaptor.result_size());
    return success();
  }
};

// Inserts IR to assert that the underlying buffer storage is compatible with
// the intended usage in the program. The allocator used to allocate the
// buffer must have compatibility with our target device allocator and the
// buffer must have at least the minimum expected size (additional padding is
// ok).
static LogicalResult buildStorageAssertions(
    Location loc, Value buffer, StringAttr message, Value allocator,
    Value minimumLength, IREE::Stream::ResourceType resourceType,
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
  LogicalResult matchAndRewrite(
      IREE::Stream::TensorImportOp importOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!importOp.source().getType().isa<IREE::HAL::BufferType>()) {
      return failure();
    }

    // TODO(benvanik): get a name for the tensor (argument name/etc).
    auto message = rewriter.getStringAttr("tensor");

    // Directly use the buffer.
    auto buffer = adaptor.source();
    rewriter.replaceOp(importOp, buffer);

    // Assert the storage is compatible with our expected device and usage.
    auto targetAllocator = lookupAllocatorFor(importOp, rewriter);
    auto resourceType =
        importOp.result().getType().cast<IREE::Stream::ResourceType>();
    if (failed(buildStorageAssertions(
            importOp.getLoc(), adaptor.source(), message, targetAllocator,
            adaptor.result_size(), resourceType, rewriter))) {
      return failure();
    }

    return success();
  }
};

struct TensorImportBufferViewOpPattern
    : public StreamConversionPattern<IREE::Stream::TensorImportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TensorImportOp importOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceType = importOp.source().getType();
    if (!sourceType.isa<IREE::HAL::BufferViewType>() &&
        !sourceType.isa<TensorType>()) {
      return failure();
    }

    auto loc = importOp.getLoc();

    // TODO(benvanik): get a name for the tensor (argument name/etc).
    auto message = rewriter.getStringAttr("tensor");

    auto bufferView = adaptor.source();
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();
    auto bufferOp = rewriter.replaceOpWithNewOp<IREE::HAL::BufferViewBufferOp>(
        importOp, bufferType, bufferView);

    // Assert the storage is compatible with our expected device and usage.
    auto targetAllocator = lookupAllocatorFor(importOp, rewriter);
    auto resourceType =
        importOp.result().getType().cast<IREE::Stream::ResourceType>();
    if (failed(buildStorageAssertions(loc, bufferOp.result(), message,
                                      targetAllocator, adaptor.result_size(),
                                      resourceType, rewriter))) {
      return failure();
    }

    return success();
  }
};

struct TensorExportBufferOpPattern
    : public StreamConversionPattern<IREE::Stream::TensorExportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TensorExportOp exportOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!exportOp.result().getType().isa<IREE::HAL::BufferType>()) {
      return failure();
    }
    rewriter.replaceOp(exportOp, adaptor.source());
    return success();
  }
};

struct TensorExportBufferViewOpPattern
    : public StreamConversionPattern<IREE::Stream::TensorExportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TensorExportOp exportOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto targetType = exportOp.result().getType();
    if (!targetType.isa<IREE::HAL::BufferViewType>() &&
        !targetType.isa<TensorType>()) {
      return failure();
    }

    auto loc = exportOp.getLoc();
    auto tensorType =
        adaptor.source_encoding().getValue().cast<RankedTensorType>();
    auto dynamicDims = adaptor.source_encoding_dims();

    // NOTE: we should have verified supported encodings/types at entry into the
    // HAL pipeline.
    auto encodingType =
        IREE::HAL::getEncodingTypeValue(tensorType.getEncoding());
    assert(encodingType.hasValue() && "invalid tensor encoding");
    auto elementType =
        IREE::HAL::getElementTypeValue(tensorType.getElementType());
    assert(elementType.hasValue() && "invalid tensor element type");

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
        exportOp, adaptor.source(), elementType.getValue(),
        encodingType.getValue(), dims);
    return success();
  }
};

struct TensorTraceOpPattern
    : public StreamConversionPattern<IREE::Stream::TensorTraceOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TensorTraceOp traceOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferViewTraceOp>(
        traceOp, traceOp.keyAttr(), adaptor.operands());
    return success();
  }
};

struct CmdFlushOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdFlushOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdFlushOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): HAL command buffer op for flush.
    rewriter.eraseOp(op);
    return success();
  }
};

struct CmdInvalidateOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdInvalidateOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdInvalidateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): HAL command buffer op for invalidate.
    rewriter.eraseOp(op);
    return success();
  }
};

struct CmdDiscardOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdDiscardOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdDiscardOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): HAL command buffer op for discard.
    rewriter.eraseOp(op);
    return success();
  }
};

struct CmdFillOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdFillOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdFillOp fillOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto commandBuffer = mapping->lookupCommandBufferFor(fillOp);
    rewriter.replaceOpWithNewOp<IREE::HAL::CommandBufferFillBufferOp>(
        fillOp, commandBuffer, adaptor.target(), adaptor.target_offset(),
        adaptor.target_length(), adaptor.value());
    return success();
  }
};

struct CmdCopyOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdCopyOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdCopyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto commandBuffer = mapping->lookupCommandBufferFor(op);
    rewriter.replaceOpWithNewOp<IREE::HAL::CommandBufferCopyBufferOp>(
        op, commandBuffer, adaptor.source(), adaptor.source_offset(),
        adaptor.target(), adaptor.target_offset(), adaptor.length());
    return success();
  }
};

struct CmdDispatchOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdDispatchOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdDispatchOp dispatchOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = dispatchOp.getLoc();
    auto commandBuffer = mapping->lookupCommandBufferFor(dispatchOp);

    // Get the device handle we're executing against in this execution region.
    // Note that this is a dynamic value: we have to treat the device as unknown
    // here.
    auto device = rewriter.create<IREE::HAL::CommandBufferDeviceOp>(
        loc, rewriter.getType<IREE::HAL::DeviceType>(), commandBuffer);

    // Get the handle to the executable that is compatible with our device.
    auto executableOp =
        cast<IREE::HAL::ExecutableOp>(SymbolTable::lookupNearestSymbolFrom(
            dispatchOp, dispatchOp.entry_point().getRootReference()));
    assert(executableOp && "dispatch target executable op not found");

    // Ask each target backend to record their dispatch logic.
    IREE::HAL::DeviceSwitchRewriter switchRewriter(loc,
                                                   /*resultTypes=*/TypeRange{},
                                                   device, rewriter);
    for (auto variantOp :
         executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
      auto entryPointOps =
          variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>();
      auto entryPointIt = llvm::find_if(
          entryPointOps, [&](IREE::HAL::ExecutableEntryPointOp op) {
            return op.getNameAttr() ==
                   dispatchOp.entry_point().getLeafReference();
          });
      if (entryPointIt == entryPointOps.end()) {
        return variantOp.emitError()
               << "hal.executable.variant is missing the flow entry point for "
               << dispatchOp.entry_point();
      }
      auto entryPointOp = *entryPointIt;
      auto interfaceOp =
          dyn_cast<IREE::HAL::InterfaceOp>(SymbolTable::lookupSymbolIn(
              executableOp, entryPointOp.interfaceAttr()));

      auto *region = switchRewriter.addConditionRegion(
          variantOp.target().getMatchExpression());
      auto &entryBlock = region->front();
      auto caseBuilder = OpBuilder::atBlockBegin(&entryBlock);

      // Record push constants and buffer bindings.
      recordParameters(loc, device, commandBuffer, dispatchOp, adaptor,
                       interfaceOp, caseBuilder);

      // Dispatch with a target-specific workgroup count.
      auto entryPointSymRef =
          SymbolRefAttr::get(caseBuilder.getContext(), executableOp.getName(),
                             {SymbolRefAttr::get(entryPointOp->getParentOp()),
                              SymbolRefAttr::get(entryPointOp)});
      auto caseWorkgroupCount = calculateDispatchWorkgroupCount(
          loc, executableOp, entryPointOp, adaptor.workgroup_count(),
          caseBuilder);
      caseBuilder.create<IREE::HAL::CommandBufferDispatchSymbolOp>(
          loc, commandBuffer, entryPointSymRef, caseWorkgroupCount[0],
          caseWorkgroupCount[1], caseWorkgroupCount[2]);

      caseBuilder.create<IREE::HAL::ReturnOp>(loc);
    }
    switchRewriter.build();

    rewriter.eraseOp(dispatchOp);
    return success();
  }

  void recordParameters(Location loc, Value device, Value commandBuffer,
                        IREE::Stream::CmdDispatchOp dispatchOp,
                        OpAdaptor adaptor, IREE::HAL::InterfaceOp interfaceOp,
                        OpBuilder &builder) const {
    auto executableLayout =
        lookupExecutableLayout(loc, device, interfaceOp, builder);

    // Push constant values.
    // TODO(#5322): symbolic push constant names on the hal.interface so we can
    // sparsely pack these.
    if (!adaptor.operands().empty()) {
      int pushConstantBase = 0;  // always 0 today
      SmallVector<Value> pushConstants;
      for (auto operand : adaptor.operands()) {
        // Need an explicit index cast to i32 since the
        // CommandBufferPushConstantsOp is intrinsically i32 based.
        // TODO(benvanik): don't force conversion yet - or do so
        // target-dependently.
        if (operand.getType().isa<IndexType>()) {
          pushConstants.push_back(builder.create<arith::IndexCastOp>(
              dispatchOp.getLoc(), builder.getIntegerType(32), operand));
        } else {
          assert(
              (operand.getType().isInteger(32) || operand.getType().isF32()) &&
              "expected a 32-bit value");
          pushConstants.push_back(operand);
        }
      }
      builder.create<IREE::HAL::CommandBufferPushConstantsOp>(
          loc, commandBuffer, executableLayout,
          builder.getIndexAttr(pushConstantBase), pushConstants);
    }

    // TODO(benvanik): typed accessors for bindings.
    auto bindingSymbols = dispatchOp->getAttr("hal.interface.bindings")
                              .dyn_cast_or_null<ArrayAttr>();
    assert(bindingSymbols &&
           "interface materialization must annotate dispatch sites");
    auto bindingOps = llvm::to_vector<
        4>(llvm::map_range(bindingSymbols, [&](Attribute symRefAttr) {
      auto bindingOp =
          SymbolTable::lookupNearestSymbolFrom<IREE::HAL::InterfaceBindingOp>(
              dispatchOp, symRefAttr.cast<SymbolRefAttr>());
      assert(bindingOp && "binding not found");
      return bindingOp;
    }));
    // Sort in set -> binding order ascending.
    llvm::sort(bindingOps, [](IREE::HAL::InterfaceBindingOp lhs,
                              IREE::HAL::InterfaceBindingOp rhs) {
      int64_t lhsSet = lhs.set().getSExtValue();
      int64_t rhsSet = rhs.set().getSExtValue();
      if (lhsSet < rhsSet) return true;
      if (rhsSet > lhsSet) return false;
      int64_t lhsBinding = lhs.binding().getSExtValue();
      int64_t rhsBinding = rhs.binding().getSExtValue();
      return lhsBinding < rhsBinding;
    });

    // Push descriptor bindings.
    int64_t currentSet = -1;
    SmallVector<IREE::HAL::DescriptorSetBindingValue> bindings;
    auto flushSet = [&]() {
      builder.create<IREE::HAL::CommandBufferPushDescriptorSetOp>(
          loc, commandBuffer, executableLayout, currentSet, bindings);
      bindings.clear();
    };
    for (unsigned i = 0; i < adaptor.resources().size(); ++i) {
      auto bindingOp = bindingOps[i];
      int64_t set = bindingOp.set().getSExtValue();
      if (currentSet != -1 && currentSet != set) flushSet();
      currentSet = set;
      IREE::HAL::DescriptorSetBindingValue binding;
      binding.ordinal = builder.create<arith::ConstantIndexOp>(
          loc, bindingOp.binding().getSExtValue());
      binding.buffer = adaptor.resources()[i];
      binding.byteOffset = adaptor.resource_offsets()[i];
      binding.byteLength = adaptor.resource_lengths()[i];
      bindings.push_back(binding);
    }
    if (currentSet != -1) flushSet();
  }

  // Calculates the workgroup count (x, y, z) for dispatching to the given
  // |entryPointOp|. The provided N-dimensional |workload| is the total number
  // of invocations required as calculated by the generic workload logic
  // (basically, number of output elements in tensors).
  static std::array<Value, 3> calculateDispatchWorkgroupCount(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, ValueRange workload,
      OpBuilder &builder) {
    Region *region = entryPointOp.getBody();
    if (region) {
      return calculateDispatchWorkgroupCountFromRegion(loc, entryPointOp,
                                                       workload, builder);
    }
    auto workgroupSize = calculateDispatchWorkgroupSize(
        loc, executableOp, entryPointOp, workload, builder);
    return calculateWorkloadWorkgroupCount(loc, workload, workgroupSize,
                                           builder);
  }

  // Calculates the workgroup size (x, y, z). These are the dimension numbers
  // for a single workgroup.
  static std::array<Value, 3> calculateDispatchWorkgroupSize(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, ValueRange workload,
      OpBuilder &builder) {
    // When no workgroup size is specified we just assume [1,1,1].
    // This yields a workgroup count that models the extents of the workload.
    return {
        builder.createOrFold<arith::ConstantIndexOp>(loc, 1),
        builder.createOrFold<arith::ConstantIndexOp>(loc, 1),
        builder.createOrFold<arith::ConstantIndexOp>(loc, 1),
    };
  }

  static std::array<Value, 3> calculateDispatchWorkgroupCountFromRegion(
      Location loc, IREE::HAL::ExecutableEntryPointOp entryPointOp,
      ValueRange workload, OpBuilder &builder) {
    // TODO(benvanik): replace with region inlining util.
    Block *body = entryPointOp.getBlock();
    BlockAndValueMapping bvm;
    for (auto args : llvm::enumerate(workload)) {
      bvm.map(body->getArgument(args.index()), args.value());
    }
    for (Operation &op : body->without_terminator()) {
      builder.clone(op, bvm);
    }
    auto returnOp = cast<IREE::HAL::ReturnOp>(body->getTerminator());
    return {
        bvm.lookup(returnOp.operands()[0]),
        bvm.lookup(returnOp.operands()[1]),
        bvm.lookup(returnOp.operands()[2]),
    };
  }

  // Calculates the workgroup count (x, y, z) given the total N-dimensional
  // |workload| and specific |workgroupSize|.
  static std::array<Value, 3> calculateWorkloadWorkgroupCount(
      Location loc, ValueRange workload,
      const std::array<Value, 3> &workgroupSize, OpBuilder &builder) {
    std::array<Value, 3> result;

    auto constantOne = builder.createOrFold<arith::ConstantIndexOp>(loc, 1);
    if (workload.size() <= 3) {
      // 1-D to 3-D are easy (pad 2 to 0 dimensions) and divide by workgroup
      // size.
      for (int i = 0; i < 3; ++i) {
        // Round up: (workload[i] + workgroup_size - 1) / workgroup_size;
        Value workloadI = i < workload.size() ? workload[i] : constantOne;
        workloadI = builder.createOrFold<arith::SubIOp>(
            loc,
            builder.createOrFold<arith::AddIOp>(loc, workloadI,
                                                workgroupSize[i]),
            constantOne);
        result[i] = builder.createOrFold<arith::DivUIOp>(loc, workloadI,
                                                         workgroupSize[i]);
      }
    } else {
      // TODO(#4140): remapping of N-D to 3-D: this is not how you do this!
      Value flatWorkload = constantOne;
      for (auto workloadI : workload) {
        flatWorkload =
            builder.createOrFold<arith::MulIOp>(loc, flatWorkload, workloadI);
      }
      for (int i = 0; i < 3; ++i) {
        // Round up: (workload[i] + workgroup_size - 1) / workgroup_size;
        auto rounded = builder.createOrFold<arith::SubIOp>(
            loc,
            builder.createOrFold<arith::AddIOp>(loc, flatWorkload,
                                                workgroupSize[i]),
            constantOne);
        auto workgroupCountI = builder.createOrFold<arith::DivUIOp>(
            loc, rounded, workgroupSize[i]);
        result[i] = workgroupCountI;

        // Multiply back out and subtract from invocations.
        flatWorkload = builder.createOrFold<arith::SubIOp>(
            loc, flatWorkload,
            builder.createOrFold<arith::MulIOp>(loc, workgroupCountI, rounded));
      }
    }

    return result;
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
  for (auto &op : block) serialOps.push_back(&op);
  for (auto *op : serialOps) {
    if (op->hasTrait<OpTrait::IsTerminator>()) continue;
    builder.setInsertionPointAfter(op);
    builder.create<IREE::HAL::CommandBufferExecutionBarrierOp>(
        loc, commandBuffer, sourceStage, targetStage, flags);
  }
}

struct CmdExecuteOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdExecuteOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdExecuteOp executeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = executeOp.getLoc();
    auto device = lookupDeviceFor(executeOp, rewriter);

    // TODO(benvanik): disable inline execution once we have semaphores.
    // We can look ahead to see if there's an await immediately to trigger the
    // inline execution.
    auto modes = IREE::HAL::CommandBufferModeBitfield::OneShot |
                 IREE::HAL::CommandBufferModeBitfield::AllowInlineExecution;

    // Derive the command buffer type based on the kind of operations present.
    // This can help the submission get routed to appropriate hardware queues
    // (like dedicated DMA controllers).
    auto commandCategories = deriveCommandCategories(executeOp.body());

    // Create a new command buffer for recording. If we were
    auto commandBuffer =
        rewriter
            .create<IREE::HAL::CommandBufferCreateOp>(
                loc, rewriter.getType<IREE::HAL::CommandBufferType>(), device,
                modes, commandCategories)
            .result();
    mapping->mapCommandBuffer(executeOp, commandBuffer);

    // Run through the execution region and serialize execution by inserting
    // barriers. Nested regions may elide barriers as needed.
    auto &bodyBlock = executeOp.body().front();
    insertSerializationBarriers(loc, bodyBlock, commandBuffer,
                                OpBuilder::atBlockBegin(&bodyBlock));

    // Begin/end recording and inline the execution region between them.
    rewriter.create<IREE::HAL::CommandBufferBeginOp>(loc, commandBuffer);
    auto endOp =
        rewriter.create<IREE::HAL::CommandBufferEndOp>(loc, commandBuffer);
    rewriter.mergeBlockBefore(&executeOp.body().front(), endOp,
                              adaptor.operands());

    // TODO(benvanik): we should queue a submit here with the semaphore instead.
    rewriter.create<IREE::HAL::ExSubmitAndWaitOp>(loc, device, commandBuffer);

    // TODO(benvanik): propagate semaphore information.
    auto resolvedTimepoint =
        rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();

    rewriter.replaceOp(executeOp, resolvedTimepoint);
    return success();
  }
};

struct CmdSerialOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdSerialOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdSerialOp serialOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto commandBuffer = mapping->lookupCommandBufferFor(serialOp);

    // Run through the execution region and serialize execution by inserting
    // barriers. Nested regions may elide barriers as needed.
    auto &bodyBlock = serialOp.body().front();
    insertSerializationBarriers(serialOp.getLoc(), bodyBlock, commandBuffer,
                                OpBuilder::atBlockBegin(&bodyBlock));

    // Inline the serial execution region.
    rewriter.mergeBlockBefore(&serialOp.body().front(), serialOp);
    rewriter.eraseOp(serialOp);
    return success();
  }
};

struct CmdConcurrentOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdConcurrentOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdConcurrentOp concurrentOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Inline the concurrent execution region.
    // TODO(benvanik): split barriers (event set/wait) when nesting.
    rewriter.mergeBlockBefore(&concurrentOp.body().front(), concurrentOp);
    rewriter.eraseOp(concurrentOp);
    return success();
  }
};

struct TimepointImmediateOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointImmediateOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointImmediateOp immediateOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): model timepoints as semaphores.
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(immediateOp, 0);
    return success();
  }
};

struct TimepointImportOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointImportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointImportOp importOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only handle imports from HAL semaphores.
    auto operands = adaptor.operands();
    if (operands.size() != 2 ||
        !operands[0].getType().isa<IREE::HAL::SemaphoreType>() ||
        !operands[1].getType().isIntOrIndex()) {
      return rewriter.notifyMatchFailure(importOp,
                                         "only imports from HAL semaphore + "
                                         "sequence value tuples are supported");
    }

    // TODO(benvanik): model timepoints as semaphores.
    // For now we just block on the semaphore.
    auto awaitOp = rewriter.create<IREE::HAL::SemaphoreAwaitOp>(
        importOp.getLoc(), rewriter.getI32Type(), operands[0], operands[1]);
    rewriter.create<IREE::Util::StatusCheckOkOp>(
        importOp.getLoc(), awaitOp.status(),
        "failed to wait on imported semaphore");
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(importOp, 0);
    return success();
  }
};

struct TimepointExportOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointExportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointExportOp exportOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only handle exports into HAL semaphores.
    if (exportOp.getNumResults() != 2 ||
        !exportOp.getResult(0).getType().isa<IREE::HAL::SemaphoreType>() ||
        !exportOp.getResult(1).getType().isIntOrIndex()) {
      return rewriter.notifyMatchFailure(exportOp,
                                         "only exports to HAL semaphore + "
                                         "sequence value tuples are supported");
    }

    auto loc = exportOp.getLoc();
    auto device = lookupDeviceFor(exportOp, rewriter);

    // TODO(benvanik): model timepoints as semaphores.
    // For now we just create a signaled semaphore.
    auto exportValue = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto exportSemaphore = rewriter.create<IREE::HAL::SemaphoreCreateOp>(
        loc, rewriter.getType<IREE::HAL::SemaphoreType>(), device, exportValue);
    rewriter.replaceOp(exportOp, {exportSemaphore, exportValue});
    return success();
  }
};

struct TimepointJoinOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointJoinOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointJoinOp joinOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): model timepoints as semaphores.
    // This should be a max() of the operand timepoints. Could be done with
    // affine expressions, but since everything is always 0 we just max(0,0)=0
    // here :)
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(joinOp, 0);
    return success();
  }
};

struct TimepointAwaitOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointAwaitOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointAwaitOp awaitOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): model timepoints as semaphores.
    rewriter.replaceOp(awaitOp, adaptor.operands());
    return success();
  }
};

struct ElideYieldOpPattern
    : public StreamConversionPattern<IREE::Stream::YieldOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::YieldOp yieldOp, OpAdaptor adaptor,
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
  LogicalResult matchAndRewrite(
      IREE::Util::GlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto initialValue = op.initial_value();
    if (!initialValue.hasValue()) return failure();
    if (!initialValue->isa<IREE::Stream::TimepointAttr>()) return failure();
    rewriter.updateRootInPlace(
        op, [&]() { op.initial_valueAttr(rewriter.getIndexAttr(0)); });
    return success();
  }
};

}  // namespace

void populateStreamToHALPatterns(MLIRContext *context,
                                 ConversionTarget &conversionTarget,
                                 TypeConverter &typeConverter,
                                 OwningRewritePatternList &patterns) {
  conversionTarget.addIllegalDialect<IREE::Stream::StreamDialect>();

  typeConverter.addConversion(
      [=](IREE::Stream::ResourceType type, SmallVectorImpl<Type> &results) {
        // Resources are just buffers (no shape/encoding/etc).
        results.push_back(IREE::HAL::BufferType::get(context));
        return success();
      });

  typeConverter.addConversion(
      [=](IREE::Stream::TimepointType type, SmallVectorImpl<Type> &results) {
        // TODO(benvanik): model timepoints as semaphores.
        // This may become a !hal.semaphore + index, or some !hal.timepoint that
        // we then do more analysis on once we know what devices are in use
        // where.
        results.push_back(IndexType::get(context));
        return success();
      });

  // Spooky action at a distance:
  patterns.insert<GlobalTimepointConversionPattern>(typeConverter, context);

  auto mapping = std::make_shared<StreamConversionMapping>();
  patterns.insert<ResourceAllocOpPattern, ResourceAllocaOpPattern,
                  ResourceDeallocaOpPattern, ResourceSizeOpPattern,
                  ResourceMapOpPattern, ResourceTryMapOpPattern,
                  ResourceLoadOpPattern, ResourceStoreOpPattern,
                  ResourceSubviewOpPattern>(mapping, typeConverter, context);
  patterns.insert<TensorImportBufferOpPattern, TensorImportBufferViewOpPattern,
                  TensorExportBufferOpPattern, TensorExportBufferViewOpPattern,
                  TensorTraceOpPattern>(mapping, typeConverter, context);
  patterns
      .insert<CmdFlushOpPattern, CmdInvalidateOpPattern, CmdDiscardOpPattern,
              CmdFillOpPattern, CmdCopyOpPattern, CmdDispatchOpPattern,
              CmdExecuteOpPattern, CmdSerialOpPattern, CmdConcurrentOpPattern>(
          mapping, typeConverter, context);
  patterns.insert<TimepointImmediateOpPattern, TimepointImportOpPattern,
                  TimepointExportOpPattern, TimepointJoinOpPattern,
                  TimepointAwaitOpPattern>(mapping, typeConverter, context);
  patterns.insert<ElideYieldOpPattern>(mapping, typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
