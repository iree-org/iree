// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/StreamToHAL/Patterns.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/DeviceSwitchBuilder.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

static Value lookupDeviceFor(Operation *op, OpBuilder &builder) {
  // TODO(benvanik): make this do multi-device lookup and other fancy things.
  auto lookupOp = builder.create<IREE::HAL::ExSharedDeviceOp>(op->getLoc());
  return lookupOp.getResult();
}

// Returns the device queue affinity mask indicating which device queues the
// operations are allowed to execute on.
static Value buildQueueAffinityMaskFor(Operation *op, Value device,
                                       OpBuilder &builder) {
  // Try to find a specified affinity. This may be on the op provided or one of
  // its parent regions.
  auto affinityAttr = IREE::Stream::AffinityAttr::lookup(op);
  if (auto queueAffinityAttr =
          affinityAttr.dyn_cast_or_null<IREE::HAL::AffinityQueueAttr>()) {
    return builder.create<arith::ConstantIntOp>(
        op->getLoc(), queueAffinityAttr.getMask(), 64);
  }

  // No affinity specified; use default (any) affinity.
  return builder.create<arith::ConstantIntOp>(op->getLoc(), -1, 64);
}

static std::tuple<Value, Value> lookupDeviceAndQueueAffinityFor(
    Operation *op, OpBuilder &builder) {
  // NOTE: we have this combined method so that we can reuse any expensive
  // lookups we need to do. Today we aren't duplicating the lookups and don't
  // bother.

  // Get a device handle used to create resources and schedule work.
  // It may be shared across many mutually-exclusive devices at runtime.
  Value device = lookupDeviceFor(op, builder);

  // Derive the queue affinity mask from the op and device combination.
  Value queueAffinity = buildQueueAffinityMaskFor(op, device, builder);

  return std::make_tuple(device, queueAffinity);
}

static Value lookupAllocatorFor(Operation *op, OpBuilder &builder) {
  auto device = lookupDeviceFor(op, builder);
  auto allocatorOp =
      builder.create<IREE::HAL::DeviceAllocatorOp>(op->getLoc(), device);
  return allocatorOp.getResult();
}

// Returns the |timepointFence| or a util.null.
static Value getOrCreateWaitFence(Location loc, Value timepointFence,
                                  OpBuilder &builder) {
  if (timepointFence) return timepointFence;
  return builder.create<IREE::Util::NullOp>(
      loc, builder.getType<IREE::HAL::FenceType>());
}

// Finds a !hal.fence bound to |timepoint| via a chain op and returns it if
// it is usable at the builder insertion point. The chain ops is only used if
// it is the only consumer of the timepoint and it is removed upon return.
static Value consumeBoundFence(Value timepoint,
                               ConversionPatternRewriter &rewriter) {
  // Must only have one use. We can't consume a fence multiple times.
  if (!timepoint.hasOneUse()) return nullptr;  // >1 use

  // The use must be an export to a fence.
  auto chainOp = dyn_cast<IREE::Stream::TimepointChainExternalOp>(
      *timepoint.getUsers().begin());
  if (!chainOp) return nullptr;  // non-export use
  assert(!chainOp.getExternalValues().empty());
  auto fence = chainOp.getExternalValues().front();
  if (!fence || !fence.getType().isa<IREE::HAL::FenceType>()) return nullptr;

  // Try really hard to figure out if the fence can be used. A larger analysis
  // pass running prior to conversion that did some code motion could help
  // ensure the fence SSA value is usable in the places it is needed - for now
  // we just do this local check that satisfies most common programs today. IPO
  // would do something like add the fence as an argument to function calls so
  // that the functions could consume it but inlining is pretty aggressive now.
  if (!IREE::Util::isValueUsableForOp(fence, rewriter.getBlock(),
                                      rewriter.getInsertionPoint())) {
    return nullptr;  // unusable
  }

  // Consume the op by erasing it.
  rewriter.eraseOp(chainOp);

  return fence;  // usable
}

// Returns the a new fence for |timepoint| or an existing fence if one was
// associated with an external fence. Returns util.null if no one observes the
// fence.
static Value getOrCreateSignalFence(Location loc, Value device, Value timepoint,
                                    ConversionPatternRewriter &rewriter) {
  // Check to see if anyone is consuming the timepoint - if not then we don't
  // need create a fence.
  if (timepoint.use_empty()) {
    return rewriter.create<IREE::Util::NullOp>(
        loc, rewriter.getType<IREE::HAL::FenceType>());
  }

  // Check to see if the timepoint is associated with a fence. In common cases
  // when along ABI boundaries we can usually find an association.
  auto fence = consumeBoundFence(timepoint, rewriter);
  if (fence) return fence;

  // Create a new fence.
  return rewriter.create<IREE::HAL::FenceCreateOp>(
      loc, rewriter.getType<IREE::HAL::FenceType>(), device,
      IREE::HAL::FenceFlagBitfield::None);
}

// Scans all of the stream.cmd.* ops in the region to derive a command category.
static IREE::HAL::CommandCategoryBitfield deriveCommandCategories(
    Region &region) {
  auto bits = IREE::HAL::CommandCategoryBitfield::None;
  for (auto &block : region) {
    for (auto &op : block) {
      if (isa<IREE::Stream::CmdCollectiveOp>(op) ||
          isa<IREE::Stream::CmdCallOp>(op)) {
        // Calls may do anything and collectives may be implemented as either
        // transfers or dispatches.
        bits = bits | IREE::HAL::CommandCategoryBitfield::Dispatch |
               IREE::HAL::CommandCategoryBitfield::Transfer;
      } else if (isa<IREE::Stream::CmdDispatchOp>(op)) {
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
      bufferUsage =
          bufferUsage | IREE::HAL::BufferUsageBitfield::SharingImmutable;
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
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
      break;
  }

  // TODO(benvanik): refine usage based on analysis.
  bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Transfer |
                IREE::HAL::BufferUsageBitfield::DispatchStorage;

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
      // (transferring devices/etc if needed).
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
    for (auto [resourceResult, storageSize] :
         llvm::zip_equal(allocOp.getResults(), allocOp.getStorageSizes())) {
      auto resourceType =
          resourceResult.getType().cast<IREE::Stream::ResourceType>();

      auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
      auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
      if (failed(deriveAllowedResourceBufferBits(allocOp.getLoc(), resourceType,
                                                 memoryTypes, bufferUsage))) {
        return failure();
      }

      auto allocateOp = rewriter.create<IREE::HAL::AllocatorAllocateOp>(
          allocOp.getLoc(), bufferType, allocator, memoryTypes, bufferUsage,
          storageSize);
      results.push_back(allocateOp.getResult());
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
    auto loc = allocaOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(allocaOp, rewriter);
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();

    // Transient allocations are device-local. Copies are required to get their
    // contents back on the host/another device.
    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::DeviceLocal;

    // TODO(benvanik): refine usage.
    // We know by construction that transient buffers are not host visible and
    // as such can only be used for device commands. We should be able to more
    // closely limit to just dispatch or transfer though.
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::Transfer |
                       IREE::HAL::BufferUsageBitfield::DispatchStorage;

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
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceDeallocaOp deallocaOp, OpAdaptor adaptor,
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
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceSizeOp sizeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferLengthOp>(
        sizeOp, rewriter.getIndexType(), adaptor.getOperand());
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

    rewriter.replaceOpWithNewOp<IREE::HAL::AllocatorAllocateInitializedOp>(
        mapOp, bufferType, allocator, memoryTypes, bufferUsage,
        adaptor.getSource(), adaptor.getSourceOffset(),
        adaptor.getResultSize());
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
        tryMapOp.getResult().getType().cast<IREE::Stream::ResourceType>();
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
                  IREE::HAL::BufferUsageBitfield::DispatchStorage;

    rewriter.replaceOpWithNewOp<IREE::HAL::AllocatorTryMapOp>(
        tryMapOp, rewriter.getI1Type(), bufferType, allocator, memoryTypes,
        bufferUsage, adaptor.getSource(), adaptor.getSourceOffset(),
        adaptor.getResultSize());
    return success();
  }
};

struct ResourceLoadOpPattern
    : public StreamConversionPattern<IREE::Stream::ResourceLoadOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceLoadOp loadOp, OpAdaptor adaptor,
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
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceStoreOp storeOp, OpAdaptor adaptor,
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
  LogicalResult matchAndRewrite(
      IREE::Stream::ResourceSubviewOp subviewOp, OpAdaptor adaptor,
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
    if (!importOp.getSource().getType().isa<IREE::HAL::BufferType>()) {
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
        importOp.getResult().getType().cast<IREE::Stream::ResourceType>();
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
  LogicalResult matchAndRewrite(
      IREE::Stream::TensorImportOp importOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceType = importOp.getSource().getType();
    if (!sourceType.isa<IREE::HAL::BufferViewType>() &&
        !sourceType.isa<TensorType>()) {
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
        importOp.getResult().getType().cast<IREE::Stream::ResourceType>();
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
  LogicalResult matchAndRewrite(
      IREE::Stream::TensorExportOp exportOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!exportOp.getResult().getType().isa<IREE::HAL::BufferType>()) {
      return failure();
    }
    rewriter.replaceOp(exportOp, adaptor.getSource());
    return success();
  }
};

struct TensorExportBufferViewOpPattern
    : public StreamConversionPattern<IREE::Stream::TensorExportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TensorExportOp exportOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto targetType = exportOp.getResult().getType();
    if (!targetType.isa<IREE::HAL::BufferViewType>() &&
        !targetType.isa<TensorType>()) {
      return failure();
    }

    auto loc = exportOp.getLoc();
    auto tensorType = adaptor.getSourceEncoding().cast<RankedTensorType>();
    auto dynamicDims = adaptor.getSourceEncodingDims();

    // NOTE: we should have verified supported encodings/types at entry into the
    // HAL pipeline.
    auto encodingType =
        IREE::HAL::getEncodingTypeValue(tensorType.getEncoding());
    assert(encodingType.has_value() && "invalid tensor encoding");
    auto elementType =
        IREE::HAL::getElementTypeValue(tensorType.getElementType());
    assert(elementType.has_value() && "invalid tensor element type");

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
        adaptor.getSourceSize(), elementType.value(), encodingType.value(),
        dims);
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
        traceOp, traceOp.getKeyAttr(), adaptor.getOperands());
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
        fillOp, commandBuffer, adaptor.getTarget(), adaptor.getTargetOffset(),
        adaptor.getTargetLength(), adaptor.getValue());
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
        op, commandBuffer, adaptor.getSource(), adaptor.getSourceOffset(),
        adaptor.getTarget(), adaptor.getTargetOffset(), adaptor.getLength());
    return success();
  }
};

// NOTE: this relies on the enums being the same today. Ew.
static IREE::HAL::CollectiveAttr convertCollectiveAttr(
    IREE::Stream::CollectiveAttr sourceAttr) {
  auto convertReductionOp =
      [](std::optional<IREE::Stream::CollectiveReductionOp> op)
      -> std::optional<IREE::HAL::CollectiveReductionOp> {
    if (!op.has_value()) return std::nullopt;
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
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdCollectiveOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto commandBuffer = mapping->lookupCommandBufferFor(op);

    Value sendBuffer;
    Value sendOffset;
    Value sendLength;
    Value recvBuffer;
    Value recvOffset;
    Value recvLength;
    switch (adaptor.getOp().getKind()) {
      default:
        assert(adaptor.getResources().size() == 2 && "should have verified");
        sendBuffer = adaptor.getResources()[0];
        sendOffset = adaptor.getResourceOffsets()[0];
        sendLength = adaptor.getResourceLengths()[0];
        recvBuffer = adaptor.getResources()[1];
        recvOffset = adaptor.getResourceOffsets()[1];
        recvLength = adaptor.getResourceLengths()[1];
        break;
      case IREE::Stream::CollectiveKind::Send:
        assert(adaptor.getResources().size() == 1 && "should have verified");
        sendBuffer = adaptor.getResources()[0];
        sendOffset = adaptor.getResourceOffsets()[0];
        sendLength = adaptor.getResourceLengths()[0];
        break;
      case IREE::Stream::CollectiveKind::Recv:
        assert(adaptor.getResources().size() == 1 && "should have verified");
        recvBuffer = adaptor.getResources()[0];
        recvOffset = adaptor.getResourceOffsets()[0];
        recvLength = adaptor.getResourceLengths()[0];
        break;
    }

    rewriter.replaceOpWithNewOp<IREE::HAL::CommandBufferCollectiveOp>(
        op, commandBuffer, adaptor.getChannel(),
        convertCollectiveAttr(adaptor.getOp()), adaptor.getElementCount(),
        adaptor.getParam(), sendBuffer, sendOffset, sendLength, recvBuffer,
        recvOffset, recvLength);
    return success();
  }
};

// Returns a hal.device.switch match expression that selects the given export.
static Attribute getExportConditionAttr(
    IREE::HAL::ExecutableExportOp exportOp) {
  // TODO(benvanik): customizable selection logic. Today this just checks
  // whether the variant target is supported but we can also allow
  // specialization of entry points based on dispatch site parameters.
  auto variantOp = exportOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  return variantOp.getTarget().getMatchExpression();
}

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

    // Ask each target backend to record their dispatch logic.
    IREE::HAL::DeviceSwitchRewriter switchRewriter(loc,
                                                   /*resultTypes=*/TypeRange{},
                                                   device, rewriter);
    dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
      // NOTE: slow lookup!
      auto exportOp =
          SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableExportOp>(
              dispatchOp, entryPointAttr);
      assert(exportOp && "dispatch target export not found");

      // Setup the case condition for the entry point.
      auto *caseRegion =
          switchRewriter.addConditionRegion(getExportConditionAttr(exportOp));
      auto &entryBlock = caseRegion->front();
      auto caseBuilder = OpBuilder::atBlockBegin(&entryBlock);

      // Record push constants and buffer bindings.
      recordParameters(loc, device, commandBuffer, dispatchOp, adaptor,
                       exportOp.getLayout(), caseBuilder);

      // Dispatch with a target-specific workgroup count.
      auto caseWorkgroupCount = exportOp.calculateWorkgroupCount(
          loc, device, adaptor.getWorkload(), caseBuilder);
      caseBuilder.create<IREE::HAL::CommandBufferDispatchSymbolOp>(
          loc, commandBuffer, entryPointAttr, caseWorkgroupCount[0],
          caseWorkgroupCount[1], caseWorkgroupCount[2]);

      caseBuilder.create<IREE::HAL::ReturnOp>(loc);
    });
    switchRewriter.build();

    rewriter.eraseOp(dispatchOp);
    return success();
  }

  void recordParameters(Location loc, Value device, Value commandBuffer,
                        IREE::Stream::CmdDispatchOp dispatchOp,
                        OpAdaptor adaptor,
                        IREE::HAL::PipelineLayoutAttr layoutAttr,
                        OpBuilder &builder) const {
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
      int pushConstantBase = 0;  // always 0 today
      SmallVector<Value> pushConstants;
      for (auto operand : adaptor.getUniformOperands()) {
        assert(
            operand.getType().isInteger(32) &&
            "expected only i32 values after iree-hal-pack-dispatch-operands");
        pushConstants.push_back(operand);
      }
      builder.create<IREE::HAL::CommandBufferPushConstantsOp>(
          loc, commandBuffer, pipelineLayout,
          builder.getIndexAttr(pushConstantBase), pushConstants);
    }

    // TODO(benvanik): typed accessors for bindings.
    auto bindingAttrs = dispatchOp->getAttr("hal.interface.bindings")
                            .dyn_cast_or_null<ArrayAttr>();
    assert(bindingAttrs &&
           "interface materialization must annotate dispatch sites");

    // Push descriptor bindings.
    int64_t currentSet = -1;
    SmallVector<IREE::HAL::DescriptorSetBindingValue> bindings;
    auto flushSet = [&]() {
      builder.create<IREE::HAL::CommandBufferPushDescriptorSetOp>(
          loc, commandBuffer, pipelineLayout, currentSet, bindings);
      bindings.clear();
    };
    for (unsigned i = 0; i < adaptor.getResources().size(); ++i) {
      auto bindingAttr =
          bindingAttrs[i].cast<IREE::HAL::InterfaceBindingAttr>();
      int64_t set = bindingAttr.getSet();
      if (currentSet != -1 && currentSet != set) flushSet();
      currentSet = set;
      IREE::HAL::DescriptorSetBindingValue binding;
      binding.ordinal =
          builder.create<arith::ConstantIndexOp>(loc, bindingAttr.getBinding());
      binding.buffer = adaptor.getResources()[i];
      binding.byteOffset = adaptor.getResourceOffsets()[i];
      binding.byteLength = adaptor.getResourceLengths()[i];
      bindings.push_back(binding);
    }
    if (currentSet != -1) flushSet();
  }
};

struct CmdFuncOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdFuncOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdFuncOp funcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newArgTypes;
    SmallVector<DictionaryAttr> newArgAttrs;
    newArgTypes.push_back(rewriter.getType<IREE::HAL::CommandBufferType>());
    newArgAttrs.push_back(rewriter.getDictionaryAttr({}));  // command buffer
    funcOp.getAllArgAttrs(newArgAttrs);
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(funcOp.getArgumentTypes(),
                                                newArgTypes)) ||
        failed(getTypeConverter()->convertTypes(funcOp.getResultTypes(),
                                                newResultTypes))) {
      return rewriter.notifyMatchFailure(funcOp, "failed to convert types");
    }
    auto newOp = rewriter.replaceOpWithNewOp<func::FuncOp>(
        funcOp, funcOp.getName(),
        rewriter.getFunctionType(newArgTypes, newResultTypes),
        funcOp.getSymVisibilityAttr(),
        rewriter.getArrayAttr(
            ArrayRef<Attribute>(newArgAttrs.data(), newArgAttrs.size())),
        funcOp.getAllResultAttrs());
    newOp->setDialectAttrs(funcOp->getDialectAttrs());
    return success();
  }
};

struct CmdCallOpPattern
    : public StreamConversionPattern<IREE::Stream::CmdCallOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdCallOp callOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto commandBuffer = mapping->lookupCommandBufferFor(callOp);

    // Always pass the command buffer as the first arg.
    SmallVector<Value> operands;
    operands.push_back(commandBuffer);
    size_t resourceIndex = 0;
    for (auto [originalOperand, convertedOperand] : llvm::zip_equal(
             callOp.getResourceOperands(), adaptor.getResourceOperands())) {
      if (originalOperand.getType().isa<IREE::Stream::ResourceType>()) {
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

    rewriter.replaceOpWithNewOp<func::CallOp>(callOp, callOp.getCalleeAttr(),
                                              resultTypes, operands);
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
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(executeOp, rewriter);

    // If there are any wait timepoints it means there's prior queued execution
    // that we may need to wait behind and we can't execute inline. HAL
    // implementations may be able to flush eagerly if they are able to tell
    // that all conditions are met during recording but we leave that to them.
    auto modes = IREE::HAL::CommandBufferModeBitfield::OneShot;
    if (!executeOp.getAwaitTimepoint()) {
      modes =
          modes | IREE::HAL::CommandBufferModeBitfield::AllowInlineExecution;
    }

    // Derive the command buffer type based on the kind of operations present.
    // This can help the submission get routed to appropriate hardware queues
    // (like dedicated DMA controllers).
    auto commandCategories = deriveCommandCategories(executeOp.getBody());

    // Create a new command buffer for recording.
    auto commandBuffer =
        rewriter
            .create<IREE::HAL::CommandBufferCreateOp>(
                loc, rewriter.getType<IREE::HAL::CommandBufferType>(), device,
                modes, commandCategories, /*binding_capacity=*/Value{})
            .getResult();
    mapping->mapCommandBuffer(executeOp, commandBuffer);

    // Run through the execution region and serialize execution by inserting
    // barriers. Nested regions may elide barriers as needed.
    auto &bodyBlock = executeOp.getBody().front();
    insertSerializationBarriers(loc, bodyBlock, commandBuffer,
                                OpBuilder::atBlockBegin(&bodyBlock));

    // Begin/end recording and inline the execution region between them.
    auto endOp =
        rewriter.create<IREE::HAL::CommandBufferFinalizeOp>(loc, commandBuffer);
    rewriter.inlineBlockBefore(&executeOp.getBody().front(), endOp,
                               adaptor.getResourceOperands());

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, executeOp.getResultTimepoint(), rewriter);

    // Queue execution.
    rewriter.create<IREE::HAL::DeviceQueueExecuteOp>(loc, device, queueAffinity,
                                                     waitFence, signalFence,
                                                     ValueRange{commandBuffer});

    rewriter.replaceOp(executeOp, signalFence);
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
    auto &bodyBlock = serialOp.getBody().front();
    insertSerializationBarriers(serialOp.getLoc(), bodyBlock, commandBuffer,
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
  LogicalResult matchAndRewrite(
      IREE::Stream::CmdConcurrentOp concurrentOp, OpAdaptor adaptor,
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
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointImmediateOp immediateOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Util::NullOp>(
        immediateOp, rewriter.getType<IREE::HAL::FenceType>());
    return success();
  }
};

struct TimepointImportOpPattern
    : public StreamConversionPattern<IREE::Stream::TimepointImportOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointImportOp importOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only handle imports from HAL semaphores _or_ fences.
    auto operands = adaptor.getOperands();
    if (operands.size() == 1 &&
        operands[0].getType().isa<IREE::HAL::FenceType>()) {
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
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointExportOp exportOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only handle exports into HAL fences.
    if (exportOp.getNumResults() != 1 ||
        !exportOp.getResult(0).getType().isa<IREE::HAL::FenceType>()) {
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
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointChainExternalOp exportOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only handle exports into HAL fences.
    auto externalValues = exportOp.getExternalValues();
    if (externalValues.size() != 1 ||
        !externalValues[0].getType().isa<IREE::HAL::FenceType>()) {
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
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointJoinOp joinOp, OpAdaptor adaptor,
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
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointBarrierOp barrierOp, OpAdaptor adaptor,
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
  LogicalResult matchAndRewrite(
      IREE::Stream::TimepointAwaitOp awaitOp, OpAdaptor adaptor,
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
  LogicalResult matchAndRewrite(
      IREE::Stream::ChannelCreateOp createOp, OpAdaptor adaptor,
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

struct ChannelRankOpPattern
    : public StreamConversionPattern<IREE::Stream::ChannelRankOp> {
  using StreamConversionPattern::StreamConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Stream::ChannelRankOp rankOp, OpAdaptor adaptor,
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
  LogicalResult matchAndRewrite(
      IREE::Stream::ChannelCountOp countOp, OpAdaptor adaptor,
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
    auto initialValue = op.getInitialValue();
    if (!initialValue.has_value()) return failure();
    if (!initialValue->isa<IREE::Stream::TimepointAttr>()) return failure();
    rewriter.updateRootInPlace(op, [&]() { op.removeInitialValueAttr(); });
    return success();
  }
};

}  // namespace

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
              CmdFillOpPattern, CmdCopyOpPattern, CmdCollectiveOpPattern,
              CmdDispatchOpPattern, CmdFuncOpPattern, CmdCallOpPattern,
              CmdExecuteOpPattern, CmdSerialOpPattern, CmdConcurrentOpPattern>(
          mapping, typeConverter, context);
  patterns.insert<TimepointImmediateOpPattern, TimepointImportOpPattern,
                  TimepointExportOpPattern, TimepointChainExternalOpPattern,
                  TimepointJoinOpPattern, TimepointBarrierOpPattern,
                  TimepointAwaitOpPattern>(mapping, typeConverter, context);
  patterns.insert<ChannelCreateOpPattern, ChannelRankOpPattern,
                  ChannelCountOpPattern>(mapping, typeConverter, context);
  patterns.insert<ElideYieldOpPattern>(mapping, typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
