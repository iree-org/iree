// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Conversion/FlowToHAL/ConvertFlowToHAL.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Utils/DeviceSwitchBuilder.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-hal"

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// Alias analysis
//===----------------------------------------------------------------------===//

using ValueAliasingMap = llvm::MapVector<Value, SmallPtrSet<Value, 16>>;

// Builds a map of value aliases from aliasee to a set of aliasers.
// Only values that alias will be present in the map.
static ValueAliasingMap computeValueAliases(
    IREE::Flow::ExStreamFragmentOp streamOp) {
  auto *streamBlock = &streamOp.body().front();
  ValueAliasingMap valueAliases;

  std::function<void(Value streamValue, Value aliasedValue)> propagateAlias;
  propagateAlias = [&](Value streamValue, Value aliasedValue) {
    auto &baseSet = valueAliases[streamValue];
    baseSet.insert(aliasedValue);
    auto &aliasedSet = valueAliases[aliasedValue];
    baseSet.insert(aliasedSet.begin(), aliasedSet.end());
    aliasedSet.insert(streamValue);
  };

  // Start with outputs so that we handle tied values that may lead all the way
  // back up the chain to the stream inputs.
  auto tiedStreamOp = cast<IREE::TiedOpInterface>(streamOp.getOperation());
  auto returnOp = cast<IREE::Flow::ReturnOp>(streamBlock->back());
  for (auto result : llvm::enumerate(streamOp.getResults())) {
    auto streamValue = returnOp.getOperand(result.index());

    // Tied stream results reuse their stream operand buffer.
    auto tiedOperandIndex =
        tiedStreamOp.getTiedResultOperandIndex(result.index());
    if (tiedOperandIndex.hasValue()) {
      auto operand = streamBlock->getArgument(tiedOperandIndex.getValue());
      propagateAlias(streamValue, operand);
    }
  }

  for (auto &op : *streamBlock) {
    auto tiedOp = dyn_cast<IREE::TiedOpInterface>(op);
    for (auto it : llvm::enumerate(op.getResults())) {
      auto result = it.value();
      if (!result.getType().isa<ShapedType>()) continue;

      // Tied results reuse their operand buffer.
      if (tiedOp) {
        auto tiedOperandIndex = tiedOp.getTiedResultOperandIndex(it.index());
        if (tiedOperandIndex.hasValue()) {
          auto operand = op.getOperand(tiedOperandIndex.getValue());
          propagateAlias(result, operand);
        }
      }
    }
  }

  // Inverse the value aliaser->aliasee map so that we have for any particular
  // value the list of all other values that alias it.
  for (auto it : valueAliases) {
    for (auto aliasee : it.second) {
      for (auto aliaser : it.second) {
        if (aliaser != aliasee) {
          valueAliases[aliasee].insert(aliaser);
        }
      }
    }
  }

  return valueAliases;
}

//===----------------------------------------------------------------------===//
// Liveness interval analysis
//===----------------------------------------------------------------------===//

static constexpr int LIVE_IN = INT_MIN;
static constexpr int LIVE_OUT = INT_MAX;
struct LivenessInterval {
  int start = 0;
  int end = 0;
  int ordinal = -1;  // unique per value
  Value value;
  bool operator<(const LivenessInterval &rhs) const {
    return ordinal < rhs.ordinal;
  }
};
using LivenessIntervalMap = DenseMap<Value, LivenessInterval>;
using LivenessIntervalList = SmallVector<LivenessInterval>;

// Computes the liveness intervals for each value in the stream.
// Returns a closed range over an arbitrary operation ordering. The LIVE_IN and
// LIVE_OUT sentinels will be used to indicate values that are live-in and
// live-out to the stream (captured input arguments and escaping output
// results).
//
// All values will have a range with aliased values sharing the union of their
// constituent ranges - including block arguments. Note that not all values will
// have buffers allocated to them - we are just tracking transitive SSA value
// lifetime.
static LivenessIntervalList computeLivenessIntervals(
    IREE::Flow::ExStreamFragmentOp streamOp,
    const ValueAliasingMap &valueAliases) {
  // Perform a liveness analysis on the stream fragment.
  // Fragments have a single block and as such the live-in/live-out block
  // information derived here applies to the entire stream region.
  assert(streamOp.body().getBlocks().size() == 1);
  auto *streamBlock = &streamOp.body().front();
  Liveness streamLiveness(streamOp);
  auto *livenessInfo = streamLiveness.getLiveness(streamBlock);

  // Operations don't allow us to get their already computed order so we make up
  // our own. We have a single block and thus the ordering is complete.
  DenseMap<Operation *, int> opOrdering;
  for (auto &op : *streamBlock) {
    opOrdering[&op] = opOrdering.size();
  }

  // Liveness doesn't track return values as live-outs so we do that here.
  SmallPtrSet<Value, 16> liveOuts;
  auto returnOp = cast<IREE::Flow::ReturnOp>(streamBlock->back());
  for (auto returnValue : returnOp.operands()) {
    if (!returnValue.getType().isa<ShapedType>()) continue;
    liveOuts.insert(returnValue);
  }

  // Compute live-in intervals special as we won't catch them in the op walk
  // below as they are block arguments.
  LivenessIntervalMap valueIntervals;
  int ordinal = 0;
  for (Value value : streamBlock->getArguments()) {
    if (!value.getType().isa<ShapedType>()) continue;
    LivenessInterval interval;
    interval.start = LIVE_IN;
    if (liveOuts.contains(value)) {
      interval.end = LIVE_OUT;
    } else {
      auto *endOp = livenessInfo->getEndOperation(value, &streamBlock->front());
      interval.end = opOrdering[endOp];
    }
    interval.value = value;
    interval.ordinal = ++ordinal;
    valueIntervals[value] = interval;
  }

  // Compute ranges for all values independently (ignoring aliasing).
  for (auto &op : *streamBlock) {
    int start = opOrdering[&op];
    for (auto value : op.getResults()) {
      if (!value.getType().isa<ShapedType>()) continue;
      LivenessInterval interval;
      interval.start = start;
      if (liveOuts.contains(value)) {
        interval.end = LIVE_OUT;
      } else {
        interval.end = start;
        for (auto &use : value.getUses()) {
          interval.end = std::max(interval.end, opOrdering[use.getOwner()]);
        }
      }
      interval.value = value;
      interval.ordinal = ++ordinal;
      valueIntervals[value] = interval;
    }
  }

  // Walk the alias map and union intervals and propagate back.
  for (auto it : valueAliases) {
    auto &aliasee = it.first;
    auto &aliasers = it.second;
    auto &aliaseeInterval = valueIntervals[aliasee];
    int start = aliaseeInterval.start;
    int end = aliaseeInterval.end;
    for (auto aliaser : aliasers) {
      auto &aliaserInterval = valueIntervals[aliaser];
      start = std::min(start, aliaserInterval.start);
      end = std::max(end, aliaserInterval.end);
    }
    aliaseeInterval.start = start;
    aliaseeInterval.end = end;
    for (auto aliaser : aliasers) {
      auto &aliaserInterval = valueIntervals[aliaser];
      aliaserInterval.start = start;
      aliaserInterval.end = end;
    }
  }

  // Sort all intervals by lifetime start. This makes the intervals easier to
  // read and deterministic across runs.
  SmallVector<LivenessInterval> sortedIntervals;
  for (auto it : valueIntervals) {
    sortedIntervals.push_back(it.second);
  }
  std::sort(sortedIntervals.begin(), sortedIntervals.end());
  return sortedIntervals;
}

//===----------------------------------------------------------------------===//
// Stateful recording storage
//===----------------------------------------------------------------------===//

struct BufferRange {
  BufferRange() = default;
  explicit BufferRange(Value buffer, Value length)
      : buffer(buffer), length(length) {}

  Value buffer = nullptr;
  Value length = nullptr;
};

// State cache used during stream scheduling.
//
// This contains caches used to memoize commonly occuring values such as
// variable loads, shapes, and computed sizes. These caches are just to lighten
// the load on cse/canonicalization and otherwise it would be fine to
// materialize IR for everything.
//
// Any allocations made are also tracked here so that the tensor->!hal.buffer
// mappings are available at any time a buffer may be required during the
// scheduling.
class StreamSchedulingState {
 public:
  explicit StreamSchedulingState(Location loc, Value device, Value allocator,
                                 ValueAliasingMap &valueAliases)
      : loc(loc),
        device_(device),
        allocator_(allocator),
        valueAliases(valueAliases) {}

  Value device() { return device_; }
  Value allocator() { return allocator_; }

  // Returns a ConstantIndexOp of |value|.
  Value lookupOrCreateIndex(int64_t value, OpBuilder &builder) {
    auto it = indexConstantMap.find(value);
    if (it != indexConstantMap.end()) return it->second;
    auto constantValue = builder.createOrFold<ConstantIndexOp>(loc, value);
    indexConstantMap.insert(std::make_pair(value, constantValue));
    return constantValue;
  }

  // Loads a variable with the given |symName|.
  Value loadVariable(Type resultType, StringRef symName, OpBuilder &builder) {
    auto it = loadedVariableMap.find(symName);
    if (it != loadedVariableMap.end()) {
      assert(it->second.getType() == resultType && "variable type mismatch");
      return it->second;
    }
    auto value = builder.createOrFold<IREE::HAL::VariableLoadOp>(
        loc, resultType, symName);
    loadedVariableMap.insert(std::make_pair(symName, value));
    return value;
  }

  // Returns an executable layout with the given attributes.
  Value lookupExecutableLayout(Type resultType, IntegerAttr pushConstantsAttr,
                               ArrayAttr layoutsAttr, OpBuilder &builder) {
    auto keyAttr = builder.getArrayAttr({pushConstantsAttr, layoutsAttr});
    auto it = executableLayoutMap.find(keyAttr);
    if (it != executableLayoutMap.end()) {
      assert(it->second.getType() == resultType && "variable type mismatch");
      return it->second;
    }
    auto value = builder.createOrFold<IREE::HAL::ExecutableLayoutLookupOp>(
        loc, IREE::HAL::ExecutableLayoutType::get(device().getContext()),
        device(), pushConstantsAttr, layoutsAttr);
    executableLayoutMap.insert(std::make_pair(keyAttr, value));
    return value;
  }

  // Returns a computed shape value inserted at |builder| based on the shape of
  // the given |streamValue|. Returns an existing size if one matching the
  // parameters has already been inserted.
  Value lookupOrComputeSize(Value streamValue, OpBuilder &builder) {
    return lookupOrComputeSize(streamValue.getType().cast<ShapedType>(),
                               Shape::buildOrFindDynamicDimsForValue(
                                   streamValue.getLoc(), streamValue, builder),
                               builder);
  }

  // Returns a computed shape value inserted at |builder| based on the given
  // shaped type and its dynamic dimensions. Returns an existing size if one
  // matching the parameters has already been inserted.
  Value lookupOrComputeSize(ShapedType shapedType, ValueRange dynamicDims,
                            OpBuilder &builder) {
    if (shapedType.hasStaticShape()) {
      auto it = staticShapeToSizeMap.find(shapedType);
      if (it != staticShapeToSizeMap.end()) return it->second;
    } else {
      auto typeIt = dynamicShapeToSizeMap.find(shapedType);
      if (typeIt != dynamicShapeToSizeMap.end()) {
        for (auto dimsIt : typeIt->second) {
          if (std::equal(dimsIt.first.begin(), dimsIt.first.end(),
                         dynamicDims.begin())) {
            return dimsIt.second;
          }
        }
      }
    }

    auto elementType = getElementType(shapedType.getElementType(), builder);
    assert(elementType && "unhandled element type for allocation");

    SmallVector<Value> shapeDims(shapedType.getRank());
    int64_t dynamicDimIndex = 0;
    for (int64_t i = 0; i < shapedType.getRank(); ++i) {
      if (shapedType.isDynamicDim(i)) {
        shapeDims[i] = dynamicDims[dynamicDimIndex++];
      } else {
        shapeDims[i] = lookupOrCreateIndex(shapedType.getDimSize(i), builder);
      }
    }

    auto size = builder.createOrFold<IREE::HAL::AllocatorComputeSizeOp>(
        loc, allocator(), shapeDims, elementType);
    if (shapedType.hasStaticShape()) {
      staticShapeToSizeMap[shapedType] = size;
    } else {
      dynamicShapeToSizeMap[shapedType].push_back(
          std::make_pair(dynamicDims, size));
    }
    return size;
  }

  // Maps |tensorValue| to the backing storage buffer defined by |bufferRange|.
  LogicalResult mapTensorToBufferRange(Value tensorValue,
                                       BufferRange bufferRange) {
    if (bufferRangeMap.count(tensorValue)) {
      return failure();
    }
    bufferRangeMap.insert(std::make_pair(tensorValue, bufferRange));

    // TODO(#5410): make alias propagation map through an indexing map for
    // slices/updates. Right now we assume all aliases are 1:1 full maps.
    for (auto alias : valueAliases[tensorValue]) {
      bufferRangeMap.insert(std::make_pair(alias, bufferRange));
    }
    return success();
  }

  // Returns a buffer range backing the given stream |tensorValue|.
  BufferRange lookupTensorBufferRange(Value tensorValue) {
    auto it = bufferRangeMap.find(tensorValue);
    assert(it != bufferRangeMap.end() && "buffer not pre-allocated for tensor");
    return it->second;
  }

  // Returns true if the given |tensorValue| has a buffer range mapped to it.
  bool hasTensorBufferRange(Value tensorValue) {
    return bufferRangeMap.count(tensorValue) != 0;
  }

  // Calls |callback| for |tensorValue| and each value aliasing it.
  void forEachEquivalentTensorValue(Value tensorValue,
                                    std::function<void(Value)> callback) {
    callback(tensorValue);
    for (auto alias : valueAliases[tensorValue]) {
      callback(alias);
    }
  }

 private:
  Value getElementType(Type elementType, OpBuilder &builder) {
    auto it = memoizedElementTypesConstants.find(elementType);
    if (it != memoizedElementTypesConstants.end()) return it->second;
    auto i32Value = IREE::HAL::getElementTypeValue(elementType);
    assert(i32Value.hasValue() && "unhandled element type for allocation");
    auto constantValue =
        builder.createOrFold<ConstantIntOp>(loc, i32Value.getValue(), 32);
    memoizedElementTypesConstants[elementType] = constantValue;
    return constantValue;
  }

  Location loc;

  // !hal.device used throughout the stream.
  Value device_;
  // !hal.allocator used throughout the stream.
  Value allocator_;

  // All values that have aliases mapped to a set of all of the values they
  // alias with. That two things alias does not imply the values can be treated
  // as equivalent: some values may be subranges of others.
  ValueAliasingMap valueAliases;

  // Index value -> std.constant index value.
  DenseMap<int64_t, Value> indexConstantMap;

  // Variable sym name -> loaded value.
  DenseMap<StringRef, Value> loadedVariableMap;

  // Key of [push constants, set layouts] -> loaded value.
  DenseMap<Attribute, Value> executableLayoutMap;

  // Small cache of constants used for element types.
  DenseMap<Type, Value> memoizedElementTypesConstants;

  // Map of static shaped types to computed size values.
  DenseMap<Type, Value> staticShapeToSizeMap;

  // Map of dynamic shaped types to a set of dynamic dimension SSA values and
  // the corresponding computed size. A single shaped type such as `?x4` may
  // have multiple unique sizes if differing dimension values are used (such as
  // `{%dimA}x4` and `{%dimB}x4`).
  DenseMap<Type, SmallVector<std::pair<SmallVector<Value>, Value>>>
      dynamicShapeToSizeMap;

  // Maps tensor values inside the stream to a buffer range that stores them.
  DenseMap<Value, BufferRange> bufferRangeMap;
};

//===----------------------------------------------------------------------===//
// Buffer allocation
//===----------------------------------------------------------------------===//

// Allocates a buffer for the given stream output value.
// |streamValue| is the Value used within the stream region and
// |externalValue| is the returned value from the stream region in the parent
// block.
static BufferRange allocateOutputBuffer(Value streamValue, Value externalValue,
                                        StreamSchedulingState &schedulingState,
                                        ConversionPatternRewriter &rewriter) {
  Location loc = externalValue.getLoc();

  // TODO(benvanik): compute from SSA use-def chain uses.
  // HACK: we have no idea right now whether the buffer escaping the stream will
  // be used on the host or the device and have to allocate it as HOST_VISIBLE.
  // Improvements to the flow dialect to track tensor lifetime and hal.stream
  // for tracking usage will reduce this to just what is required.
  IREE::HAL::MemoryTypeBitfield memoryTypes =
      IREE::HAL::MemoryTypeBitfield::DeviceLocal |
      IREE::HAL::MemoryTypeBitfield::HostVisible;
  IREE::HAL::BufferUsageBitfield bufferUsage =
      IREE::HAL::BufferUsageBitfield::All;

  // Compute the allocation size for the value.
  auto allocationSize = schedulingState.lookupOrComputeSize(
      streamValue.getType().cast<ShapedType>(),
      Shape::buildOrFindDynamicDimsForValue(streamValue.getLoc(), streamValue,
                                            rewriter),
      rewriter);

  auto buffer = rewriter
                    .create<IREE::HAL::AllocatorAllocateOp>(
                        loc, IREE::HAL::BufferType::get(rewriter.getContext()),
                        schedulingState.allocator(), memoryTypes, bufferUsage,
                        allocationSize)
                    .getResult();

  return BufferRange{buffer, allocationSize};
}

// Allocates all output buffers for the stream and populates the
// |schedulingState| with the new mappings. Returns the set of output buffers
// mapping 1:1 with the |streamOp| results.
static LogicalResult allocateOutputBuffers(
    IREE::Flow::ExStreamFragmentOp streamOp,
    StreamSchedulingState &schedulingState, ConversionPatternRewriter &rewriter,
    SmallVectorImpl<Value> &output) {
  auto tiedStreamOp = cast<IREE::TiedOpInterface>(streamOp.getOperation());
  auto &entryBlock = streamOp.body().front();

  SmallVector<Value> outputBuffers;

  // Allocate output buffers and replace the original uses with the buffers.
  auto returnOp = cast<IREE::Flow::ReturnOp>(streamOp.body().front().back());
  for (auto result : llvm::enumerate(streamOp.getResults())) {
    auto streamValue = returnOp.getOperand(result.index());
    auto externalValue = result.value();

    // Ignore already allocated buffers.
    if (schedulingState.hasTensorBufferRange(streamValue)) {
      outputBuffers.push_back(
          schedulingState.lookupTensorBufferRange(streamValue).buffer);
      continue;
    }

    // Tied stream results reuse their operand buffer.
    BufferRange bufferRange;
    auto tiedOperandIndex =
        tiedStreamOp.getTiedResultOperandIndex(result.index());
    if (tiedOperandIndex.hasValue()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    -- REUSING TIED OPERAND("
                 << tiedOperandIndex.getValue() << ") BUFFER FOR STREAM RESULT("
                 << result.index() << "): " << streamOp << "\n");
      auto operand = entryBlock.getArgument(tiedOperandIndex.getValue());
      bufferRange = schedulingState.lookupTensorBufferRange(operand);
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "    -- ALLOCATE BUFFER FOR STREAM ESCAPE RESULT("
                 << result.index() << ")\n");
      bufferRange = allocateOutputBuffer(streamValue, externalValue,
                                         schedulingState, rewriter);
    }
    if (!bufferRange.buffer) {
      return streamOp.emitOpError() << "buffer range has no buffer";
    }
    outputBuffers.push_back(bufferRange.buffer);
    if (failed(
            schedulingState.mapTensorToBufferRange(streamValue, bufferRange))) {
      return streamOp.emitOpError() << "tensor was mapped to multiple buffer "
                                       "ranges while allocating output buffers";
    }
  }

  output = outputBuffers;
  return success();
}

// Allocates transient buffers to store the intra-stream results and populates
// the |schedulingState| with the new mappings.
static LogicalResult allocateTransientBuffers(
    IREE::Flow::ExStreamFragmentOp streamOp,
    LivenessIntervalList &livenessIntervals,
    StreamSchedulingState &schedulingState,
    ConversionPatternRewriter &rewriter) {
  // TODO(#5410): unify with slice/update handling below. We should have a
  // more generic way of handling these special ops and need to be able to hook
  // into ones that directly control aliasing behavior like slice/update.
  SmallPtrSet<Value, 16> coveredValues;
  auto walkResult = streamOp.walk([&](IREE::HAL::ConstantSubspanOp subspanOp) {
    auto tensorValue = subspanOp.result();
    auto bufferValue = schedulingState.loadVariable(
        IREE::HAL::BufferType::get(rewriter.getContext()),
        subspanOp.runtime_buffer().getLeafReference(), rewriter);
    auto offsetValue = schedulingState.lookupOrCreateIndex(
        subspanOp.runtime_range().offset().getSExtValue(), rewriter);
    auto lengthValue = schedulingState.lookupOrCreateIndex(
        subspanOp.runtime_range().length().getSExtValue(), rewriter);
    auto subspanValue = rewriter.createOrFold<IREE::HAL::BufferSubspanOp>(
        subspanOp.getLoc(), bufferValue.getType(), bufferValue, offsetValue,
        lengthValue);
    auto bufferRange = BufferRange{subspanValue, lengthValue};
    if (failed(
            schedulingState.mapTensorToBufferRange(tensorValue, bufferRange))) {
      return WalkResult::interrupt();
    }
    schedulingState.forEachEquivalentTensorValue(
        tensorValue, [&](Value alias) { coveredValues.insert(alias); });
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return streamOp.emitOpError() << "constant subspan op was mapped to "
                                     "multiple buffer ranges while allocating "
                                     "transient buffers";
  }

  // Gather all of the transient values we need to allocate buffers for.
  SmallVector<Value> transientValues;
  SmallVector<int64_t> lifetimeIntervals;
  SmallVector<Value> dynamicSliceSizes;
  AsmState state(streamOp);
  for (auto valueInterval : livenessIntervals) {
    auto value = valueInterval.value;
    auto valueType = value.getType().dyn_cast<ShapedType>();
    if (!valueType) continue;

    // Only handle transient buffers (created/used/dropped within the stream).
    if (valueInterval.start == LIVE_IN || valueInterval.end == LIVE_OUT) {
      continue;
    }

    // Ignore covered values.
    if (schedulingState.hasTensorBufferRange(value) ||
        coveredValues.contains(value)) {
      continue;
    }

    transientValues.push_back(value);
    lifetimeIntervals.push_back(valueInterval.start);
    lifetimeIntervals.push_back(valueInterval.end);

    // Compute the allocation size for the value.
    auto allocationSize = schedulingState.lookupOrComputeSize(
        valueType,
        Shape::buildOrFindDynamicDimsForValue(value.getLoc(), value, rewriter),
        rewriter);
    dynamicSliceSizes.push_back(allocationSize);

    // Mark as covered so we don't allocate it again.
    schedulingState.forEachEquivalentTensorValue(
        value, [&](Value alias) { coveredValues.insert(alias); });
  }
  if (transientValues.empty()) {
    // No transients required.
    return success();
  }

  // Insert the hal.allocator.pack op to compute the packed offsets and total
  // buffer size required for all transients.
  auto indexType = rewriter.getIndexType();
  SmallVector<Type> packedOffsetTypes(dynamicSliceSizes.size(), indexType);
  auto packOp = rewriter.create<IREE::HAL::AllocatorPackOp>(
      streamOp.getLoc(), indexType, packedOffsetTypes,
      schedulingState.allocator(),
      /*offset=*/nullptr, rewriter.getIndexArrayAttr(lifetimeIntervals),
      dynamicSliceSizes);

  // Allocate the transient storage buffer.
  // TODO(benvanik): compute from SSA use-def chain uses.
  IREE::HAL::MemoryTypeBitfield memoryTypes =
      IREE::HAL::MemoryTypeBitfield::DeviceLocal;
  IREE::HAL::BufferUsageBitfield bufferUsage =
      IREE::HAL::BufferUsageBitfield::Dispatch |
      IREE::HAL::BufferUsageBitfield::Transfer;
  auto allocateOp = rewriter.create<IREE::HAL::AllocatorAllocateOp>(
      streamOp.getLoc(), IREE::HAL::BufferType::get(rewriter.getContext()),
      schedulingState.allocator(), memoryTypes, bufferUsage,
      packOp.total_length());

  // Add a buffer set map entry for each transient buffer that references into
  // a subspan of the transient storage buffer.
  for (size_t i = 0; i < transientValues.size(); ++i) {
    auto value = transientValues[i];
    auto offset = packOp.packed_offsets()[i];
    auto subspanValue = rewriter.createOrFold<IREE::HAL::BufferSubspanOp>(
        value.getLoc(), allocateOp.result().getType(), allocateOp.result(),
        offset, dynamicSliceSizes[i]);
    auto bufferRange = BufferRange{subspanValue, dynamicSliceSizes[i]};
    if (failed(schedulingState.mapTensorToBufferRange(value, bufferRange))) {
      return streamOp.emitOpError()
             << "tensor for buffer subspan was mapped to multiple buffer "
                "ranges while allocating transient buffers";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

// Records a full execution barrier that forces visibility of all buffers.
static void recordFullExecutionBarrier(Value commandBuffer, Location loc,
                                       ConversionPatternRewriter &rewriter) {
  rewriter.create<IREE::HAL::CommandBufferExecutionBarrierOp>(
      loc, commandBuffer,
      IREE::HAL::ExecutionStageBitfield::CommandRetire |
          IREE::HAL::ExecutionStageBitfield::Dispatch,
      IREE::HAL::ExecutionStageBitfield::CommandIssue |
          IREE::HAL::ExecutionStageBitfield::Dispatch,
      IREE::HAL::ExecutionBarrierFlagBitfield::None);
}

// Records a dispatch using the given bindings attribute set populated by
// the -iree-hal-materialize-interfaces2 pass.
static void recordInterfaceBindings(Value device, Value commandBuffer,
                                    IREE::Flow::DispatchOp &dispatchOp,
                                    IREE::HAL::InterfaceOp &interfaceOp,
                                    Value executableLayout,
                                    ArrayAttr bindingsAttr,
                                    StreamSchedulingState &schedulingState,
                                    ConversionPatternRewriter &rewriter) {
  // Accumulate a potentially sparse set of push constants.
  // If we had canonicalizers for hal.command_buffer.push_constants then we
  // would instead just emit each constant individually and let that collapse
  // things later on.
  int pushConstantBase = 0;  // always 0 today
  SmallVector<Value> pushConstantValues;
  pushConstantValues.resize(
      interfaceOp.push_constants().getValueOr(APInt(64, 0)).getSExtValue());

  // Accumulate a potentially sparse set of bindings.
  int setOrdinal = 0;  // always 0 today
  SmallVector<IREE::HAL::DescriptorSetBindingValue, 4> bindings;

  auto zeroOffset = schedulingState.lookupOrCreateIndex(0, rewriter);
  auto push_buffer_binding = [&](StringRef bindingName, Value tensorValue) {
    auto bindingOp =
        interfaceOp.lookupSymbol<IREE::HAL::InterfaceBindingOp>(bindingName);
    assert(bindingOp);
    assert(bindingOp.set().getSExtValue() == 0);

    auto bufferRange = schedulingState.lookupTensorBufferRange(tensorValue);
    assert(bufferRange.buffer && "buffer not preallocated");
    assert(bufferRange.length && "buffer has no precomputed size");
    bindings.push_back(
        std::make_tuple(schedulingState.lookupOrCreateIndex(
                            bindingOp.binding().getSExtValue(), rewriter),
                        bufferRange.buffer, zeroOffset, bufferRange.length));
  };

  for (auto bindingAttr : bindingsAttr) {
    if (auto constantStorageAttr =
            bindingAttr.dyn_cast<IREE::HAL::ExConstantStorageAttr>()) {
      auto bindingOp = interfaceOp.lookupSymbol<IREE::HAL::InterfaceBindingOp>(
          constantStorageAttr.binding());
      assert(bindingOp);
      assert(bindingOp.set().getSExtValue() == setOrdinal);
      auto storageBuffer = schedulingState.loadVariable(
          IREE::HAL::BufferType::get(rewriter.getContext()),
          constantStorageAttr.storage(), rewriter);
      bindings.push_back(std::make_tuple(
          schedulingState.lookupOrCreateIndex(
              bindingOp.binding().getSExtValue(), rewriter),
          storageBuffer,
          schedulingState.lookupOrCreateIndex(
              constantStorageAttr.offset().getSExtValue(), rewriter),
          schedulingState.lookupOrCreateIndex(
              constantStorageAttr.length().getSExtValue(), rewriter)));
    } else if (auto pushConstantAttr =
                   bindingAttr.dyn_cast<IREE::HAL::ExPushConstantAttr>()) {
      auto inputValue =
          dispatchOp.operands()[pushConstantAttr.operand().getSExtValue()];
      auto pushConstantValue = rewriter.getRemappedValue(inputValue);
      // Need an explicit index cast to i32 since the
      // CommandBufferPushConstantsOp is intrinsically i32 based.
      if (inputValue.getType().isa<IndexType>()) {
        pushConstantValue = rewriter.create<mlir::IndexCastOp>(
            dispatchOp.getLoc(), rewriter.getIntegerType(32),
            pushConstantValue);
      }
      pushConstantValues[pushConstantAttr.ordinal().getSExtValue()] =
          pushConstantValue;
    } else if (auto operandBufferAttr =
                   bindingAttr.dyn_cast<IREE::HAL::ExOperandBufferAttr>()) {
      auto tensorValue =
          dispatchOp.operands()[operandBufferAttr.operand().getSExtValue()];
      push_buffer_binding(operandBufferAttr.binding(), tensorValue);
    } else if (auto resultBufferAttr =
                   bindingAttr.dyn_cast<IREE::HAL::ExResultBufferAttr>()) {
      auto tensorValue =
          dispatchOp.results()[resultBufferAttr.result().getSExtValue()];
      push_buffer_binding(resultBufferAttr.binding(), tensorValue);
    }
  }

  rewriter.create<IREE::HAL::CommandBufferPushDescriptorSetOp>(
      dispatchOp.getLoc(), commandBuffer, executableLayout,
      schedulingState.lookupOrCreateIndex(setOrdinal, rewriter), bindings);

  if (!pushConstantValues.empty()) {
    rewriter.create<IREE::HAL::CommandBufferPushConstantsOp>(
        dispatchOp.getLoc(), commandBuffer, executableLayout,
        rewriter.getIndexAttr(pushConstantBase), pushConstantValues);
  }
}

// Records a dispatch operation.
static LogicalResult recordDispatch(Value device, Value commandBuffer,
                                    IREE::Flow::DispatchOp &dispatchOp,
                                    StreamSchedulingState &schedulingState,
                                    ConversionPatternRewriter &rewriter) {
  // Get the handle to the executable that is compatible with our device.
  auto executableOp =
      cast<IREE::HAL::ExecutableOp>(SymbolTable::lookupNearestSymbolFrom(
          dispatchOp, dispatchOp.executable()));

  IREE::HAL::TargetBackend::DispatchState dispatchState;
  dispatchState.dispatchOp = dispatchOp;
  dispatchState.executableOp = executableOp;
  dispatchState.device = device;
  dispatchState.commandBuffer = commandBuffer;
  for (auto dim : dispatchOp.workgroup_count()) {
    dispatchState.workgroupCount.push_back(rewriter.getRemappedValue(dim));
  }
  // TODO(benvanik): support extended push constants.
  dispatchState.basePushConstantOffset = 0;

  // Ask each target backend to record their dispatch logic.
  IREE::HAL::DeviceSwitchRewriter switchRewriter(dispatchOp.getLoc(),
                                                 /*resultTypes=*/TypeRange{},
                                                 device, rewriter);
  for (auto targetOp :
       executableOp.getBlock().getOps<IREE::HAL::ExecutableTargetOp>()) {
    for (auto &targetBackend : IREE::HAL::matchTargetBackends(
             {targetOp.target_backend_filter().str()})) {
      auto entryPointOps =
          targetOp.getBlock().getOps<IREE::HAL::ExecutableEntryPointOp>();
      if (entryPointOps.empty()) {
        return dispatchOp.emitOpError() << "need at least one entry point";
      }
      auto entryPointOp = *entryPointOps.begin();
      auto interfaceOp =
          dyn_cast<IREE::HAL::InterfaceOp>(SymbolTable::lookupSymbolIn(
              executableOp, entryPointOp.interfaceAttr()));
      auto executableLayout = schedulingState.lookupExecutableLayout(
          IREE::HAL::ExecutableLayoutType::get(interfaceOp.getContext()),
          interfaceOp.push_constantsAttr(),
          interfaceOp.getExecutableSetLayoutsAttr(), rewriter);

      auto bindingsAttr = dispatchOp->getAttrOfType<ArrayAttr>("hal.bindings");
      assert(bindingsAttr);
      recordInterfaceBindings(device, commandBuffer, dispatchOp, interfaceOp,
                              executableLayout, bindingsAttr, schedulingState,
                              rewriter);

      dispatchState.entryPointOp = entryPointOp;
      dispatchState.interfaceOp = interfaceOp;
      dispatchState.executableLayout = executableLayout;
      if (failed(targetBackend->recordDispatch(
              dispatchOp.getLoc(), dispatchState, switchRewriter))) {
        return dispatchOp.emitError()
               << "unable to record dispatch for target backend "
               << targetBackend->name();
      }
    }
  }
  switchRewriter.build();

  // Full barriers for now as we aren't scheduling things in waves.
  recordFullExecutionBarrier(commandBuffer, dispatchOp.getLoc(), rewriter);
  return success();
}

static LogicalResult recordTensorClone(Value device, Value commandBuffer,
                                       IREE::Flow::TensorCloneOp &cloneOp,
                                       StreamSchedulingState &schedulingState,
                                       ConversionPatternRewriter &rewriter) {
  auto operandBuffer =
      schedulingState.lookupTensorBufferRange(cloneOp.operand());
  auto resultBuffer = schedulingState.lookupTensorBufferRange(cloneOp.result());

  auto zeroOffset = schedulingState.lookupOrCreateIndex(0, rewriter);
  rewriter.create<IREE::HAL::CommandBufferCopyBufferOp>(
      cloneOp.getLoc(), commandBuffer, operandBuffer.buffer, zeroOffset,
      resultBuffer.buffer, zeroOffset, operandBuffer.length);

  // Full barriers for now as we aren't scheduling things.
  recordFullExecutionBarrier(commandBuffer, cloneOp.getLoc(), rewriter);
  return success();
}

// TODO(#5410): make this an aliasing operation in allocateTransientBuffers.
static LogicalResult recordTensorSlice(Value device, Value commandBuffer,
                                       IREE::Flow::TensorSliceOp &sliceOp,
                                       StreamSchedulingState &schedulingState,
                                       ConversionPatternRewriter &rewriter) {
  auto sourceBuffer = schedulingState.lookupTensorBufferRange(sliceOp.source());
  auto resultBuffer = schedulingState.lookupTensorBufferRange(sliceOp.result());

  // TODO(benvanik): use something other than the BufferRange::buffer?
  // This may require us to subview the buffer first.
  auto source = IREE::HAL::TensorRewriteAdaptor::getChecked(
      sliceOp.getLoc(), sliceOp.source(), sourceBuffer.buffer, rewriter);
  auto result = IREE::HAL::TensorRewriteAdaptor::getChecked(
      sliceOp.getLoc(), sliceOp.result(), resultBuffer.buffer, rewriter);
  if (!source.hasValue() || !result.hasValue()) {
    return sliceOp.emitOpError()
           << "cannot create adaptors for tensor slice operands/results";
  }

  // Compute the size of the update range.
  auto startIndices = llvm::to_vector<4>(llvm::map_range(
      sliceOp.start_indices(),
      [&](Value value) { return rewriter.getRemappedValue(value); }));
  auto shapeDims = result->getShapeDims();
  if (!shapeDims) return failure();
  auto sourceRange = source->computeRange(startIndices, *shapeDims);
  if (!sourceRange) return failure();

  // TODO(benvanik): slice left/mid/right, but really just don't do this.
  auto zeroOffset = schedulingState.lookupOrCreateIndex(0, rewriter);
  rewriter.create<IREE::HAL::CommandBufferCopyBufferOp>(
      sliceOp.getLoc(), commandBuffer, sourceBuffer.buffer, sourceRange->offset,
      resultBuffer.buffer, zeroOffset, sourceRange->length);

  // Full barriers for now as we aren't scheduling things.
  // TODO(benvanik): don't add at the end of the command buffer (we could
  // also do a canonicalization step that removed trailing barriers).
  recordFullExecutionBarrier(commandBuffer, sliceOp.getLoc(), rewriter);
  return success();
}

// TODO(#5410): make this an aliasing operation in allocateTransientBuffers.
static LogicalResult recordTensorUpdate(Value device, Value commandBuffer,
                                        IREE::Flow::TensorUpdateOp &updateOp,
                                        StreamSchedulingState &schedulingState,
                                        ConversionPatternRewriter &rewriter) {
  auto updateBuffer =
      schedulingState.lookupTensorBufferRange(updateOp.update());
  auto targetBuffer =
      schedulingState.lookupTensorBufferRange(updateOp.target());

  // TODO(benvanik): use something other than the BufferRange::buffer?
  // This may require us to subview the buffer first.
  auto update = IREE::HAL::TensorRewriteAdaptor::getChecked(
      updateOp.getLoc(), updateOp.update(), updateBuffer.buffer, rewriter);
  auto target = IREE::HAL::TensorRewriteAdaptor::getChecked(
      updateOp.getLoc(), updateOp.target(), targetBuffer.buffer, rewriter);
  if (!update.hasValue() || !target.hasValue()) {
    return updateOp.emitOpError()
           << "cannot create adaptors for tensor update operands/results";
  }

  // Compute the size of the update range.
  auto startIndices = llvm::to_vector<4>(llvm::map_range(
      updateOp.start_indices(),
      [&](Value value) { return rewriter.getRemappedValue(value); }));
  auto shapeDims = update->getShapeDims();
  if (!shapeDims) return failure();
  auto targetRange =
      target->computeRange(startIndices, *update->getShapeDims());
  if (!targetRange) return failure();

  // TODO(benvanik): slice left/mid/right, but really just don't do this.
  auto zeroOffset = schedulingState.lookupOrCreateIndex(0, rewriter);
  rewriter.create<IREE::HAL::CommandBufferCopyBufferOp>(
      updateOp.getLoc(), commandBuffer, updateBuffer.buffer, zeroOffset,
      targetBuffer.buffer, targetRange->offset, targetRange->length);

  // Full barriers for now as we aren't scheduling things.
  // TODO(benvanik): don't add at the end of the command buffer (we could
  // also do a canonicalization step that removed trailing barriers).
  recordFullExecutionBarrier(commandBuffer, updateOp.getLoc(), rewriter);
  return success();
}

static LogicalResult recordStreamCommands(
    Value device, Value commandBuffer, Block &streamBlock,
    StreamSchedulingState &schedulingState,
    ConversionPatternRewriter &rewriter) {
  for (auto &op : streamBlock) {
    if (auto dispatchOp = dyn_cast<IREE::Flow::DispatchOp>(op)) {
      if (failed(recordDispatch(device, commandBuffer, dispatchOp,
                                schedulingState, rewriter))) {
        return failure();
      }
    } else if (auto cloneOp = dyn_cast<IREE::Flow::TensorCloneOp>(op)) {
      if (failed(recordTensorClone(device, commandBuffer, cloneOp,
                                   schedulingState, rewriter))) {
        return failure();
      }
    } else if (auto sliceOp = dyn_cast<IREE::Flow::TensorSliceOp>(op)) {
      if (failed(recordTensorSlice(device, commandBuffer, sliceOp,
                                   schedulingState, rewriter))) {
        return failure();
      }
    } else if (auto updateOp = dyn_cast<IREE::Flow::TensorUpdateOp>(op)) {
      if (failed(recordTensorUpdate(device, commandBuffer, updateOp,
                                    schedulingState, rewriter))) {
        return failure();
      }
    } else if (auto returnOp = dyn_cast<IREE::Flow::ReturnOp>(op)) {
      // No-op; handled by the buffer allocation.
    } else if (isa<ConstantOp>(op)) {
      // HACK: all this code is going away soon.
      auto newOp = rewriter.clone(op);
      op.replaceAllUsesWith(newOp);
    } else if (isa<IREE::HAL::ConstantSubspanOp>(op) ||
               isa<IREE::Flow::TensorReshapeOp>(op)) {
      // No work to perform.
    } else {
      return op.emitOpError() << "unexpected in stream";
    }
  }
  return success();
}

class ExStreamFragmentOpConversion
    : public OpConversionPattern<IREE::Flow::ExStreamFragmentOp> {
 public:
  using OpConversionPattern<
      IREE::Flow::ExStreamFragmentOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::ExStreamFragmentOp streamOp, ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::ExStreamFragmentOp::Adaptor adaptor(
        newOperands, streamOp->getAttrDictionary());

    auto valueAliases = computeValueAliases(streamOp);
    auto livenessIntervals = computeLivenessIntervals(streamOp, valueAliases);

    auto device =
        rewriter.createOrFold<IREE::HAL::ExSharedDeviceOp>(streamOp.getLoc());
    auto allocator =
        rewriter.create<IREE::HAL::DeviceAllocatorOp>(streamOp.getLoc(), device)
            .getResult();
    StreamSchedulingState schedulingState(streamOp.getLoc(), device, allocator,
                                          valueAliases);

    // Map stream captures to their external buffers or SSA values.
    // This covers all of the live-in stream values.
    auto &entryBlock = streamOp.body().front();
    for (int i = 0; i < adaptor.operands().size(); ++i) {
      auto streamValue = entryBlock.getArgument(i);
      auto bufferValue = adaptor.operands()[i];
      if (auto shapedType = streamValue.getType().dyn_cast<TensorType>()) {
        BufferRange bufferRange;
        if (bufferValue.getType().isa<IREE::HAL::BufferViewType>()) {
          bufferRange = BufferRange{
              rewriter.createOrFold<IREE::HAL::BufferViewBufferOp>(
                  streamOp.getLoc(),
                  IREE::HAL::BufferType::get(rewriter.getContext()),
                  bufferValue),
              schedulingState.lookupOrComputeSize(streamValue, rewriter)};
        } else {
          bufferRange = BufferRange{
              bufferValue,
              schedulingState.lookupOrComputeSize(streamValue, rewriter)};
        }
        if (failed(schedulingState.mapTensorToBufferRange(streamValue,
                                                          bufferRange))) {
          return streamOp.emitOpError()
                 << "tensor was mapped to multiple buffer ranges";
        }

      } else {
        rewriter.replaceUsesOfBlockArgument(streamValue, bufferValue);
      }
    }

    // Allocate buffers for values that escape the stream via return.
    // These may alias input buffers above such as when an input is returned or
    // a return value is tied.
    SmallVector<Value> outputBuffers;
    if (failed(allocateOutputBuffers(streamOp, schedulingState, rewriter,
                                     outputBuffers))) {
      return failure();
    }

    // Allocate all of the transient buffers used entirely within the stream.
    // These all end up aliased from a single slab allocation and use the
    // computed liveness information to know the lifetime intervals. Note that
    // after we perform this allocation we can no longer safely rearrange the
    // ops as buffers will start to alias. All reordering must have happened
    // prior to this conversion.
    if (failed(allocateTransientBuffers(streamOp, livenessIntervals,
                                        schedulingState, rewriter))) {
      return failure();
    }

    // Allocate and begin the command buffer.
    // In a real version we would want to pick the device based on the placement
    // information attached to the stream.
    // TODO(benvanik): choose buffer mode/category based on stream commands.
    // NOTE: we are not doing any overlapping work today and can always allow
    // inline execution.
    auto mode = IREE::HAL::CommandBufferModeBitfield::OneShot |
                IREE::HAL::CommandBufferModeBitfield::AllowInlineExecution;
    auto category = IREE::HAL::CommandCategoryBitfield::Dispatch |
                    IREE::HAL::CommandCategoryBitfield::Transfer;
    auto commandBuffer =
        rewriter.createOrFold<IREE::HAL::CommandBufferCreateOp>(
            streamOp.getLoc(),
            IREE::HAL::CommandBufferType::get(rewriter.getContext()), device,
            mode, category);
    rewriter.create<IREE::HAL::CommandBufferBeginOp>(streamOp.getLoc(),
                                                     commandBuffer);

    // Record all of the commands into the command buffer.
    if (failed(recordStreamCommands(device, commandBuffer, entryBlock,
                                    schedulingState, rewriter))) {
      return failure();
    }

    // End and submit the command buffer.
    // In a real version we'd want to setup a semaphore chain instead of
    // submitting and waiting.
    rewriter.create<IREE::HAL::CommandBufferEndOp>(streamOp.getLoc(),
                                                   commandBuffer);
    rewriter.create<IREE::HAL::ExSubmitAndWaitOp>(streamOp.getLoc(), device,
                                                  commandBuffer);

    // It's annoying but we need to do this replacement at the very end as
    // otherwise we lose access to the original values (which we need for
    // shape information).
    for (int i = 0; i < adaptor.operands().size(); ++i) {
      if (adaptor.operands()[i].getType().isa<IREE::HAL::BufferType>()) {
        rewriter.replaceUsesOfBlockArgument(entryBlock.getArgument(i),
                                            adaptor.operands()[i]);
      }
    }

    rewriter.replaceOp(streamOp, outputBuffers);
    return success();
  }
};

}  // namespace

void populateFlowStreamToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &converter) {
  patterns.insert<ExStreamFragmentOpConversion>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
