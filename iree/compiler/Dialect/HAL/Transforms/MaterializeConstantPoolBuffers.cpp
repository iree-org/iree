// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class MaterializeConstantPoolBuffersPass
    : public PassWrapper<MaterializeConstantPoolBuffersPass,
                         OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<IREE::Util::UtilDialect>();
    registry.insert<IREE::HAL::HALDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-materialize-constant-pool-buffers";
  }

  StringRef getDescription() const override {
    return "Materializes runtime buffers for constant pools.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Today we simply materialize a !hal.buffer variable for each storage
    // buffer and initialize it in a naive way. Really we should be aggregating
    // all constant pools and issuing a command buffer on a DMA queue to upload
    // everything but this is a start that works on unified memory systems ok.
    // TODO(benvanik): command buffer-based DMA/uploads.
    //
    // We also handle the specific constant types directly in this file. Instead
    // we could synthesize pseudo ops (hal.constant.populate %buffer, ...) that
    // then were rewritten to the logic during conversion. This would let us
    // more easily add new types (including things like target-specific constant
    // types).
    SymbolTable moduleSymbolTable(moduleOp);
    auto poolOps = llvm::to_vector<4>(moduleOp.getOps<ConstantPoolOp>());
    for (auto poolOp : poolOps) {
      auto insertionPoint = ++Block::iterator(poolOp);

      // 1:1 storage to runtime buffers.
      for (auto storageOp : poolOp.getOps<ConstantStorageOp>()) {
        makeStorageBufferRuntimeGlobal(poolOp, storageOp, moduleSymbolTable,
                                       insertionPoint);
      }

      // We currently put all splats on their own so that we are always able to
      // map the storage buffers above as read-only.
      auto splatOps = llvm::to_vector<4>(poolOp.getOps<ConstantPoolSplatOp>());
      if (!splatOps.empty()) {
        makeSplatRuntimeGlobal(poolOp, splatOps, moduleSymbolTable,
                               insertionPoint);
      }
    }
  }

 private:
  // Creates a runtime buffer into which the storage buffer will be mapped or
  // uploaded.
  void makeStorageBufferRuntimeGlobal(ConstantPoolOp poolOp,
                                      ConstantStorageOp storageOp,
                                      SymbolTable &moduleSymbolTable,
                                      Block::iterator insertionPoint) {
    auto *context = poolOp.getContext();
    auto variableName =
        (poolOp.getName() + storageOp.getName() + "_buffer").str();
    auto variableType = IREE::HAL::BufferType::get(context);
    auto globalOp = OpBuilder(context).create<IREE::Util::GlobalOp>(
        storageOp.getLoc(), variableName, /*isMutable=*/false, variableType);
    moduleSymbolTable.insert(globalOp, insertionPoint);
    globalOp.setPrivate();

    // Find all the spans in the pool that map into this storage buffer so that
    // we can update them with their runtime offsets. Note that since we are
    // uploading 1:1 today all the offsets are the same as their storage ones.
    auto variableSymRef = SymbolRefAttr::get(context, globalOp.getName());
    for (auto spanOp : poolOp.getOps<ConstantPoolSpanOp>()) {
      if (spanOp.storage_buffer().getLeafReference() != storageOp.getName()) {
        continue;
      }
      spanOp.runtime_bufferAttr(variableSymRef);
      spanOp.runtime_rangeAttr(spanOp.storage_range());
    }

    makeStorageBufferRuntimeInitializer(globalOp, storageOp,
                                        poolOp.buffer_constraints());
  }

  // Creates an initializer function that unpacks the given storage op into a
  // new buffer.
  void makeStorageBufferRuntimeInitializer(
      IREE::Util::GlobalOp globalOp, ConstantStorageOp storageOp,
      BufferConstraintsAttr bufferConstraints) {
    auto loc = globalOp.getLoc();
    auto *context = storageOp.getContext();
    OpBuilder moduleBuilder(context);
    moduleBuilder.setInsertionPointAfter(globalOp);
    auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);

    auto builder = OpBuilder::atBlockBegin(initializerOp.addEntryBlock());

    // HACK: use default allocator.
    auto deviceValue = builder.createOrFold<IREE::HAL::ExSharedDeviceOp>(loc);
    auto allocatorValue =
        builder.createOrFold<IREE::HAL::DeviceAllocatorOp>(loc, deviceValue);

    // Today we always map the buffer directly. We should be using a device
    // switch to schedule the upload if needed.
    // TODO(benvanik): allocate based on usage tracking.
    auto sourceValue = builder.createOrFold<IREE::HAL::ConstantStorageLookupOp>(
        loc, IREE::Util::ByteBufferType::get(context),
        builder.getSymbolRefAttr(
            storageOp->getParentOfType<ConstantPoolOp>().getName(),
            {builder.getSymbolRefAttr(storageOp)}));
    auto offsetValue = builder.createOrFold<mlir::ConstantIndexOp>(loc, 0);
    auto storageValueAttr =
        storageOp.value().cast<IREE::Util::SerializableAttrInterface>();
    uint64_t runtimeLength =
        align(storageValueAttr.getStorageSize(),
              bufferConstraints.min_buffer_range_alignment());
    auto lengthValue =
        builder.createOrFold<mlir::ConstantIndexOp>(loc, runtimeLength);
    auto memoryType = IREE::HAL::MemoryTypeBitfield::DeviceLocal |
                      IREE::HAL::MemoryTypeBitfield::HostVisible;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::Constant |
                       IREE::HAL::BufferUsageBitfield::All;
    auto bufferValue = builder.createOrFold<IREE::HAL::AllocatorMapOp>(
        loc, IREE::HAL::BufferType::get(context), allocatorValue, memoryType,
        bufferUsage, sourceValue, offsetValue, lengthValue);

    builder.create<IREE::Util::GlobalStoreOp>(loc, bufferValue,
                                              globalOp.getName());

    builder.create<IREE::Util::InitializerReturnOp>(loc);
  }

  // Creates a runtime buffer for the given constant pool splats and constructs
  // its initializer to fill the contents.
  void makeSplatRuntimeGlobal(ConstantPoolOp poolOp,
                              ArrayRef<ConstantPoolSplatOp> splatOps,
                              SymbolTable &moduleSymbolTable,
                              Block::iterator insertionPoint) {
    auto *context = poolOp.getContext();

    // TODO(benvanik): we don't need host-visible here as we could require that
    // all reads go through staging. When we want to support devices with
    // unmappable memory we'll need to adjust this. Usage analysis on whether
    // the buffer is ever read back or only used on device will help determine
    // things.
    auto variableType = IREE::HAL::BufferType::get(context);

    auto globalLoc =
        FusedLoc::get(context, llvm::to_vector<8>(llvm::map_range(
                                   splatOps, [](ConstantPoolSplatOp splatOp) {
                                     return splatOp.getLoc();
                                   })));
    auto variableName = (poolOp.getName() + "_splats").str();
    auto globalOp = OpBuilder(context).create<IREE::Util::GlobalOp>(
        globalLoc, variableName, /*isMutable=*/false, variableType);
    moduleSymbolTable.insert(globalOp, insertionPoint);
    globalOp.setPrivate();

    // Compute the ranges for all the splats at runtime and the required buffer
    // size based on the constraints provided.
    auto bufferConstraints = poolOp.buffer_constraints();
    auto variableSymRef = SymbolRefAttr::get(context, globalOp.getName());
    uint64_t bufferLength = 0;
    for (auto splatOp : poolOp.getOps<ConstantPoolSplatOp>()) {
      uint64_t splatOffset =
          align(bufferLength, bufferConstraints.min_buffer_offset_alignment());
      uint64_t unpaddedLength =
          getRoundedElementByteWidth(
              splatOp.value().getType().getElementType()) *
          splatOp.value().getNumElements();
      uint64_t splatLength =
          align(unpaddedLength, bufferConstraints.min_buffer_range_alignment());
      splatOp.runtime_bufferAttr(variableSymRef);
      splatOp.runtime_rangeAttr(
          IREE::Util::ByteRangeAttr::get(context, splatOffset, splatLength));
      bufferLength = splatOffset + splatLength;
    }

    // TODO(benvanik): if we spill here we'll need to create more buffers. We
    // could flip this loop inside out and first allocate the splats.
    if (bufferLength > bufferConstraints.max_buffer_range().getZExtValue()) {
      globalOp.emitError()
          << "constant splat buffer length " << bufferLength
          << " spills max buffer range of "
          << bufferConstraints.max_buffer_range().getZExtValue()
          << " - contents may not be accessible at runtime";
    }

    makeSplatRuntimeInitializer(globalOp, splatOps, bufferLength);
  }

  // Creates an initializer function that allocates the runtime buffer and
  // splats the values into it.
  void makeSplatRuntimeInitializer(IREE::Util::GlobalOp globalOp,
                                   ArrayRef<ConstantPoolSplatOp> splatOps,
                                   uint64_t bufferLength) {
    auto loc = globalOp.getLoc();
    auto *context = loc.getContext();
    OpBuilder moduleBuilder(context);
    moduleBuilder.setInsertionPointAfter(globalOp);
    auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);

    auto builder = OpBuilder::atBlockBegin(initializerOp.addEntryBlock());

    // HACK: use default allocator.
    auto deviceValue = builder.createOrFold<IREE::HAL::ExSharedDeviceOp>(loc);
    auto allocatorValue =
        builder.createOrFold<IREE::HAL::DeviceAllocatorOp>(loc, deviceValue);

    // Allocate buffer with empty contents.
    auto memoryType = IREE::HAL::MemoryTypeBitfield::DeviceLocal |
                      IREE::HAL::MemoryTypeBitfield::HostVisible;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::Constant |
                       IREE::HAL::BufferUsageBitfield::All;
    auto allocationSizeValue =
        builder.createOrFold<mlir::ConstantIndexOp>(loc, bufferLength);
    auto bufferValue = builder.createOrFold<IREE::HAL::AllocatorAllocateOp>(
        loc, IREE::HAL::BufferType::get(context), allocatorValue, memoryType,
        bufferUsage, allocationSizeValue);

    // Fill the buffers (memset).
    // We do this with a command buffer so that we can allow the device to
    // fill them in asynchronously and without memory mapping.
    auto commandBufferValue =
        builder.createOrFold<IREE::HAL::CommandBufferCreateOp>(
            loc, IREE::HAL::CommandBufferType::get(context), deviceValue,
            IREE::HAL::CommandBufferModeBitfield::OneShot |
                IREE::HAL::CommandBufferModeBitfield::AllowInlineExecution,
            IREE::HAL::CommandCategoryBitfield::Transfer);
    builder.create<IREE::HAL::CommandBufferBeginOp>(loc, commandBufferValue);
    for (auto splatOp : splatOps) {
      auto runtimeRange = splatOp.runtime_range().getValue();
      auto offsetValue = builder.createOrFold<mlir::ConstantIndexOp>(
          splatOp.getLoc(), runtimeRange.getOffset());
      auto lengthValue = builder.createOrFold<mlir::ConstantIndexOp>(
          splatOp.getLoc(), runtimeRange.getLength());
      uint32_t pattern = makePatternFromSplatValue(
          splatOp.value().cast<SplatElementsAttr>().getSplatValue());
      auto patternValue = builder.createOrFold<mlir::ConstantIntOp>(
          loc, static_cast<int64_t>(pattern), 32);
      builder.create<IREE::HAL::CommandBufferFillBufferOp>(
          splatOp.getLoc(), commandBufferValue, bufferValue, offsetValue,
          lengthValue, patternValue);
    }
    builder.create<IREE::HAL::CommandBufferEndOp>(loc, commandBufferValue);
    builder.create<IREE::HAL::ExSubmitAndWaitOp>(loc, deviceValue,
                                                 commandBufferValue);

    builder.create<IREE::Util::GlobalStoreOp>(loc, bufferValue,
                                              globalOp.getName());

    builder.create<IREE::Util::InitializerReturnOp>(loc);
  }

  // Makes a 4-byte pattern from a splat value for use at runtime.
  // Asserts if the pattern cannot be constructed (not 4-byte compatible). TBD.
  // TODO(benvanik): support 8-byte fill patterns (via fallback executable).
  uint32_t makePatternFromSplatValue(Attribute elementAttr) {
    assert(elementAttr.getType().getIntOrFloatBitWidth() <= 32);  // i64/f64 TBD
    if (auto intAttr = elementAttr.dyn_cast<IntegerAttr>()) {
      return static_cast<uint32_t>(
          APInt::getSplat(32, intAttr.getValue()).getZExtValue());
    } else if (auto fltAttr = elementAttr.dyn_cast<FloatAttr>()) {
      return static_cast<uint32_t>(
          APInt::getSplat(32, fltAttr.getValue().bitcastToAPInt())
              .getZExtValue());
    }
    assert(false && "unsupported splat type");
    return 0;
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createMaterializeConstantPoolBuffersPass() {
  return std::make_unique<MaterializeConstantPoolBuffersPass>();
}

static PassRegistration<MaterializeConstantPoolBuffersPass> pass;

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
