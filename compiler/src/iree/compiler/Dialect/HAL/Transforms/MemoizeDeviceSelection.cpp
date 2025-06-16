// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_MEMOIZEDEVICESELECTIONPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-memoize-device-selection
//===----------------------------------------------------------------------===//

struct SelectOpOperands {
  SmallVector<StringAttr, 2> deviceSymbols;
  SmallVector<IntegerAttr, 2> queueAffinities;
  IntegerAttr memoryType;
  IntegerAttr bufferUsage;

  // Ideally the SelectOpOperands should be a valid key to the DenseMap, so we
  // could use it directly. However, we just take the shortcut of
  // converting to an ArrayAttr and using that as the key.
  ArrayAttr toArrayAttr(MLIRContext *context) const {
    SmallVector<Attribute> components;
    for (auto deviceSymbol : deviceSymbols) {
      components.push_back(FlatSymbolRefAttr::get(deviceSymbol));
    }
    for (auto queueAffinity : queueAffinities) {
      components.push_back(queueAffinity);
    }
    components.push_back(memoryType);
    components.push_back(bufferUsage);
    return ArrayAttr::get(context, components);
  }
};

static LogicalResult memoizeAllocatorSelectOp(
    const SelectOpOperands &selectOpOperands,
    const SmallVectorImpl<IREE::HAL::AllocatorSelectOp> &selectOps,
    SymbolTable &symbolTable) {
  // The ops are in module order.
  auto firstSelectOp = selectOps.front();
  auto selectOpLocs = llvm::map_to_vector(
      selectOps, [](auto selectOp) { return selectOp.getLoc(); });
  auto fusedLoc = FusedLoc::get(firstSelectOp.getContext(), selectOpLocs);

  // Since all globals must be initialized prior to first use we will insert
  // the new global and initializer immediately prior to the function containing
  // the op. Subsequent select ops will be later in the module.
  OpBuilder moduleBuilder(firstSelectOp.getContext());
  auto firstFuncOp =
      firstSelectOp->getParentOfType<mlir::FunctionOpInterface>();
  moduleBuilder.setInsertionPoint(firstFuncOp);

  // Pick a unique name prefix for the new globals.
  std::string globalNamePrefix = "__allocator_select";

  // Insert globals for query results.
  auto selectedDeviceGlobalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
      fusedLoc, globalNamePrefix + "_device", /*isMutable=*/false,
      moduleBuilder.getType<IREE::HAL::DeviceType>());
  symbolTable.insert(selectedDeviceGlobalOp);
  selectedDeviceGlobalOp.setPrivate();
  auto selectedQueueAffinityGlobalOp =
      moduleBuilder.create<IREE::Util::GlobalOp>(
          fusedLoc, globalNamePrefix + "_affinity", /*isMutable=*/false,
          moduleBuilder.getIntegerType(64));
  symbolTable.insert(selectedQueueAffinityGlobalOp);
  selectedQueueAffinityGlobalOp.setPrivate();

  // Build initializer for the globals.
  auto initializerOp =
      moduleBuilder.create<IREE::Util::InitializerOp>(fusedLoc);
  {
    auto initializerBuilder =
        OpBuilder::atBlockBegin(initializerOp.addEntryBlock());

    SmallVector<Value> deviceValues;
    SmallVector<Value> queueAffinityValues;
    for (auto [deviceSymbol, queueAffinity] :
         llvm::zip_equal(selectOpOperands.deviceSymbols,
                         selectOpOperands.queueAffinities)) {
      auto deviceGlobalOp =
          symbolTable.lookupNearestSymbolFrom<IREE::Util::GlobalOpInterface>(
              firstSelectOp, deviceSymbol);
      Value deviceValue =
          deviceGlobalOp.createLoadOp(fusedLoc, initializerBuilder)
              .getLoadedGlobalValue();
      deviceValues.push_back(deviceValue);
      queueAffinityValues.push_back(
          initializerBuilder.create<arith::ConstantOp>(fusedLoc,
                                                       queueAffinity));
    }

    Value memoryTypeValue = initializerBuilder.create<IREE::HAL::MemoryTypeOp>(
        fusedLoc, cast<MemoryTypeBitfieldAttr>(selectOpOperands.memoryType));

    Value bufferUsageValue =
        initializerBuilder.create<IREE::HAL::BufferUsageOp>(
            fusedLoc,
            cast<BufferUsageBitfieldAttr>(selectOpOperands.bufferUsage));

    auto newSelectOp = initializerBuilder.create<IREE::HAL::AllocatorSelectOp>(
        fusedLoc, deviceValues, queueAffinityValues, memoryTypeValue,
        bufferUsageValue);

    selectedDeviceGlobalOp.createStoreOp(
        fusedLoc, newSelectOp.getSelectedDevice(), initializerBuilder);
    selectedQueueAffinityGlobalOp.createStoreOp(
        fusedLoc, newSelectOp.getSelectedQueueAffinity(), initializerBuilder);
    initializerBuilder.create<IREE::Util::ReturnOp>(fusedLoc);
  }

  // Replace all select ops with global loads.
  for (auto selectOp : selectOps) {
    OpBuilder selectBuilder(selectOp);
    Value selectedDeviceValue =
        selectedDeviceGlobalOp.createLoadOp(selectOp.getLoc(), selectBuilder)
            .getLoadedGlobalValue();
    Value selectedQueueAffinityValue =
        selectedQueueAffinityGlobalOp
            .createLoadOp(selectOp.getLoc(), selectBuilder)
            .getLoadedGlobalValue();
    selectOp.replaceAllUsesWith(ValueRange{
        selectedDeviceValue,
        selectedQueueAffinityValue,
    });
    selectOp.erase();
  }

  return success();
}

struct MemoizeDeviceSelectionPass
    : public IREE::HAL::impl::MemoizeDeviceSelectionPassBase<
          MemoizeDeviceSelectionPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    // Gather all select ops in the program and bucket by unique key.
    // For each bucket the first op will be the first that appears in the
    // module for that given bucket.
    DenseMap<ArrayAttr, std::pair<SelectOpOperands,
                                  SmallVector<IREE::HAL::AllocatorSelectOp>>>
        selectOps;
    for (auto callableOp : moduleOp.getOps<mlir::CallableOpInterface>()) {
      // TODO(benvanik): an interface for when we have other select ops. For now
      // we only have AllocatorSelectOp.
      callableOp.walk([&](IREE::HAL::AllocatorSelectOp selectOp) {
        // Build unique key from device symbols, queue affinities, memory type,
        // and buffer usage. If we fail to determine any of these values,
        // we skip the op as we cannot be sure the key is unique.
        SelectOpOperands selectOpOperands;

        // Add device symbols.
        for (Value device : selectOp.getDevices()) {
          auto deviceGlobals = deviceAnalysis.lookupDeviceGlobals(device);
          // If we cannot find a device or we have more than one, we skip the
          // op.
          if (!deviceGlobals || deviceGlobals->size() != 1) {
            return;
          }
          selectOpOperands.deviceSymbols.push_back(
              deviceGlobals->front().getGlobalName());
        }

        // Add queue affinities.
        for (Value queueAffinity : selectOp.getQueueAffinities()) {
          IntegerAttr queueAffinityAttr;
          if (!matchPattern(queueAffinity, m_Constant(&queueAffinityAttr))) {
            return;
          }
          selectOpOperands.queueAffinities.push_back(queueAffinityAttr);
        }

        // Add memory type.
        if (!matchPattern(selectOp.getMemoryTypes(),
                          m_Constant(&selectOpOperands.memoryType))) {
          return;
        }

        // Add buffer usage.
        if (!matchPattern(selectOp.getBufferUsage(),
                          m_Constant(&selectOpOperands.bufferUsage))) {
          return;
        }

        auto &selectOpBucket =
            selectOps[selectOpOperands.toArrayAttr(moduleOp.getContext())];
        selectOpBucket.first = std::move(selectOpOperands);
        selectOpBucket.second.push_back(selectOp);
      });
    }
    if (selectOps.empty()) {
      // Nothing to do.
      markAllAnalysesPreserved();
      return;
    }

    // Insert globals/an initializer/swap ops with lookups.
    for (auto [key, allOps] : selectOps) {
      if (failed(memoizeAllocatorSelectOp(allOps.first, allOps.second,
                                          symbolTable))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
