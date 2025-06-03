// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_MEMOIZEDEVICESELECTIONPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-memoize-device-selection
//===----------------------------------------------------------------------===//

static LogicalResult memoizeAllocatorSelectOp(
    SmallVectorImpl<IREE::HAL::AllocatorSelectAttrOp> &selectOps,
    SymbolTable &symbolTable) {
  // The ops are in module order.
  auto firstSelectOp = selectOps.front();
  auto selectOpLocs = llvm::map_to_vector(
      selectOps, [](auto selectOp) { return selectOp.getLoc(); });
  auto fusedLoc = FusedLoc::get(firstSelectOp.getContext(), selectOpLocs);

  // Gather the globals for each device and the queue affinity masks 1:1.
  SmallVector<IREE::Util::GlobalOpInterface> deviceGlobalOps;
  SmallVector<int64_t> queueMasks;
  auto optimalSetAttr = firstSelectOp.getOptimalSetAttr();
  for (auto affinityAttr : optimalSetAttr.getAffinities()) {
    auto deviceAffinityAttr =
        dyn_cast<IREE::HAL::DeviceAffinityAttr>(affinityAttr);
    if (!deviceAffinityAttr) {
      return firstSelectOp.emitError()
             << "has invalid affinity " << affinityAttr
             << "; expected only #hal.device.affinity";
    }
    auto deviceGlobalOp =
        symbolTable.lookupNearestSymbolFrom<IREE::Util::GlobalOpInterface>(
            firstSelectOp, deviceAffinityAttr.getDevice());
    deviceGlobalOps.push_back(deviceGlobalOp);
    queueMasks.push_back(deviceAffinityAttr.getQueueMask());
  }

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
    SmallVector<Value> queueMaskValues;
    for (auto [deviceGlobalOp, queueMask] :
         llvm::zip_equal(deviceGlobalOps, queueMasks)) {
      deviceValues.push_back(
          deviceGlobalOp.createLoadOp(fusedLoc, initializerBuilder)
              .getLoadedGlobalValue());
      queueMaskValues.push_back(initializerBuilder.create<arith::ConstantIntOp>(
          fusedLoc, queueMask, 64));
    }
    Value memoryTypesValue = initializerBuilder.create<arith::ConstantIntOp>(
        fusedLoc,
        MemoryTypeOp::getTypeValue(firstSelectOp.getMemoryTypes()).value(), 32);
    Value bufferUsageValue = initializerBuilder.create<arith::ConstantIntOp>(
        fusedLoc,
        BufferUsageOp::getUsageValue(firstSelectOp.getBufferUsage()).value(),
        32);
    auto selectOp = initializerBuilder.create<IREE::HAL::AllocatorSelectOp>(
        fusedLoc, deviceValues, queueMaskValues, memoryTypesValue,
        bufferUsageValue);
    selectedDeviceGlobalOp.createStoreOp(fusedLoc, selectOp.getSelectedDevice(),
                                         initializerBuilder);
    selectedQueueAffinityGlobalOp.createStoreOp(
        fusedLoc, selectOp.getSelectedQueueAffinity(), initializerBuilder);
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

    // Gather all select ops in the program and bucket by unique key.
    // For each bucket the first op will be the first that appears in the
    // module for that given bucket.
    DenseMap<Attribute, SmallVector<IREE::HAL::AllocatorSelectAttrOp>>
        selectOps;
    for (auto callableOp : moduleOp.getOps<mlir::CallableOpInterface>()) {
      // TODO(benvanik): an interface for when we have other select ops. For now
      // we only have AllocatorSelectAttrOp.
      callableOp.walk([&](IREE::HAL::AllocatorSelectAttrOp selectOp) {
        auto fullKey = ArrayAttr::get(
            moduleOp.getContext(),
            {
                selectOp.getOptimalSetAttr(),
                MemoryTypeOp::getTypeAttr(selectOp.getMemoryTypes()),
                BufferUsageOp::getUsageAttr(selectOp.getBufferUsage()),
            });
        selectOps[fullKey].push_back(selectOp);
      });
    }
    if (selectOps.empty()) {
      // Nothing to do.
      markAllAnalysesPreserved();
      return;
    }

    // Insert globals/an initializer/swap ops with lookups.
    for (auto [key, allOps] : selectOps) {
      if (failed(memoizeAllocatorSelectOp(allOps, symbolTable))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
