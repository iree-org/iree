// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
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

static LogicalResult memoizeAllocatorSelectOp(
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

    IRMapping mapping;

    // Clone all operands
    for (Value operand : firstSelectOp.getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      initializerBuilder.clone(*defOp, mapping);
    }

    // Clone the select op itself using the mapping
    auto clonedSelectOp = cast<IREE::HAL::AllocatorSelectOp>(
        initializerBuilder.clone(*firstSelectOp.getOperation(), mapping));

    selectedDeviceGlobalOp.createStoreOp(
        fusedLoc, clonedSelectOp.getSelectedDevice(), initializerBuilder);
    selectedQueueAffinityGlobalOp.createStoreOp(
        fusedLoc, clonedSelectOp.getSelectedQueueAffinity(),
        initializerBuilder);
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
    llvm::StringMap<SmallVector<IREE::HAL::AllocatorSelectOp>> selectOps;
    for (auto callableOp : moduleOp.getOps<mlir::CallableOpInterface>()) {
      // TODO(benvanik): an interface for when we have other select ops. For now
      // we only have AllocatorSelectOp.
      callableOp.walk([&](IREE::HAL::AllocatorSelectOp selectOp) {
        // Build unique key from device symbols, queue affinities, memory type,
        // and buffer usage IF we fail to determine any of these values as
        // constants, we skip the op
        std::string key;

        // Add device symbols
        for (Value device : selectOp.getDevices()) {
          if (auto globalLoadOp =
                  device.getDefiningOp<IREE::Util::GlobalLoadOp>()) {
            key += globalLoadOp.getGlobal().str() + "_";
          } else {
            return;
          }
        }

        // Add queue affinities
        for (Value queueAffinity : selectOp.getQueueAffinities()) {
          if (auto constOp =
                  queueAffinity.getDefiningOp<arith::ConstantIntOp>()) {
            key += std::to_string(constOp.value()) + "_";
          } else {
            return;
          }
        }

        // Add memory type
        if (std::optional<int32_t> memoryType =
                IREE::HAL::MemoryTypeOp::getTypeValue(
                    selectOp.getMemoryTypes())) {
          key += std::to_string(*memoryType) + "_";
        } else {
          return;
        }

        // Add buffer usage
        if (std::optional<int32_t> bufferUsage =
                IREE::HAL::BufferUsageOp::getUsageValue(
                    selectOp.getBufferUsage())) {
          key += std::to_string(*bufferUsage);
        } else {
          return;
        }

        selectOps[key].push_back(selectOp);
      });
    }
    if (selectOps.empty()) {
      // Nothing to do.
      markAllAnalysesPreserved();
      return;
    }

    // Insert globals/an initializer/swap ops with lookups.
    for (const auto &allOps : selectOps) {
      if (failed(memoizeAllocatorSelectOp(allOps.getValue(), symbolTable))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
