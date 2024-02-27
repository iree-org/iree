// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"

#include "iree/compiler/Dialect/HAL/Analysis/Attributes/DeviceGlobalPVS.h"
#include "iree/compiler/Dialect/HAL/Analysis/Attributes/DeviceTargetPVS.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// DeviceAnalysis
//===----------------------------------------------------------------------===//

DeviceAnalysis::DeviceAnalysis(Operation *rootOp)
    : explorer(rootOp, TraversalAction::SHALLOW), solver(explorer, allocator) {
  explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
      TraversalAction::RECURSE);
  explorer.setOpAction<mlir::scf::ForOp>(TraversalAction::RECURSE);
  explorer.setOpAction<mlir::scf::IfOp>(TraversalAction::RECURSE);
  explorer.setOpAction<mlir::scf::WhileOp>(TraversalAction::RECURSE);
  // Ignore the contents of executables (linalg goo, etc).
  explorer.setOpAction<IREE::HAL::ExecutableOp>(TraversalAction::IGNORE);
  explorer.initialize();
}

DeviceAnalysis::~DeviceAnalysis() = default;

LogicalResult DeviceAnalysis::run() {
  // TODO(multi-device): remove this fallback path when device globals are fully
  // plumbed through. Today we still have inputs with the hal.device.targets
  // attribute.
  if (auto targetsAttr = explorer.getRootOp()->getAttrOfType<ArrayAttr>(
          "hal.device.targets")) {
    if (!targetsAttr.empty()) {
      defaultDeviceSet = DeviceSet(targetsAttr);
    }
  }

  // Initialize device globals (in declaration order).
  for (auto globalOp : explorer.getRootOp()
                           ->getRegion(0)
                           .getOps<IREE::Util::GlobalOpInterface>()) {
    auto globalType = globalOp.getGlobalType();
    if (isa<IREE::HAL::DeviceType>(globalType)) {
      solver.getOrCreateElementFor<DeviceTargetGlobalPVS>(
          Position::forOperation(globalOp));
      deviceGlobals.push_back(globalOp);
    }
  }

  // Initialize all SSA values so we can do just with trivial search.
  explorer.walkValuesOfType<IREE::HAL::DeviceType>([&](Value value) {
    solver.getOrCreateElementFor<DeviceGlobalValuePVS>(
        Position::forValue(value));
    solver.getOrCreateElementFor<DeviceTargetValuePVS>(
        Position::forValue(value));
    return WalkResult::advance();
  });

  return solver.run();
}

std::optional<SetVector<IREE::Util::GlobalOpInterface>>
DeviceAnalysis::lookupDeviceGlobals(Value deviceValue) {
  auto globalPVS = solver.lookupElementFor<DeviceGlobalValuePVS>(
      Position::forValue(deviceValue));
  if (!globalPVS || !globalPVS->isValidState() ||
      globalPVS->isUndefContained()) {
    return std::nullopt;
  }
  SetVector<IREE::Util::GlobalOpInterface> globalOps;
  for (auto globalOp : globalPVS->getAssumedSet()) {
    globalOps.insert(globalOp);
  }
  return globalOps;
}

std::optional<DeviceSet>
DeviceAnalysis::lookupDeviceTargets(Value deviceValue) {
  auto valuePVS = solver.lookupElementFor<DeviceTargetValuePVS>(
      Position::forValue(deviceValue));
  if (!valuePVS || !valuePVS->isValidState() || valuePVS->isUndefContained()) {
    return defaultDeviceSet;
  }
  return DeviceSet(valuePVS->getAssumedSet());
}

} // namespace mlir::iree_compiler::IREE::HAL
