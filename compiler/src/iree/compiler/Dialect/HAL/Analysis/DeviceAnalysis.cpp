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
    : explorer(rootOp, TraversalAction::RECURSE), solver(explorer, allocator) {
  explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
      TraversalAction::RECURSE);
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

// Sorts |globalOps| in the natural affinity sort order.
// We unfortunately have to do this as the PVS elements we source from are
// unsorted.
static void
sortGlobalOps(SmallVectorImpl<IREE::Util::GlobalOpInterface> &globalOps) {
  // HACK: this should probably do a type id ordering followed by a
  // type-specific ordering (interface compare method?). We just need this to be
  // stable as the globals come from multiple DenseSets that have run-to-run
  // ordering variance. This is very inefficient but is only used when there are
  // multiple possible devices and we try to avoid that anyway.
  if (globalOps.size() <= 1) {
    return;
  }
  llvm::stable_sort(globalOps, [](IREE::Util::GlobalOpInterface lhs,
                                  IREE::Util::GlobalOpInterface rhs) {
    std::string lhsStr;
    llvm::raw_string_ostream lhsStream(lhsStr);
    lhs.print(lhsStream);
    std::string rhsStr;
    llvm::raw_string_ostream rhsStream(rhsStr);
    rhs.print(rhsStream);
    return lhsStr < rhsStr;
  });
}

std::optional<SmallVector<IREE::Util::GlobalOpInterface>>
DeviceAnalysis::lookupDeviceGlobals(Value deviceValue) {
  auto globalPVS = solver.lookupElementFor<DeviceGlobalValuePVS>(
      Position::forValue(deviceValue));
  if (!globalPVS || !globalPVS->isValidState() ||
      globalPVS->isUndefContained()) {
    return std::nullopt;
  }
  SmallVector<IREE::Util::GlobalOpInterface> globalOps;
  for (auto globalOp : globalPVS->getAssumedSet()) {
    globalOps.push_back(globalOp);
  }
  sortGlobalOps(globalOps);
  return globalOps;
}

std::optional<DeviceSet> DeviceAnalysis::lookupDeviceTargets(
    IREE::Util::GlobalOpInterface deviceGlobalOp) {
  return lookupDeviceTargets(FlatSymbolRefAttr::get(deviceGlobalOp));
}

std::optional<DeviceSet>
DeviceAnalysis::lookupDeviceTargets(SymbolRefAttr deviceGlobalAttr) {
  SetVector<IREE::HAL::DeviceTargetAttr> resultSet;
  gatherDeviceTargets(deviceGlobalAttr, explorer.getRootOp(), resultSet);
  return DeviceSet(resultSet.getArrayRef());
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

// Recursively resolves the referenced device into targets.
void DeviceAnalysis::gatherDeviceTargets(
    Attribute rootAttr, Operation *fromOp,
    SetVector<IREE::HAL::DeviceTargetAttr> &resultSet) {
  IREE::HAL::gatherDeviceTargetAttrs(
      rootAttr, fromOp,
      [&](SymbolRefAttr symRefAttr) {
        return explorer.getSymbolTables()
            .lookupNearestSymbolFrom<IREE::Util::GlobalOpInterface>(fromOp,
                                                                    symRefAttr);
      },
      resultSet);
}

void DeviceAnalysis::gatherAllDeviceTargets(
    SetVector<IREE::HAL::DeviceTargetAttr> &resultSet) {
  for (auto globalOp : deviceGlobals) {
    gatherDeviceTargets(FlatSymbolRefAttr::get(globalOp), explorer.getRootOp(),
                        resultSet);
  }
}

void DeviceAnalysis::gatherDeviceAffinityTargets(
    IREE::Stream::AffinityAttr affinityAttr, Operation *fromOp,
    SetVector<IREE::HAL::DeviceTargetAttr> &resultSet) {
  // We currently only know how to handle HAL device affinities.
  // We could support other ones via an interface but instead we just fall back
  // to default logic if no affinity or an unknown one is found.
  auto deviceAffinityAttr =
      dyn_cast_if_present<IREE::HAL::DeviceAffinityAttr>(affinityAttr);
  if (!deviceAffinityAttr) {
    IREE::HAL::gatherLegacyDeviceTargetAttrs(fromOp, resultSet);
    return;
  }

  // Recursively resolve the referenced device into targets.
  gatherDeviceTargets(deviceAffinityAttr.getDevice(), fromOp, resultSet);
}

void DeviceAnalysis::gatherAllExecutableTargets(
    SetVector<IREE::HAL::ExecutableTargetAttr> &resultSet) {
  SetVector<IREE::HAL::DeviceTargetAttr> deviceTargetSet;
  gatherAllDeviceTargets(deviceTargetSet);
  for (auto deviceTargetAttr : deviceTargetSet) {
    deviceTargetAttr.getExecutableTargets(resultSet);
  }
}

void DeviceAnalysis::gatherRequiredExecutableTargets(
    Operation *forOp, SetVector<IREE::HAL::ExecutableTargetAttr> &resultSet) {
  // Get the affinity from the op or an ancestor. Note that there may be no
  // affinity specified at all.
  auto affinityAttr = IREE::Stream::AffinityAttr::lookupOrDefault(forOp);

  // Gather the device targets that are referenced by the affinity.
  SetVector<IREE::HAL::DeviceTargetAttr> deviceTargetSet;
  gatherDeviceAffinityTargets(affinityAttr, forOp, deviceTargetSet);

  // Add all executable targets on the device targets.
  for (auto deviceTargetAttr : deviceTargetSet) {
    resultSet.insert(deviceTargetAttr.getExecutableTargets().begin(),
                     deviceTargetAttr.getExecutableTargets().end());
  }
}

void DeviceAnalysis::gatherRequiredExecutableTargets(
    IREE::Stream::AffinityAttr affinityAttr, Operation *fromOp,
    SetVector<IREE::HAL::ExecutableTargetAttr> &resultSet) {
  SetVector<IREE::HAL::DeviceTargetAttr> deviceTargetAttrs;
  gatherDeviceAffinityTargets(affinityAttr, fromOp, deviceTargetAttrs);
  for (auto deviceTargetAttr : deviceTargetAttrs) {
    deviceTargetAttr.getExecutableTargets(resultSet);
  }
}

} // namespace mlir::iree_compiler::IREE::HAL
