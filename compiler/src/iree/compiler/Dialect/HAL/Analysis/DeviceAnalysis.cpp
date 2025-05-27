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

// Returns a set of target devices that may be active for the given
// operation. This will recursively walk parent operations until one with
// the `hal.device.targets` attribute is found.
//
// This is a legacy mechanism for performing the search. Newer code should use
// affinities or !hal.device analysis instead.
static void gatherLegacyDeviceTargetAttrs(
    Operation *op, SetVector<IREE::HAL::DeviceTargetAttr> &resultSet) {
  auto attrId = StringAttr::get(op->getContext(), "hal.device.targets");
  while (op) {
    auto targetsAttr = op->getAttrOfType<ArrayAttr>(attrId);
    if (targetsAttr) {
      for (auto elementAttr : targetsAttr) {
        if (auto targetAttr =
                dyn_cast<IREE::HAL::DeviceTargetAttr>(elementAttr)) {
          resultSet.insert(targetAttr);
        } else {
          // HACK: this legacy approach is deprecated and only preserved for
          // existing behavior. It's ok to get angry here as users should not be
          // trying to use this pass prior to device materialization.
          assert(false &&
                 "legacy hal.device.targets only support hal.device.targets");
        }
      }
      return;
    }
    op = op->getParentOp();
  }
  // No devices found; let caller decide what to do.
}

// Recursively resolves the referenced device into targets.
void DeviceAnalysis::gatherDeviceTargets(
    Attribute rootAttr, Operation *fromOp,
    SetVector<IREE::HAL::DeviceTargetAttr> &resultSet) {
  SetVector<Attribute> worklist;
  worklist.insert(rootAttr);
  do {
    auto attr = worklist.pop_back_val();
    if (!TypeSwitch<Attribute, bool>(attr)
             .Case<SymbolRefAttr>([&](auto symRefAttr) {
               auto globalOp =
                   explorer.getSymbolTables()
                       .lookupNearestSymbolFrom<IREE::Util::GlobalOpInterface>(
                           fromOp, symRefAttr);
               assert(globalOp && "global reference must be valid");
               if (auto initialValueAttr = globalOp.getGlobalInitialValue()) {
                 // Global with a device initialization value we can analyze.
                 worklist.insert(initialValueAttr);
                 return true;
               } else {
                 return false;
               }
             })
             .Case<IREE::HAL::DeviceTargetAttr>([&](auto targetAttr) {
               resultSet.insert(targetAttr);
               return true;
             })
             .Case<IREE::HAL::DeviceFallbackAttr>([&](auto fallbackAttr) {
               worklist.insert(fallbackAttr.getName());
               return true;
             })
             .Case<IREE::HAL::DeviceSelectAttr>([&](auto selectAttr) {
               worklist.insert(selectAttr.getDevices().begin(),
                               selectAttr.getDevices().end());
               return true;
             })
             .Default([](auto attr) { return false; })) {
      // No initial value means fall back to defaults. We do that by
      // inserting all knowable targets.
      gatherLegacyDeviceTargetAttrs(fromOp, resultSet);
      return;
    }
  } while (!worklist.empty());
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
  // If we have a device optimal attr, we need to gather the targets for each
  // of the affinities.
  if (auto optimalAttr =
          mlir::dyn_cast_or_null<IREE::HAL::DeviceOptimalAttr>(affinityAttr)) {
    for (auto affinity : optimalAttr.getAffinities()) {
      gatherDeviceAffinityTargets(affinity, fromOp, resultSet);
    }
  // We currently only know how to handle HAL device affinities.
  // We could support other ones via an interface but instead we just fall back
  // to default logic if no affinity or an unknown one is found.
  } else if (auto deviceAffinityAttr =
                 dyn_cast_if_present<IREE::HAL::DeviceAffinityAttr>(
                     affinityAttr)) {
    gatherDeviceTargets(deviceAffinityAttr.getDevice(), fromOp, resultSet);
  } else {
    gatherLegacyDeviceTargetAttrs(fromOp, resultSet);
  }
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
