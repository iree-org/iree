// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_ASSIGNLEGACYTARGETDEVICESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-assign-legacy-target-devices
//===----------------------------------------------------------------------===//

struct AssignLegacyTargetDevicesPass
    : public IREE::HAL::impl::AssignLegacyTargetDevicesPassBase<
          AssignLegacyTargetDevicesPass> {
  using IREE::HAL::impl::AssignLegacyTargetDevicesPassBase<
      AssignLegacyTargetDevicesPass>::AssignLegacyTargetDevicesPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    for (StringRef name : targetRegistry->getRegisteredTargetBackends()) {
      targetRegistry->getTargetBackend(name)->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // If no targets are specified we can't do anything - another pass earlier
    // in the pipeline will have had to add the targets.
    if (targetBackends.empty()) {
      return;
    }

    // Check to see if targets are already specified and if so then no-op the
    // pass so that we don't mess with whatever the user intended.
    auto existingTargetsAttr =
        moduleOp->getAttrOfType<ArrayAttr>("hal.device.targets");
    if (existingTargetsAttr) {
      return;
    }

    // If there are any device globals declared then bail as it means the user
    // has already materialized the devices they want.
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
      if (isa<IREE::HAL::DeviceType>(globalOp.getGlobalType())) {
        return;
      }
    }

    llvm::SmallDenseSet<Attribute> targetAttrSet;
    SmallVector<Attribute> targetAttrs;
    for (const auto &targetBackendName : targetBackends) {
      auto targetBackend = targetRegistry->getTargetBackend(targetBackendName);
      if (!targetBackend) {
        auto diagnostic = emitError(moduleOp.getLoc())
                          << "target backend '" << targetBackendName
                          << "' not registered; registered backends: [";
        llvm::interleaveComma(targetRegistry->getRegisteredTargetBackends(),
                              diagnostic);
        diagnostic << "]";
        return signalPassFailure();
      }
      auto targetDeviceName = targetBackend->getLegacyDefaultDeviceID();
      auto targetDevice = targetRegistry->getTargetDevice(targetDeviceName);
      if (!targetDevice) {
        auto diagnostic = emitError(moduleOp.getLoc())
                          << "target device '" << targetDeviceName
                          << "' not registered; registered devices: [";
        llvm::interleaveComma(targetRegistry->getRegisteredTargetDevices(),
                              diagnostic);
        diagnostic << "]";
        return signalPassFailure();
      }

      // Ask the target backend for its default device specification attribute.
      auto targetAttr = targetDevice->getDefaultDeviceTarget(
          moduleOp.getContext(), *targetRegistry.value);
      if (!targetAttr) {
        emitError(moduleOp.getLoc()) << "no default device targets available";
        return signalPassFailure();
      }
      if (!targetAttrSet.contains(targetAttr)) {
        targetAttrSet.insert(targetAttr);
        targetAttrs.push_back(targetAttr);
      }
    }

    Attribute targetsAttr;
    if (targetAttrs.size() == 1) {
      targetsAttr = targetAttrs.front();
    } else {
      targetsAttr =
          IREE::HAL::DeviceSelectAttr::get(moduleOp.getContext(), targetAttrs);
    }
    moduleOp->setAttr("hal.device.targets",
                      ArrayAttr::get(moduleOp.getContext(), targetsAttr));
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
