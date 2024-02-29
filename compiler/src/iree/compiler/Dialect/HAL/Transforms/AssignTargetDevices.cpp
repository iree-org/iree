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

#define GEN_PASS_DEF_ASSIGNTARGETDEVICESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-assign-target-devices
//===----------------------------------------------------------------------===//

struct AssignTargetDevicesPass
    : public IREE::HAL::impl::AssignTargetDevicesPassBase<
          AssignTargetDevicesPass> {
  using IREE::HAL::impl::AssignTargetDevicesPassBase<
      AssignTargetDevicesPass>::AssignTargetDevicesPassBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    for (auto &targetBackend : targetRegistry->getTargetBackends(
             targetRegistry->getRegisteredTargetBackends())) {
      targetBackend->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Check to see if targets are already specified.
    auto existingTargetsAttr =
        moduleOp->getAttrOfType<ArrayAttr>("hal.device.targets");
    if (existingTargetsAttr) {
      // Targets already exist on the module; no-op the pass so that we don't
      // mess with whatever the user intended.
      return;
    }

    // If no targets are specified we can't do anything - another pass earlier
    // in the pipeline will have had to add the targets.
    if (targetBackends.empty()) {
      emitRemark(moduleOp.getLoc())
          << "no target HAL target backends specified during assignment";
      return;
    }

    llvm::SmallDenseSet<Attribute> targetAttrSet;
    SmallVector<Attribute> targetAttrs;
    for (const auto &targetBackendName : targetBackends) {
      auto targetBackend = targetRegistry->getTargetBackend(targetBackendName);
      if (!targetBackend) {
        std::string backends;
        llvm::raw_string_ostream os(backends);
        llvm::interleaveComma(targetRegistry->getRegisteredTargetBackends(), os,
                              [&os](const std::string &name) { os << name; });
        emitError(moduleOp.getLoc())
            << "target backend '" << targetBackendName
            << "' not registered; registered backends: " << os.str();
        signalPassFailure();
        return;
      }
      auto targetDeviceName = targetBackend->getLegacyDefaultDeviceID();
      auto targetDevice = targetRegistry->getTargetDevice(targetDeviceName);
      if (!targetDevice) {
        std::string devices;
        llvm::raw_string_ostream os(devices);
        llvm::interleaveComma(targetRegistry->getRegisteredTargetDevices(), os,
                              [&os](const std::string &name) { os << name; });
        emitError(moduleOp.getLoc())
            << "target device '" << targetDeviceName
            << "' not registered; registered devices: " << os.str();
        signalPassFailure();
        return;
      }

      // Ask the target backend for its default device specification attribute.
      auto targetAttr = targetDevice->getDefaultDeviceTarget(
          moduleOp.getContext(), *targetRegistry.value);
      if (!targetAttrSet.contains(targetAttr)) {
        targetAttrSet.insert(targetAttr);
        targetAttrs.push_back(targetAttr);
      }
    }

    moduleOp->setAttr("hal.device.targets",
                      ArrayAttr::get(moduleOp.getContext(), targetAttrs));
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
