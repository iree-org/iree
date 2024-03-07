// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_VERIFYDEVICESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-verify-devices
//===----------------------------------------------------------------------===//

static void printAvailable(InFlightDiagnostic &diagnostic,
                           const TargetRegistry &targetRegistry) {
  diagnostic << "available devices: [";
  llvm::interleaveComma(targetRegistry.getRegisteredTargetDevices(),
                        diagnostic);
  diagnostic << "], available backends = [";
  llvm::interleaveComma(targetRegistry.getRegisteredTargetBackends(),
                        diagnostic);
  diagnostic << "]";
}

static LogicalResult
verifyDeviceTargetAttr(Operation *deviceOp,
                       IREE::HAL::DeviceTargetAttr deviceTargetAttr,
                       const TargetRegistry &targetRegistry) {
  auto targetDevice =
      targetRegistry.getTargetDevice(deviceTargetAttr.getDeviceID().getValue());
  if (!targetDevice) {
    auto diagnostic = deviceOp->emitError();
    diagnostic << "unregistered target device "
               << deviceTargetAttr.getDeviceID()
               << "; ensure it is linked in to the compiler (available = [ ";
    for (const auto &targetName : targetRegistry.getRegisteredTargetDevices()) {
      diagnostic << "'" << targetName << "' ";
    }
    diagnostic << "])";
    return diagnostic;
  }

  for (auto executableTargetAttr : deviceTargetAttr.getExecutableTargets()) {
    auto targetBackend = targetRegistry.getTargetBackend(
        executableTargetAttr.getBackend().getValue());
    if (!targetBackend) {
      auto diagnostic = deviceOp->emitError();
      diagnostic << "unregistered target backend "
                 << executableTargetAttr.getBackend()
                 << "; ensure it is linked in to the compiler (available = [ ";
      for (const auto &targetName :
           targetRegistry.getRegisteredTargetBackends()) {
        diagnostic << "'" << targetName << "' ";
      }
      diagnostic << "])";
      return diagnostic;
    }
  }

  return success();
}

static LogicalResult verifyAttr(Operation *deviceOp, Attribute attr,
                                const TargetRegistry &targetRegistry) {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<IREE::HAL::DeviceTargetAttr>([&](auto deviceTargetAttr) {
        return verifyDeviceTargetAttr(deviceOp, deviceTargetAttr,
                                      targetRegistry);
      })
      .Case<IREE::HAL::DeviceSelectAttr>([&](auto deviceSelectAttr) {
        for (auto attr : deviceSelectAttr.getDevices().getValue()) {
          if (failed(verifyAttr(deviceOp, attr, targetRegistry))) {
            return failure();
          }
        }
        return success();
      })
      .Default([&](auto attr) {
        return success(); // probably fallback/ordinal/etc - can't verify
      });
}

struct VerifyDevicesPass
    : public IREE::HAL::impl::VerifyDevicesPassBase<VerifyDevicesPass> {
  using IREE::HAL::impl::VerifyDevicesPassBase<
      VerifyDevicesPass>::VerifyDevicesPassBase;
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Devices are required if we need to convert host code or executables.
    // If we only have hal.executables as input then we can bypass this.
    // We could extend this check to be a bit smarter at the risk of false
    // negatives - today this is just handling the standalone hal.executable
    // compilation workflow.
    bool anyNonExecutableOps = false;
    for (auto &op : moduleOp.getOps()) {
      if (!isa<IREE::HAL::ExecutableOp>(op)) {
        anyNonExecutableOps = true;
        break;
      }
    }
    if (!anyNonExecutableOps) {
      return;
    }

    // Analyze the module to find all devices.
    DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    // Must have at least one device specified.
    if (deviceAnalysis.getDeviceGlobals().empty()) {
      auto diagnostic = moduleOp.emitError();
      diagnostic
          << "no HAL devices defined in the module; use the module-level "
             "hal.device.targets attribute, the --iree-hal-target-device= "
             "flags, or provide inputs with global !hal.devices defined; ";
      printAvailable(diagnostic, *targetRegistry.value);
      return signalPassFailure();
    }

    // Walk all devices and verify them.
    for (auto deviceOp : deviceAnalysis.getDeviceGlobals()) {
      if (auto initialValue = deviceOp.getGlobalInitialValue()) {
        if (failed(verifyAttr(deviceOp, initialValue, *targetRegistry.value))) {
          return signalPassFailure();
        }
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
