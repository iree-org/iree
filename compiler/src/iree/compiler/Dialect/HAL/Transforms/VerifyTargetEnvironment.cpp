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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

class VerifyTargetEnvironmentPass
    : public PassWrapper<VerifyTargetEnvironmentPass, OperationPass<ModuleOp>> {
public:
  VerifyTargetEnvironmentPass(const TargetBackendRegistry &targetRegistry)
      : targetRegistry(targetRegistry) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-verify-target-environment";
  }

  StringRef getDescription() const override {
    return "Verifies that the target execution environment is valid.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Targets are required if we need to convert host code or executables.
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
    if (!anyNonExecutableOps)
      return;

    // Must have targets specified.
    auto targetsAttr = moduleOp->getAttrOfType<ArrayAttr>("hal.device.targets");
    if (!targetsAttr || targetsAttr.empty()) {
      auto diagnostic = moduleOp.emitError();
      diagnostic
          << "no HAL target devices specified on the module (available = [ ";
      for (const auto &targetName :
           targetRegistry.getRegisteredTargetBackends()) {
        diagnostic << "'" << targetName << "' ";
      }
      diagnostic << "])";
      signalPassFailure();
      return;
    }

    // Verify each target is registered.
    for (auto attr : targetsAttr) {
      auto targetAttr = llvm::dyn_cast<IREE::HAL::DeviceTargetAttr>(attr);
      if (!targetAttr) {
        moduleOp.emitError() << "invalid target attr type: " << attr;
        signalPassFailure();
        return;
      }

      auto targetBackend =
          targetRegistry.getTargetBackend(targetAttr.getDeviceID().getValue());
      if (!targetBackend) {
        auto diagnostic = moduleOp.emitError();
        diagnostic
            << "unregistered target backend " << targetAttr.getDeviceID()
            << "; ensure it is linked in to the compiler (available = [ ";
        for (const auto &targetName :
             targetRegistry.getRegisteredTargetBackends()) {
          diagnostic << "'" << targetName << "' ";
        }
        diagnostic << "])";
        signalPassFailure();
        return;
      }
    }
  }

  const TargetBackendRegistry &targetRegistry;
};

std::unique_ptr<OperationPass<ModuleOp>>
createVerifyTargetEnvironmentPass(const TargetBackendRegistry &targetRegistry) {
  return std::make_unique<VerifyTargetEnvironmentPass>(targetRegistry);
}

static PassRegistration<VerifyTargetEnvironmentPass> pass([] {
  return std::make_unique<VerifyTargetEnvironmentPass>(
      TargetBackendRegistry::getGlobal());
});

} // namespace mlir::iree_compiler::IREE::HAL
