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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

class AssignTargetDevicesPass
    : public PassWrapper<AssignTargetDevicesPass, OperationPass<ModuleOp>> {
public:
  AssignTargetDevicesPass()
      : targetRegistry(TargetBackendRegistry::getGlobal()) {}
  AssignTargetDevicesPass(const AssignTargetDevicesPass &pass)
      : targetRegistry(pass.targetRegistry) {}
  AssignTargetDevicesPass(const TargetBackendRegistry &targetRegistry,
                          ArrayRef<std::string> targets)
      : targetRegistry(targetRegistry) {
    this->targets = targets;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    for (auto &targetBackend : targetRegistry.getTargetBackends(
             targetRegistry.getRegisteredTargetBackends())) {
      targetBackend->getDependentDialects(registry);
    }
  }

  StringRef getArgument() const override {
    return "iree-hal-assign-target-devices";
  }

  StringRef getDescription() const override {
    return "Assigns the HAL devices the module will target to the given list "
           "of targets.";
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
    if (targets.empty()) {
      emitRemark(moduleOp.getLoc())
          << "no target HAL devices specified during assignment";
      return;
    }

    llvm::SmallDenseSet<Attribute> targetAttrSet;
    SmallVector<Attribute> targetAttrs;
    for (const auto &targetName : targets) {
      auto targetBackend = targetRegistry.getTargetBackend(targetName);
      if (!targetBackend) {
        std::string backends;
        llvm::raw_string_ostream os(backends);
        llvm::interleaveComma(
            targetRegistry.getTargetBackends(
                targetRegistry.getRegisteredTargetBackends()),
            os,
            [&os](const std::shared_ptr<
                  mlir::iree_compiler::IREE::HAL::TargetBackend>
                      b) { os << b->name(); });
        emitError(moduleOp.getLoc())
            << "target backend '" << targetName
            << "' not registered; registered backends: " << os.str();
        signalPassFailure();
        return;
      }

      // Ask the target backend for its default device specification attribute.
      auto targetAttr =
          targetBackend->getDefaultDeviceTarget(moduleOp.getContext());
      if (!targetAttrSet.contains(targetAttr)) {
        targetAttrSet.insert(targetAttr);
        targetAttrs.push_back(targetAttr);
      }
    }

    moduleOp->setAttr("hal.device.targets",
                      ArrayAttr::get(moduleOp.getContext(), targetAttrs));
  }

private:
  ListOption<std::string> targets{*this, "targets",
                                  llvm::cl::desc("List of devices to target."),
                                  llvm::cl::ZeroOrMore};

  const TargetBackendRegistry &targetRegistry;
};

std::unique_ptr<OperationPass<ModuleOp>>
createAssignTargetDevicesPass(const TargetBackendRegistry &targetRegistry,
                              ArrayRef<std::string> targets) {
  return std::make_unique<AssignTargetDevicesPass>(targetRegistry, targets);
}

static PassRegistration<AssignTargetDevicesPass> pass([] {
  return std::make_unique<AssignTargetDevicesPass>();
});

} // namespace mlir::iree_compiler::IREE::HAL
