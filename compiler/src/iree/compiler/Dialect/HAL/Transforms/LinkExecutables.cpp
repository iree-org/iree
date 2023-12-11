// Copyright 2020 The IREE Authors
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
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::HAL {

class LinkTargetExecutablesPass
    : public PassWrapper<LinkTargetExecutablesPass,
                         OperationPass<mlir::ModuleOp>> {
public:
  LinkTargetExecutablesPass()
      : targetRegistry(TargetBackendRegistry::getGlobal()) {}
  LinkTargetExecutablesPass(const LinkTargetExecutablesPass &pass)
      : targetRegistry(pass.targetRegistry) {}
  LinkTargetExecutablesPass(const TargetBackendRegistry &targetRegistry,
                            StringRef target)
      : targetRegistry(targetRegistry) {
    this->target = target.str();
  }

  StringRef getArgument() const override {
    return "iree-hal-link-target-executables";
  }

  StringRef getDescription() const override {
    return "Links together hal.executables for the specified target.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    auto targetBackend = targetRegistry.getTargetBackend(target);
    if (targetBackend) {
      targetBackend->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto targetBackend = targetRegistry.getTargetBackend(target);
    if (!targetBackend) {
      moduleOp.emitError() << "unregistered target backend '" << target << "'";
      return signalPassFailure();
    }

    OpPassManager passManager(moduleOp.getOperationName());
    targetBackend->buildLinkingPassPipeline(passManager);
    if (failed(runPipeline(passManager, moduleOp))) {
      moduleOp.emitError()
          << "failed to run linking of executable variants for backend "
          << target;
      return signalPassFailure();
    }
  }

private:
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("Target backend name whose executables will be linked by "
                     "this pass.")};

  const TargetBackendRegistry &targetRegistry;
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createLinkTargetExecutablesPass(const TargetBackendRegistry &targetRegistry,
                                StringRef target) {
  return std::make_unique<LinkTargetExecutablesPass>(targetRegistry, target);
}

static PassRegistration<LinkTargetExecutablesPass> linkTargetPass([] {
  return std::make_unique<LinkTargetExecutablesPass>();
});

class LinkExecutablesPass
    : public PassWrapper<LinkExecutablesPass, OperationPass<mlir::ModuleOp>> {
public:
  LinkExecutablesPass(const TargetBackendRegistry &targetRegistry)
      : targetRegistry(targetRegistry) {}

  StringRef getArgument() const override { return "iree-hal-link-executables"; }

  StringRef getDescription() const override {
    return "Links together hal.executables depending on target backend rules";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Add pipelines for each target backend used in the module.
    // These will create/rearrange executables.
    OpPassManager passManager(moduleOp.getOperationName());
    for (const auto &targetName : gatherExecutableTargetNames(moduleOp)) {
      passManager.addPass(
          createLinkTargetExecutablesPass(targetRegistry, targetName));
    }

    // Cleanup any remaining empty executables after each pipeline has run.
    // We do this to aid debugging as then the pipelines can (mostly) be run in
    // any order and not radically change the IR.
    passManager.addPass(mlir::createSymbolDCEPass());

    if (failed(runPipeline(passManager, moduleOp))) {
      moduleOp.emitError() << "failed to link executables";
      return signalPassFailure();
    }
  }

  const TargetBackendRegistry &targetRegistry;
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createLinkExecutablesPass(const TargetBackendRegistry &targetRegistry) {
  return std::make_unique<LinkExecutablesPass>(targetRegistry);
}

static PassRegistration<LinkExecutablesPass> linkPass([] {
  return std::make_unique<LinkExecutablesPass>(
      TargetBackendRegistry::getGlobal());
});

} // namespace mlir::iree_compiler::IREE::HAL
