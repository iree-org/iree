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

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class LinkTargetExecutablesPass
    : public PassWrapper<LinkTargetExecutablesPass,
                         OperationPass<mlir::ModuleOp>> {
 public:
  LinkTargetExecutablesPass() = default;
  LinkTargetExecutablesPass(const LinkTargetExecutablesPass &pass) {}
  LinkTargetExecutablesPass(StringRef target) { this->target = target.str(); }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    for (auto &targetBackend : matchTargetBackends({target})) {
      targetBackend->getDependentDialects(registry);
    }
  }

  StringRef getArgument() const override {
    return "iree-hal-link-target-executables";
  }

  StringRef getDescription() const override {
    return "Links together hal.executables for the specified target.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    for (auto &targetBackend : matchTargetBackends({target})) {
      // Ask the target backend to link all executables it wants.
      if (failed(targetBackend->linkExecutables(moduleOp))) {
        moduleOp.emitError() << "failed to link executables for target backend "
                             << targetBackend->name();
        return signalPassFailure();
      }
    }

    // Backends may move target ops from executables into linked executables.
    // If an executable ends up with no targets, remove it.
    auto executableOps =
        llvm::to_vector<4>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
    for (auto executableOp : executableOps) {
      auto targetOps = executableOp.getOps<IREE::HAL::ExecutableVariantOp>();
      if (targetOps.empty()) {
        executableOp.erase();
      }
    }
  }

 private:
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("Target backend name whose executables will be linked by "
                     "this pass.")};
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLinkTargetExecutablesPass(
    StringRef target) {
  return std::make_unique<LinkTargetExecutablesPass>(target);
}

static PassRegistration<LinkTargetExecutablesPass> linkTargetPass([] {
  return std::make_unique<LinkTargetExecutablesPass>();
});

class LinkExecutablesPass
    : public PassWrapper<LinkExecutablesPass, OperationPass<mlir::ModuleOp>> {
 public:
  LinkExecutablesPass() = default;

  StringRef getArgument() const override { return "iree-hal-link-executables"; }

  StringRef getDescription() const override {
    return "Links together hal.executables depending on target backend rules";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    OpPassManager passManager(moduleOp.getOperationName());
    for (auto &targetBackend :
         matchTargetBackends(gatherExecutableTargetNames(moduleOp))) {
      passManager.addPass(
          createLinkTargetExecutablesPass(targetBackend->filter_pattern()));
    }
    if (failed(runPipeline(passManager, moduleOp))) {
      moduleOp.emitError() << "failed to link executables";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLinkExecutablesPass() {
  return std::make_unique<LinkExecutablesPass>();
}

static PassRegistration<LinkExecutablesPass> linkPass([] {
  return std::make_unique<LinkExecutablesPass>();
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
