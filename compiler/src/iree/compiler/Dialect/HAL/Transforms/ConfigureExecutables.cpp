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
#include "iree/compiler/Utils/TracingUtils.h"
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

class ConfigureTargetExecutableVariantsPass
    : public PassWrapper<ConfigureTargetExecutableVariantsPass,
                         OperationPass<IREE::HAL::ExecutableVariantOp>> {
public:
  ConfigureTargetExecutableVariantsPass()
      : targetRegistry(TargetBackendRegistry::getGlobal()) {}
  ConfigureTargetExecutableVariantsPass(
      const ConfigureTargetExecutableVariantsPass &pass)
      : targetRegistry(pass.targetRegistry) {}
  ConfigureTargetExecutableVariantsPass(
      const TargetBackendRegistry &targetRegistry, StringRef target)
      : targetRegistry(targetRegistry) {
    this->target = target.str();
  }

  StringRef getArgument() const override {
    return "iree-hal-configure-target-executable-variants";
  }

  StringRef getDescription() const override {
    return "Configures a hal.executable.variant op for translation";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    auto targetBackend = targetRegistry.getTargetBackend(target);
    if (targetBackend) {
      targetBackend->getDependentDialects(registry);
    }
  }
  void runOnOperation() override {
    auto variantOp = getOperation();
    if (variantOp.getTarget().getBackend().getValue() != target)
      return;

    auto targetBackend = targetRegistry.getTargetBackend(target);
    if (!targetBackend) {
      variantOp.emitError() << "unregistered target backend '" << target << "'";
      return signalPassFailure();
    }

    OpPassManager passManager(variantOp.getOperationName());
    targetBackend->buildConfigurationPassPipeline(variantOp, passManager);

    // This pipeline is optional, and the default is no passes, in which case
    // nothing is needed.
    if (!passManager.empty()) {
      if (failed(runPipeline(passManager, variantOp))) {
        variantOp.emitError() << "failed to run translation of source "
                                 "executable to target executable for backend "
                              << variantOp.getTarget();
        return signalPassFailure();
      }
    }
  }

private:
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc(
          "Target backend name whose executables will be translated by "
          "this pass.")};

  const TargetBackendRegistry &targetRegistry;
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createConfigureTargetExecutableVariantsPass(
    const TargetBackendRegistry &targetRegistry, StringRef target) {
  return std::make_unique<ConfigureTargetExecutableVariantsPass>(targetRegistry,
                                                                 target);
}

static PassRegistration<ConfigureTargetExecutableVariantsPass> linkTargetPass(
    [] { return std::make_unique<ConfigureTargetExecutableVariantsPass>(); });

class ConfigureExecutablesPass
    : public PassWrapper<ConfigureExecutablesPass,
                         OperationPass<IREE::HAL::ExecutableOp>> {
public:
  ConfigureExecutablesPass()
      : targetRegistry(TargetBackendRegistry::getGlobal()) {}
  ConfigureExecutablesPass(const ConfigureExecutablesPass &pass)
      : targetRegistry(pass.targetRegistry) {}
  ConfigureExecutablesPass(const TargetBackendRegistry &targetRegistry)
      : targetRegistry(targetRegistry) {}

  StringRef getArgument() const override {
    return "iree-hal-configure-executables";
  }

  StringRef getDescription() const override {
    return "Configures hal.executable.variant ops for translation to "
           "hal.executable.binary ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    auto targetBackends = targetRegistry.getTargetBackends(
        targetRegistry.getRegisteredTargetBackends());
    for (auto &targetBackend : targetBackends) {
      targetBackend->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    auto executableOp = getOperation();
    OpPassManager passManager(executableOp.getOperationName());
    for (const auto &targetName : gatherExecutableTargetNames(executableOp)) {
      passManager.addNestedPass<IREE::HAL::ExecutableVariantOp>(
          createConfigureTargetExecutableVariantsPass(targetRegistry,
                                                      targetName));
    }

    IREE_COMPILER_TRACE_MESSAGE_DYNAMIC(INFO, executableOp.getSymName().str());

    if (failed(runPipeline(passManager, executableOp))) {
      executableOp.emitError() << "failed to configure executables";
      return signalPassFailure();
    }
  }

private:
  const TargetBackendRegistry &targetRegistry;
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createConfigureExecutablesPass(const TargetBackendRegistry &targetRegistry) {
  return std::make_unique<ConfigureExecutablesPass>(targetRegistry);
}

static PassRegistration<ConfigureExecutablesPass> translatePass([] {
  return std::make_unique<ConfigureExecutablesPass>();
});

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
