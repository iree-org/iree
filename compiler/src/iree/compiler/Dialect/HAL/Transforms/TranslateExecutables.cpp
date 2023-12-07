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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::HAL {

class TranslateTargetExecutableVariantsPass
    : public PassWrapper<TranslateTargetExecutableVariantsPass,
                         OperationPass<IREE::HAL::ExecutableVariantOp>> {
public:
  TranslateTargetExecutableVariantsPass()
      : targetRegistry(TargetBackendRegistry::getGlobal()) {}
  TranslateTargetExecutableVariantsPass(
      const TranslateTargetExecutableVariantsPass &pass)
      : targetRegistry(pass.targetRegistry) {}
  TranslateTargetExecutableVariantsPass(
      const TargetBackendRegistry &targetRegistry, StringRef target)
      : targetRegistry(targetRegistry) {
    this->target = target.str();
  }

  StringRef getArgument() const override {
    return "iree-hal-translate-target-executable-variants";
  }

  StringRef getDescription() const override {
    return "Serializes hal.executable.variant ops to hal.executable.binary ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    registry.insert<bufferization::BufferizationDialect>();
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
    targetBackend->buildTranslationPassPipeline(variantOp, passManager);
    if (failed(runPipeline(passManager, variantOp))) {
      variantOp.emitError() << "failed to run translation of source "
                               "executable to target executable for backend "
                            << variantOp.getTarget();
      return signalPassFailure();
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
createTranslateTargetExecutableVariantsPass(
    const TargetBackendRegistry &targetRegistry, StringRef target) {
  return std::make_unique<TranslateTargetExecutableVariantsPass>(targetRegistry,
                                                                 target);
}

static PassRegistration<TranslateTargetExecutableVariantsPass> linkTargetPass(
    [] { return std::make_unique<TranslateTargetExecutableVariantsPass>(); });

class TranslateExecutablesPass
    : public PassWrapper<TranslateExecutablesPass,
                         OperationPass<IREE::HAL::ExecutableOp>> {
public:
  TranslateExecutablesPass()
      : targetRegistry(TargetBackendRegistry::getGlobal()) {}
  TranslateExecutablesPass(const TranslateExecutablesPass &pass)
      : targetRegistry(pass.targetRegistry) {}
  TranslateExecutablesPass(const TargetBackendRegistry &targetRegistry)
      : targetRegistry(targetRegistry) {}

  StringRef getArgument() const override {
    return "iree-hal-translate-executables";
  }

  StringRef getDescription() const override {
    return "Serializes hal.executable.variant ops to hal.executable.binary ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    registry.insert<bufferization::BufferizationDialect>();
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
          createTranslateTargetExecutableVariantsPass(targetRegistry,
                                                      targetName));
    }

    IREE_COMPILER_TRACE_MESSAGE_DYNAMIC(INFO, executableOp.getSymName().str());

    if (failed(runPipeline(passManager, executableOp))) {
      executableOp.emitError() << "failed to serialize executables";
      return signalPassFailure();
    }
  }

  const TargetBackendRegistry &targetRegistry;
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createTranslateExecutablesPass(const TargetBackendRegistry &targetRegistry) {
  return std::make_unique<TranslateExecutablesPass>(targetRegistry);
}

static PassRegistration<TranslateExecutablesPass> translatePass([] {
  return std::make_unique<TranslateExecutablesPass>();
});

} // namespace mlir::iree_compiler::IREE::HAL
