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
#include "llvm/Support/FileSystem.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::HAL {

class SerializeTargetExecutablesPass
    : public PassWrapper<SerializeTargetExecutablesPass,
                         OperationPass<IREE::HAL::ExecutableOp>> {
public:
  SerializeTargetExecutablesPass()
      : targetRegistry(TargetBackendRegistry::getGlobal()) {}
  SerializeTargetExecutablesPass(const SerializeTargetExecutablesPass &pass)
      : targetRegistry(pass.targetRegistry) {}
  SerializeTargetExecutablesPass(const TargetBackendRegistry &targetRegistry,
                                 StringRef target, int debugLevel,
                                 std::string dumpIntermediatesPath,
                                 std::string dumpBinariesPath)
      : targetRegistry(targetRegistry) {
    this->target = target.str();
    this->debugLevel = debugLevel;
    this->dumpIntermediatesPath = dumpIntermediatesPath;
    this->dumpBinariesPath = dumpBinariesPath;
  }

  StringRef getArgument() const override {
    return "iree-hal-serialize-target-executables";
  }

  StringRef getDescription() const override {
    return "Serializes hal.executable.variant ops to hal.executable.binary ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    auto targetBackend = targetRegistry.getTargetBackend(target);
    if (targetBackend) {
      targetBackend->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    auto executableOp = getOperation();
    auto moduleOp = executableOp->getParentOfType<mlir::ModuleOp>();

    auto targetBackend = targetRegistry.getTargetBackend(target);
    if (!targetBackend) {
      executableOp.emitError()
          << "unregistered target backend '" << target << "'";
      return signalPassFailure();
    }

    // Setup serialization options for dumping artifacts (if requested).
    // Target backends are allowed to generate multiple binaries for each
    // serialization.
    TargetBackend::SerializationOptions serializationOptions;
    serializationOptions.dumpBaseName =
        (moduleOp.getName().value_or("module") + "_" + executableOp.getName())
            .str();
    serializationOptions.debugLevel = debugLevel;
    serializationOptions.dumpIntermediatesPath = dumpIntermediatesPath;
    serializationOptions.dumpBinariesPath = dumpBinariesPath;
    if (!dumpIntermediatesPath.empty()) {
      llvm::sys::fs::create_directories(dumpIntermediatesPath);
    }
    if (!dumpBinariesPath.empty()) {
      llvm::sys::fs::create_directories(dumpBinariesPath);
    }

    auto variantOps = llvm::to_vector(
        executableOp.getBlock().getOps<IREE::HAL::ExecutableVariantOp>());
    for (auto variantOp : variantOps) {
      if (variantOp.getTarget().getBackend().getValue() != target)
        continue;
      OpBuilder executableBuilder(variantOp);
      // Ask the target backend to serialize the executable. Note that it
      // may create one or more hal.executable.binary ops in the case of
      // multi-architecture binaries.
      if (failed(targetBackend->serializeExecutable(
              serializationOptions, variantOp, executableBuilder))) {
        variantOp.emitError()
            << "failed to serialize executable for target backend " << target;
        return signalPassFailure();
      }
      variantOp.erase();
    }
  }

private:
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc(
          "Target backend name whose executables will be serialized by "
          "this pass.")};

  Option<int> debugLevel{*this, "debug-level",
                         llvm::cl::desc("Debug level for serialization (0-3)"),
                         llvm::cl::init(2)};
  Option<std::string> dumpIntermediatesPath{
      *this, "dump-intermediates-path",
      llvm::cl::desc("Path to write translated executable intermediates (.bc, "
                     ".o, etc) into for debugging.")};
  Option<std::string> dumpBinariesPath{
      *this, "dump-binaries-path",
      llvm::cl::desc("Path to write translated and serialized executable "
                     "binaries into for debugging.")};

  const TargetBackendRegistry &targetRegistry;
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createSerializeTargetExecutablesPass(
    const TargetBackendRegistry &targetRegistry, StringRef target,
    int debugLevel, std::string dumpIntermediatesPath,
    std::string dumpBinariesPath) {
  return std::make_unique<SerializeTargetExecutablesPass>(
      targetRegistry, target, debugLevel, dumpIntermediatesPath,
      dumpBinariesPath);
}

static PassRegistration<SerializeTargetExecutablesPass> linkTargetPass([] {
  return std::make_unique<SerializeTargetExecutablesPass>();
});

class SerializeExecutablesPass
    : public PassWrapper<SerializeExecutablesPass,
                         OperationPass<IREE::HAL::ExecutableOp>> {
public:
  SerializeExecutablesPass()
      : targetRegistry(TargetBackendRegistry::getGlobal()) {}
  SerializeExecutablesPass(const TargetBackendRegistry &targetRegistry,
                           int debugLevel, std::string dumpIntermediatesPath,
                           std::string dumpBinariesPath)
      : targetRegistry(targetRegistry), debugLevel(debugLevel),
        dumpIntermediatesPath(dumpIntermediatesPath),
        dumpBinariesPath(dumpBinariesPath) {}

  StringRef getArgument() const override {
    return "iree-hal-serialize-executables";
  }

  StringRef getDescription() const override {
    return "Serializes hal.executable.variant ops to hal.executable.binary ops";
  }

  void runOnOperation() override {
    auto executableOp = getOperation();
    OpPassManager passManager(executableOp.getOperationName());
    for (const auto &targetName : gatherExecutableTargetNames(executableOp)) {
      passManager.addPass(createSerializeTargetExecutablesPass(
          targetRegistry, targetName, debugLevel, dumpIntermediatesPath,
          dumpBinariesPath));
    }
    if (failed(runPipeline(passManager, executableOp))) {
      executableOp.emitError() << "failed to serialize executables";
      return signalPassFailure();
    }
  }

private:
  const TargetBackendRegistry &targetRegistry;
  int debugLevel;
  std::string dumpIntermediatesPath;
  std::string dumpBinariesPath;
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createSerializeExecutablesPass(const TargetBackendRegistry &targetRegistry,
                               int debugLevel,
                               std::string dumpIntermediatesPath,
                               std::string dumpBinariesPath) {
  return std::make_unique<SerializeExecutablesPass>(
      targetRegistry, debugLevel, dumpIntermediatesPath, dumpBinariesPath);
}

static PassRegistration<SerializeExecutablesPass> linkPass([] {
  return std::make_unique<SerializeExecutablesPass>();
});

} // namespace mlir::iree_compiler::IREE::HAL
