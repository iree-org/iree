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

class SerializeTargetExecutablesPass
    : public PassWrapper<SerializeTargetExecutablesPass,
                         OperationPass<IREE::HAL::ExecutableOp>> {
 public:
  SerializeTargetExecutablesPass() = default;
  SerializeTargetExecutablesPass(const SerializeTargetExecutablesPass &pass) {}
  SerializeTargetExecutablesPass(StringRef target) {
    this->target = target.str();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    for (auto &targetBackend : matchTargetBackends({target})) {
      targetBackend->getDependentDialects(registry);
    }
  }

  StringRef getArgument() const override {
    return "iree-hal-serialize-target-executables";
  }

  StringRef getDescription() const override {
    return "Serializes hal.executable.variant ops to hal.executable.binary ops";
  }

  void runOnOperation() override {
    auto executableOp = getOperation();
    auto variantOps = llvm::to_vector<4>(
        executableOp.getBlock().getOps<IREE::HAL::ExecutableVariantOp>());
    for (auto variantOp : variantOps) {
      OpBuilder executableBuilder(variantOp);
      for (auto &targetBackend : matchTargetBackends({target})) {
        if (TargetBackend::matchPattern(target,
                                        variantOp.target_backend_filter())) {
          // Ask the target backend to serialize the executable. Note that it
          // may create one or more hal.executable.binary ops in the case of
          // multi-architecture binaries.
          if (failed(targetBackend->serializeExecutable(variantOp,
                                                        executableBuilder))) {
            variantOp.emitError()
                << "failed to serialize executable for target backend "
                << targetBackend->name();
            return signalPassFailure();
          }
          variantOp.erase();
          break;
        }
      }
    }
  }

 private:
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc(
          "Target backend name whose executables will be serialized by "
          "this pass.")};
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createSerializeTargetExecutablesPass(StringRef target) {
  return std::make_unique<SerializeTargetExecutablesPass>(target);
}

static PassRegistration<SerializeTargetExecutablesPass> linkTargetPass([] {
  return std::make_unique<SerializeTargetExecutablesPass>();
});

class SerializeExecutablesPass
    : public PassWrapper<SerializeExecutablesPass,
                         OperationPass<IREE::HAL::ExecutableOp>> {
 public:
  SerializeExecutablesPass() = default;

  StringRef getArgument() const override {
    return "iree-hal-serialize-executables";
  }

  StringRef getDescription() const override {
    return "Serializes hal.executable.variant ops to hal.executable.binary ops";
  }

  void runOnOperation() override {
    auto executableOp = getOperation();
    OpPassManager passManager(executableOp.getOperationName());
    for (auto target : gatherExecutableTargetNames(executableOp)) {
      passManager.addPass(createSerializeTargetExecutablesPass(target));
    }
    if (failed(runPipeline(passManager, executableOp))) {
      executableOp.emitError() << "failed to serialize executables";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createSerializeExecutablesPass() {
  return std::make_unique<SerializeExecutablesPass>();
}

static PassRegistration<SerializeExecutablesPass> linkPass([] {
  return std::make_unique<SerializeExecutablesPass>();
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
