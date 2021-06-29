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

class LinkExecutablesPass
    : public PassWrapper<LinkExecutablesPass, OperationPass<mlir::ModuleOp>> {
 public:
  explicit LinkExecutablesPass(TargetOptions executableOptions)
      : executableOptions_(executableOptions) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    auto targetBackends = matchTargetBackends(executableOptions_.targets);
    for (auto &targetBackend : targetBackends) {
      targetBackend->getDependentDialects(registry);
    }
  }

  StringRef getArgument() const override { return "iree-hal-link-executables"; }

  StringRef getDescription() const override {
    return "Links together hal.executables depending on target backend rules";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    for (auto &targetBackend :
         matchTargetBackends(executableOptions_.targets)) {
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
  TargetOptions executableOptions_;
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLinkExecutablesPass(
    TargetOptions executableOptions) {
  return std::make_unique<LinkExecutablesPass>(executableOptions);
}

static PassRegistration<LinkExecutablesPass> pass([] {
  auto options = getTargetOptionsFromFlags();
  return std::make_unique<LinkExecutablesPass>(options);
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
