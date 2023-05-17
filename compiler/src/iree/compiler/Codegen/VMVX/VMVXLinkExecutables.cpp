// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Utils/LinkingUtils.h"
#include "iree/compiler/Codegen/VMVX/VMVXPasses.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct VMVXLinkExecutablesPass
    : public VMVXLinkExecutablesBase<VMVXLinkExecutablesPass> {
  VMVXLinkExecutablesPass() = default;
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());

    auto sourceExecutableOps =
        llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
    if (sourceExecutableOps.size() <= 1) return;

    // Guess a module name, if needed, to make the output files readable.
    auto moduleName = guessModuleName(moduleOp, "vmvx_module");

    // Create our new "linked" hal.executable.
    std::string linkedExecutableName =
        llvm::formatv("{0}_linked_{1}", moduleName, "vmvx");
    auto linkedExecutableOp = moduleBuilder.create<IREE::HAL::ExecutableOp>(
        moduleOp.getLoc(), linkedExecutableName);
    linkedExecutableOp.setVisibility(
        sourceExecutableOps.front().getVisibility());
    auto executableBuilder =
        OpBuilder::atBlockBegin(&linkedExecutableOp.getBlock());

    // Gather all unique executable targets - we may have multiple.
    auto executableTargetAttrs = gatherExecutableTargets(sourceExecutableOps);
    for (auto executableTargetAttr : executableTargetAttrs) {
      // Add our VMVX hal.executable.variant with an empty module.
      auto linkedTargetOp =
          executableBuilder.create<IREE::HAL::ExecutableVariantOp>(
              moduleOp.getLoc(), executableTargetAttr.getSymbolNameFragment(),
              executableTargetAttr);
      auto targetBuilder = OpBuilder::atBlockBegin(&linkedTargetOp.getBlock());
      auto linkedModuleOp = targetBuilder.create<ModuleOp>(moduleOp.getLoc());

      // Add an empty vm.module to that module as our vm.funcs must live in it.
      auto nestedBuilder = OpBuilder::atBlockBegin(linkedModuleOp.getBody());
      nestedBuilder.create<IREE::VM::ModuleOp>(moduleOp.getLoc(),
                                               "linked_module");

      // Try linking together all executable variants for this target.
      if (failed(linkExecutablesInto(
              moduleOp, sourceExecutableOps, linkedExecutableOp, linkedTargetOp,
              [](mlir::ModuleOp moduleOp) {
                return *moduleOp.getOps<IREE::VM::ModuleOp>().begin();
              },
              nestedBuilder))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createVMVXLinkExecutablesPass() {
  return std::make_unique<VMVXLinkExecutablesPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
