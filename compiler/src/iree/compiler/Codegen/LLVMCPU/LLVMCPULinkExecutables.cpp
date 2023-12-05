// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Utils/LinkingUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct LLVMCPULinkExecutablesPass
    : public LLVMCPULinkExecutablesBase<LLVMCPULinkExecutablesPass> {
  LLVMCPULinkExecutablesPass() = default;
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());

    auto sourceExecutableOps =
        llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
    if (sourceExecutableOps.size() <= 1)
      return;

    // Guess a module name, if needed, to make the output files readable.
    auto moduleName = guessModuleName(moduleOp, "llvm_module");

    // Create our new "linked" hal.executable.
    std::string linkedExecutableName =
        llvm::formatv("{0}_linked_{1}", moduleName, "llvm_cpu");
    auto linkedExecutableOp = moduleBuilder.create<IREE::HAL::ExecutableOp>(
        moduleOp.getLoc(), linkedExecutableName);
    linkedExecutableOp.setVisibility(
        sourceExecutableOps.front().getVisibility());
    auto executableBuilder =
        OpBuilder::atBlockBegin(&linkedExecutableOp.getBlock());

    // Gather all unique executable targets - we may have multiple.
    auto executableTargetAttrs = gatherExecutableTargets(sourceExecutableOps);
    for (auto executableTargetAttr : executableTargetAttrs) {
      // Add our hal.executable.variant with an empty module.
      auto linkedTargetOp =
          executableBuilder.create<IREE::HAL::ExecutableVariantOp>(
              moduleOp.getLoc(), executableTargetAttr.getSymbolNameFragment(),
              executableTargetAttr);
      auto targetBuilder = OpBuilder::atBlockBegin(&linkedTargetOp.getBlock());
      targetBuilder.create<mlir::ModuleOp>(moduleOp.getLoc());

      // Try linking together all executables in moduleOp.
      if (failed(linkExecutablesInto(
              moduleOp, sourceExecutableOps, linkedExecutableOp, linkedTargetOp,
              [](mlir::ModuleOp moduleOp) { return moduleOp; }))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createLLVMCPULinkExecutablesPass() {
  return std::make_unique<LLVMCPULinkExecutablesPass>();
}

} // namespace iree_compiler
} // namespace mlir
