// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/LinkingUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

namespace {

struct SPIRVLinkExecutablesPass final
    : SPIRVLinkExecutablesBase<SPIRVLinkExecutablesPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    OpBuilder moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());

    SmallVector<IREE::HAL::ExecutableOp, 8> sourceExecutableOps =
        llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
    if (sourceExecutableOps.size() <= 1)
      return;

    // Guess a module name, if needed, to make the output files readable.
    std::string moduleName = guessModuleName(moduleOp, "spirv_module");

    // Create our new "linked" hal.executable.
    std::string linkedExecutableName =
        llvm::formatv("{0}_linked_{1}", moduleName, "spirv");
    auto linkedExecutableOp = moduleBuilder.create<IREE::HAL::ExecutableOp>(
        moduleOp.getLoc(), linkedExecutableName);
    linkedExecutableOp.setVisibility(
        sourceExecutableOps.front().getVisibility());
    OpBuilder executableBuilder =
        OpBuilder::atBlockBegin(&linkedExecutableOp.getBlock());

    // Gather all unique executable targets - we may have multiple.
    SetVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs =
        gatherExecutableTargets(sourceExecutableOps);
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
              [](mlir::ModuleOp moduleOp) { return moduleOp; },
              targetBuilder))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createSPIRVLinkExecutablesPass() {
  return std::make_unique<SPIRVLinkExecutablesPass>();
}

} // namespace mlir::iree_compiler
