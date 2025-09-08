// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Utils/LinkingUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPULINKEXECUTABLESPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

struct LLVMCPULinkExecutablesPass
    : public impl::LLVMCPULinkExecutablesPassBase<LLVMCPULinkExecutablesPass> {
  using impl::LLVMCPULinkExecutablesPassBase<
      LLVMCPULinkExecutablesPass>::LLVMCPULinkExecutablesPassBase;
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());

    auto sourceExecutableOps = gatherExecutablesForTarget(moduleOp, target);
    if (sourceExecutableOps.size() <= 1)
      return;

    // Guess a module name, if needed, to make the output files readable.
    auto moduleName = guessModuleName(moduleOp, "module");

    // Create our new "linked" hal.executable.
    SymbolTable moduleTable(moduleOp);
    std::string linkedExecutableName = llvm::formatv("{}_linked", moduleName);
    auto linkedExecutableOp = IREE::HAL::ExecutableOp::create(
        moduleBuilder, moduleOp.getLoc(), linkedExecutableName);
    linkedExecutableOp.setVisibility(
        sourceExecutableOps.front().getVisibility());
    moduleTable.insert(linkedExecutableOp);
    auto executableBuilder =
        OpBuilder::atBlockBegin(&linkedExecutableOp.getBlock());

    // Gather all unique executable targets - we may have multiple.
    auto executableTargetAttrs = gatherExecutableTargets(sourceExecutableOps);
    for (auto [index, targetAttr] : llvm::enumerate(executableTargetAttrs)) {
      // Only link the target specified. If none specified link all.
      if (!target.empty() && targetAttr.getBackend().getValue() != target) {
        continue; // not linking this target
      }

      // Add our hal.executable.variant with an empty module.
      std::string linkedVariantName =
          executableTargetAttrs.size() == 1
              ? targetAttr.getSymbolNameFragment()
              : llvm::formatv("{}_{}", targetAttr.getSymbolNameFragment(),
                              index);
      auto linkedTargetOp = IREE::HAL::ExecutableVariantOp::create(
          executableBuilder, moduleOp.getLoc(), linkedVariantName, targetAttr);
      auto targetBuilder = OpBuilder::atBlockBegin(&linkedTargetOp.getBlock());
      mlir::ModuleOp::create(targetBuilder, moduleOp.getLoc());

      auto mergeModuleFn = [](mlir::ModuleOp sourceInnerModule,
                              mlir::ModuleOp linkedInnerModule,
                              DenseMap<StringRef, Operation *> &symbolMap) {
        return mergeModuleInto(sourceInnerModule, linkedInnerModule, symbolMap);
      };

      // Try linking together all executables in moduleOp.
      if (failed(linkExecutablesInto(moduleOp, sourceExecutableOps,
                                     linkedExecutableOp, linkedTargetOp,
                                     mergeModuleFn))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
