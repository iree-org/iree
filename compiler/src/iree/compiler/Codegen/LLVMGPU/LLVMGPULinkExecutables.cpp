// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/LinkingUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPULINKEXECUTABLESPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

// Returns true if the address space of a global symbol is private to the module
// scope it originates in. AMD and NVIDIA disagree on the naming but the values
// match. LLVM is a mess here.
static bool isSymbolAddressSpacePrivate(uint32_t addressSpace) {
  return addressSpace == /*local*/ 3 || addressSpace == /*private*/ 5;
}

static SymbolTable::Visibility
convertLinkageToVisibility(LLVM::Linkage linkage) {
  switch (linkage) {
  case LLVM::Linkage::Private:
    return SymbolTable::Visibility::Private;
  case LLVM::Linkage::External:
    return SymbolTable::Visibility::Public;
  default:
    return SymbolTable::Visibility::Public;
  }
}

// Returns true if we are allowed to rename |op| as part of merging.
// The LLVMGPU lowering is super careful about assigning linkage so we err on
// the side of renaming (as 100% of usage today does not reference external
// things).
static bool allowRenamingPrivateLLVMSymbols(Operation *op) {
  if (auto globalOp = dyn_cast<LLVM::GlobalOp>(op)) {
    if (isSymbolAddressSpacePrivate(globalOp.getAddrSpace())) {
      return true;
    }
    return convertLinkageToVisibility(globalOp.getLinkage()) ==
           SymbolTable::Visibility::Private;
  } else if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {
    return convertLinkageToVisibility(funcOp.getLinkage()) ==
           SymbolTable::Visibility::Private;
  }
  return SymbolTable::getSymbolVisibility(op) ==
         SymbolTable::Visibility::Private;
}

struct LLVMGPULinkExecutablesPass
    : public impl::LLVMGPULinkExecutablesPassBase<LLVMGPULinkExecutablesPass> {
  using impl::LLVMGPULinkExecutablesPassBase<
      LLVMGPULinkExecutablesPass>::LLVMGPULinkExecutablesPassBase;
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
    std::string linkedExecutableName = llvm::formatv("{0}_linked", moduleName);
    auto linkedExecutableOp = moduleBuilder.create<IREE::HAL::ExecutableOp>(
        moduleOp.getLoc(), linkedExecutableName);
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
              : llvm::formatv("{0}_{1}", targetAttr.getSymbolNameFragment(),
                              index);
      auto linkedTargetOp =
          executableBuilder.create<IREE::HAL::ExecutableVariantOp>(
              moduleOp.getLoc(), linkedVariantName, targetAttr);
      auto targetBuilder = OpBuilder::atBlockBegin(&linkedTargetOp.getBlock());
      targetBuilder.create<mlir::ModuleOp>(moduleOp.getLoc());

      auto mergeModuleFn = [](mlir::ModuleOp sourceInnerModule,
                              mlir::ModuleOp linkedInnerModule,
                              DenseMap<StringRef, Operation *> &symbolMap) {
        return mergeModuleInto(sourceInnerModule, linkedInnerModule, symbolMap,
                               allowRenamingPrivateLLVMSymbols);
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
