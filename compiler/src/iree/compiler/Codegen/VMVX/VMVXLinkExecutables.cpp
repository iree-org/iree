// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/LinkingUtils.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VMVXLINKEXECUTABLESPASS
#include "iree/compiler/Codegen/VMVX/Passes.h.inc"

namespace {

static LogicalResult linkAllTargetExecutables(StringRef target, bool lazy,
                                              ModuleOp moduleOp,
                                              OpBuilder &moduleBuilder) {
  auto sourceExecutableOps = gatherExecutablesForTarget(moduleOp, target);
  if (sourceExecutableOps.size() <= 1) {
    return success();
  }

  // Guess a module name, if needed, to make the output files readable.
  auto moduleName = guessModuleName(moduleOp, "module");

  // Create our new "linked" hal.executable.
  std::string linkedExecutableName =
      llvm::formatv("{}_linked_{}", moduleName, target);
  auto linkedExecutableOp = IREE::HAL::ExecutableOp::create(
      moduleBuilder, moduleOp.getLoc(), linkedExecutableName);
  linkedExecutableOp.setVisibility(sourceExecutableOps.front().getVisibility());
  auto executableBuilder =
      OpBuilder::atBlockBegin(&linkedExecutableOp.getBlock());

  // Gather all unique executable targets - we may have multiple.
  auto executableTargetAttrs = gatherExecutableTargets(sourceExecutableOps);
  for (auto [index, targetAttr] : llvm::enumerate(executableTargetAttrs)) {
    // Add our VMVX hal.executable.variant with an empty module.
    std::string linkedVariantName =
        executableTargetAttrs.size() == 1
            ? targetAttr.getSymbolNameFragment()
            : llvm::formatv("{}_{}", targetAttr.getSymbolNameFragment(), index);
    auto linkedTargetOp = IREE::HAL::ExecutableVariantOp::create(
        executableBuilder, moduleOp.getLoc(), linkedVariantName, targetAttr);
    auto targetBuilder = OpBuilder::atBlockBegin(&linkedTargetOp.getBlock());
    auto linkedModuleOp = ModuleOp::create(targetBuilder, moduleOp.getLoc());

    // Add an empty vm.module to that module as our vm.funcs must live in it.
    auto nestedBuilder = OpBuilder::atBlockBegin(linkedModuleOp.getBody());
    IREE::VM::ModuleOp::create(nestedBuilder, moduleOp.getLoc(),
                               "linked_module");

    auto mergeModuleFn = [](mlir::ModuleOp sourceInnerModule,
                            mlir::ModuleOp linkedInnerModule,
                            DenseMap<StringRef, Operation *> &symbolMap) {
      auto srcModule = sourceInnerModule.getOps<IREE::VM::ModuleOp>().begin();
      auto dstModule = linkedInnerModule.getOps<IREE::VM::ModuleOp>().begin();
      return mergeModuleInto(*srcModule, *dstModule, symbolMap);
    };

    // Try linking together all executable variants for this target.
    if (failed(linkExecutablesInto(moduleOp, sourceExecutableOps,
                                   linkedExecutableOp, linkedTargetOp,
                                   mergeModuleFn))) {
      return failure();
    }
  }

  // Remove if we didn't add anything.
  if (linkedExecutableOp.getBody().empty()) {
    linkedExecutableOp.erase();
  }
  return success();
}

struct VMVXLinkExecutablesPass
    : public impl::VMVXLinkExecutablesPassBase<VMVXLinkExecutablesPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    StringRef target = "vmvx";
    if (failed(linkAllTargetExecutables(target, /*lazy=*/false, moduleOp,
                                        moduleBuilder))) {
      return signalPassFailure();
    }
    if (failed(linkAllTargetExecutables(target, /*lazy=*/true, moduleOp,
                                        moduleBuilder))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
