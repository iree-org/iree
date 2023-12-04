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
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
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

    // Retain only non-external source executables. Linking right now happens as
    // placing spirv.module ops into the same hal.executable.variant ops.
    // External source executables won't have any spirv.modules inside.
    int retainSize = 0;
    for (int i = 0, e = sourceExecutableOps.size(); i < e; ++i) {
      IREE::HAL::ExecutableOp executable = sourceExecutableOps[i];
      if (llvm::all_of(executable.getOps<IREE::HAL::ExecutableVariantOp>(),
                       [](auto op) { return !op.getObjects().has_value(); })) {
        sourceExecutableOps[retainSize++] = executable;
      }
    }
    sourceExecutableOps.resize(retainSize);

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
    for (auto [index, attr] : llvm::enumerate(executableTargetAttrs)) {
      // Add our hal.executable.variant with an empty module.
      std::string linkedVariantName =
          executableTargetAttrs.size() == 1
              ? attr.getSymbolNameFragment()
              : llvm::formatv("{0}_{1}", attr.getSymbolNameFragment(), index);
      auto linkedTargetOp =
          executableBuilder.create<IREE::HAL::ExecutableVariantOp>(
              moduleOp.getLoc(), linkedVariantName, attr);
      auto targetBuilder = OpBuilder::atBlockBegin(&linkedTargetOp.getBlock());
      targetBuilder.create<mlir::ModuleOp>(moduleOp.getLoc());

      auto mergeModuleFn = [](mlir::ModuleOp sourceInnerModule,
                              mlir::ModuleOp linkedInnerModule,
                              DenseMap<StringRef, Operation *> &symbolMap) {
        // spirv.module is isolated from above. It does not define symbols or
        // reference outside symbols too. So we can just simply move it to the
        // linked inner module.
        auto srcModules = sourceInnerModule.getOps<spirv::ModuleOp>();
        assert(std::distance(srcModules.begin(), srcModules.end()) == 1);
        Operation *srcModule = *srcModules.begin();
        Block &targetBlock = *linkedInnerModule->getRegion(0).begin();
        if (!targetBlock.empty()) {
          srcModule->moveAfter(&targetBlock.back());
        } else {
          srcModule->moveBefore(&targetBlock, targetBlock.end());
        }
        return success();
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

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createSPIRVLinkExecutablesPass() {
  return std::make_unique<SPIRVLinkExecutablesPass>();
}

} // namespace mlir::iree_compiler
